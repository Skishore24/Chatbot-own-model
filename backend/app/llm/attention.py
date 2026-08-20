"""
backend/app/llm/attention.py
----------------------------------------------------
Causal Grouped-Query Attention (GQA) with KV-Cache & RoPE.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.llm.config import GPTConfig
from app.llm.positional import apply_rotary_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats Key/Value heads when H_q != H_kv for Grouped-Query Attention."""
    if n_rep == 1:
        return x
    B, seq_len, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(B, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(B, seq_len, n_kv_heads * n_rep, head_dim)
    )


class CausalGroupedQueryAttention(nn.Module):
    """Causal Grouped-Query Attention (GQA) with RoPE and KV-Cache."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = self.n_head // self.n_kv_head
        self.head_dim = config.head_dim

        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=config.bias)
        self.wk = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.wv = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.wo = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_k: Optional[torch.Tensor] = None,
        past_v: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        x: [B, seq_len, n_embd]
        past_k: [B, past_len, n_kv_head, head_dim]
        past_v: [B, past_len, n_kv_head, head_dim]
        """
        B, seq_len, _ = x.shape
        offset = past_k.shape[1] if past_k is not None else 0

        # Project Q, K, V
        q = self.wq(x).view(B, seq_len, self.n_head, self.head_dim)
        k = self.wk(x).view(B, seq_len, self.n_kv_head, self.head_dim)
        v = self.wv(x).view(B, seq_len, self.n_kv_head, self.head_dim)

        # Apply RoPE with position offset for cached decoding
        q, k = apply_rotary_emb(q, k, freqs_cis, offset=offset)

        # Update KV cache
        if past_k is not None:
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        present_k = k if use_cache else None
        present_v = v if use_cache else None

        # Repeat KV heads for GQA if necessary
        k_rep = repeat_kv(k, self.n_rep)
        v_rep = repeat_kv(v, self.n_rep)

        # Reshape for PyTorch Scaled Dot Product Attention: [B, H, seq_len, head_dim]
        q_t = q.transpose(1, 2)
        k_t = k_rep.transpose(1, 2)
        v_t = v_rep.transpose(1, 2)

        # Determine causal mask
        # If seq_len == 1 and we have past tokens in cache, no future tokens exist (is_causal=False)
        is_causal = (seq_len > 1)

        attn_out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            dropout_p=self.config.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # Transpose back: [B, seq_len, H * head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, seq_len, self.n_head * self.head_dim)
        output = self.dropout(self.wo(attn_out))

        return output, present_k, present_v
