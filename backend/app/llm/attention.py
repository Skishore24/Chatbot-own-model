"""
backend/app/llm/attention.py
----------------------------------------------------
Cache-Aware Causal Grouped-Query Attention (GQA) with RoPE and KV-Cache.
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
    """Cache-Aware Causal Grouped-Query Attention (GQA) with RoPE and KV-Cache."""

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
        attention_mask: Optional[torch.Tensor] = None,
        past_k: Optional[torch.Tensor] = None,
        past_v: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        x: [B, seq_len, n_embd]
        attention_mask: [B, total_k_len] or [B, 1, seq_len, total_k_len]
        past_k: [B, past_len, n_kv_head, head_dim]
        past_v: [B, past_len, n_kv_head, head_dim]
        """
        B, seq_len, _ = x.shape
        offset = past_k.shape[1] if past_k is not None else 0

        # 1. Project Q, K, V
        q = self.wq(x).view(B, seq_len, self.n_head, self.head_dim)
        k = self.wk(x).view(B, seq_len, self.n_kv_head, self.head_dim)
        v = self.wv(x).view(B, seq_len, self.n_kv_head, self.head_dim)

        # 2. Apply RoPE with position offset for cached decoding
        q, k = apply_rotary_emb(q, k, freqs_cis, offset=offset)

        # 3. Update KV cache
        if past_k is not None:
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        present_k = k if use_cache else None
        present_v = v if use_cache else None

        # 4. Repeat KV heads for GQA
        k_rep = repeat_kv(k, self.n_rep)
        v_rep = repeat_kv(v, self.n_rep)

        # 5. Reshape for Scaled Dot-Product Attention: [B, H, seq_len, head_dim]
        q_t = q.transpose(1, 2)
        k_t = k_rep.transpose(1, 2)
        v_t = v_rep.transpose(1, 2)
        total_k_len = k_t.shape[-2]

        # 6. Explicit Cache-Aware Causal & Padding Mask Construction
        # A query at offset + i can attend to keys at positions <= offset + i
        q_pos = torch.arange(offset, offset + seq_len, device=x.device).unsqueeze(1)
        k_pos = torch.arange(total_k_len, device=x.device).unsqueeze(0)
        causal_mask = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, total_k_len]

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_mask = (attention_mask != 0).unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:
                pad_mask = (attention_mask != 0).unsqueeze(1)
            else:
                pad_mask = (attention_mask != 0)
            combined_mask = causal_mask & pad_mask
        else:
            combined_mask = causal_mask

        # Fast path if all True (e.g. single-token decode with full valid past cache)
        if combined_mask.all():
            attn_mask = None
            is_causal = False
        elif seq_len == total_k_len and attention_mask is None and offset == 0:
            attn_mask = None
            is_causal = True
        else:
            attn_mask = combined_mask
            is_causal = False

        attn_out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=attn_mask,
            dropout_p=self.config.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # 7. Transpose back: [B, seq_len, H * head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, seq_len, self.n_head * self.head_dim)
        output = self.dropout(self.wo(attn_out))

        return output, present_k, present_v
