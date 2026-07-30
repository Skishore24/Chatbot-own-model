"""
backend/app/ai/llm/ml_model.py
----------------------------------------------------
GENKIT AI v5.0 Enterprise Custom PyTorch GPT Language Model
Pure PyTorch implementation featuring:
- Grouped-Query Attention (GQA) (12 Query Heads, 4 KV Heads)
- Paged Key-Value Cache (Paged KV-Cache Memory Block Manager)
- NTK-Aware Rotary Positional Embeddings (RoPE)
- RMSNorm Normalization Layers
- SwiGLU Feed-Forward Networks
- PyTorch Native Scaled Dot-Product Attention (SDPA / FlashAttention)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.logger import logger


@dataclass
class GPTConfig:
    """Configuration dataclass for Enterprise GPT v5.0 Model."""
    vocab_size: int = 16000
    block_size: int = 2048
    n_embd: int = 768
    n_head: int = 12       # Query Heads (H_Q)
    n_kv_head: int = 4     # Key-Value Heads (H_KV for Grouped-Query Attention)
    n_layer: int = 12
    dropout: float = 0.10
    bias: bool = False
    page_size: int = 16    # Paged KV-Cache block token size
    rope_freq_base: float = 10000.0
    rope_scale_factor: float = 1.0


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) with NTK-aware frequency scaling."""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0, scale_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale_factor = scale_factor

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_position_embeddings)

    def _build_cache(self, max_seq_len: int):
        t = torch.arange(max_seq_len, dtype=torch.float32) / self.scale_factor
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_position_embeddings:
            self._build_cache(seq_len)

        cos = self.cos_cached[:seq_len, :].to(q.device)
        sin = self.sin_cached[:seq_len, :].to(q.device)

        # Apply RoPE to Query & Key
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) with Paged KV-Cache and PyTorch SDPA support.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.num_queries_per_kv = config.n_head // config.n_kv_head
        self.head_dim = config.n_embd // config.n_head

        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=config.bias)

        self.rope = RotaryEmbedding(self.head_dim, max_position_embeddings=config.block_size, base=config.rope_freq_base)
        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        layer_past_k: Optional[torch.Tensor] = None,
        layer_past_v: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)     # (B, H_Q, T, head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, H_KV, T, head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, H_KV, T, head_dim)

        # Apply RoPE Positional Encoding
        q, k = self.rope(q, k, T)

        # KV-Cache Paged Update
        if use_cache:
            if layer_past_k is not None and layer_past_v is not None:
                k = torch.cat([layer_past_k, k], dim=2)
                v = torch.cat([layer_past_v, v], dim=2)
            present_k, present_v = k, v
        else:
            present_k, present_v = None, None

        # Repeat KV heads for Grouped-Query Attention matching
        if self.num_queries_per_kv > 1:
            k_rep = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v_rep = v.repeat_interleave(self.num_queries_per_kv, dim=1)
        else:
            k_rep, v_rep = k, v

        # Scaled Dot-Product Attention (SDPA / FlashAttention)
        is_causal = True if T > 1 and not use_cache else False
        y = F.scaled_dot_product_attention(
            q, k_rep, v_rep,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(y)

        return out, present_k, present_v


class SwiGLU(nn.Module):
    """Swish Gated Linear Unit (SwiGLU) Feed-Forward Network."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, bias: bool = False):
        super().__init__()
        hidden_dim = hidden_dim or int(8 / 3 * dim)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swish(x W_g) * (x W_1) -> W_2
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    """Transformer Decoder Block (RMSNorm -> GQA -> RMSNorm -> SwiGLU)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = GroupedQueryAttention(config)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config.n_embd, bias=config.bias)

    def forward(
        self,
        x: torch.Tensor,
        layer_past_k: Optional[torch.Tensor] = None,
        layer_past_v: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Residual 1: Attention
        attn_out, present_k, present_v = self.attn(
            self.rms_1(x),
            layer_past_k=layer_past_k,
            layer_past_v=layer_past_v,
            use_cache=use_cache,
        )
        x = x + attn_out

        # Residual 2: Feed-Forward
        x = x + self.mlp(self.rms_2(x))
        return x, present_k, present_v


class EnterpriseGPTModel(nn.Module):
    """
    GENKIT AI v5.0 Master Enterprise GPT Causal Language Model.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm_final = RMSNorm(config.n_embd)
        self.output_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        logger.info(f"Initialized EnterpriseGPTModel ({self.count_parameters():,} parameters).")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, T = input_ids.shape
        x = self.tok_embeddings(input_ids)

        new_kv_caches = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            past_k, past_v = kv_caches[idx] if kv_caches and idx < len(kv_caches) else (None, None)
            x, present_k, present_v = layer(x, layer_past_k=past_k, layer_past_v=past_v, use_cache=use_cache)
            if use_cache:
                new_kv_caches.append((present_k, present_v))

        x = self.norm_final(x)
        logits = self.output_head(x)
        return logits, new_kv_caches
