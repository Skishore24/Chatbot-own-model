"""
backend/app/llm/positional.py
----------------------------------------------------
Rotary Position Embedding (RoPE) with correct offset handling for cached decoding.
"""

from typing import Tuple
import torch
import torch.nn as nn


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precomputes frequency tensor for complex rotary embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64 [end, dim // 2]
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary embeddings to query and key tensors with position offset.
    xq: [B, seq_len, H_q, head_dim]
    xk: [B, seq_len, H_kv, head_dim]
    freqs_cis: [max_seq_len, head_dim // 2]
    offset: starting position for cached autoregressive tokens (past_length).
    """
    seq_len = xq.shape[1]

    # Reshape queries and keys into complex numbers: [B, seq_len, H, head_dim // 2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Slice freqs_cis from offset to offset + seq_len: [seq_len, head_dim // 2]
    freqs_cis = freqs_cis[offset : offset + seq_len].to(xq.device)
    freqs_cis = freqs_cis.view(1, seq_len, 1, -1)

    # Apply complex multiplication
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
