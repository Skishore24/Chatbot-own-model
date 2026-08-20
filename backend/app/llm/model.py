"""
backend/app/llm/model.py
----------------------------------------------------
Enterprise GPT Model Architecture:
- RMSNorm
- Rotary Positional Embeddings (RoPE)
- Cache-Aware Causal Grouped-Query Attention (GQA) with KV Cache & Padding Mask
- SwiGLU Feed-Forward Network
- Weight-tied output projection
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.llm.config import GPTConfig
from app.llm.normalization import RMSNorm
from app.llm.positional import precompute_freqs_cis
from app.llm.attention import CausalGroupedQueryAttention


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network: w2(F.silu(w1(x)) * w3(x))."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = config.intermediate_dim
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Single Transformer Decoder Layer."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = CausalGroupedQueryAttention(config)
        self.rms_2 = RMSNorm(config.n_embd)
        self.ffn = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_k: Optional[torch.Tensor] = None,
        past_v: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Pre-LN Attention
        attn_out, present_k, present_v = self.attn(
            self.rms_1(x),
            freqs_cis,
            attention_mask=attention_mask,
            past_k=past_k,
            past_v=past_v,
            use_cache=use_cache,
        )
        x = x + attn_out

        # Pre-LN Feed-Forward
        x = x + self.ffn(self.rms_2(x))
        return x, present_k, present_v


class EnterpriseGPTModel(nn.Module):
    """Enterprise GPT Decoder-Only Language Model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=config.pad_token_id)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between embedding and final linear projection
        self.tok_embeddings.weight = self.lm_head.weight

        # Precompute RoPE complex frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(config.head_dim, config.block_size * 2, theta=config.rope_freq_base),
            persistent=False,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes weights using truncated normal / standard GPT initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        input_ids: [B, seq_len]
        attention_mask: [B, seq_len] or [B, total_k_len]
        past_key_values: list of (past_k, past_v) for each layer
        returns: (logits [B, seq_len, vocab_size], presents)
        """
        B, seq_len = input_ids.shape

        # Token embeddings
        h = self.dropout(self.tok_embeddings(input_ids))

        presents: List[Tuple[torch.Tensor, torch.Tensor]] = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            past_k, past_v = past_key_values[idx] if past_key_values is not None else (None, None)
            h, present_k, present_v = layer(
                h,
                self.freqs_cis,
                attention_mask=attention_mask,
                past_k=past_k,
                past_v=past_v,
                use_cache=use_cache,
            )
            if use_cache:
                presents.append((present_k, present_v))

        # Final RMS Normalization
        h = self.norm(h)
        logits = self.lm_head(h)

        return logits, presents
