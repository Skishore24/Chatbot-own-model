"""
backend/app/llm/config.py
----------------------------------------------------
Neural LLM Configuration and Hyperparameter dataclasses.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPTConfig:
    vocab_size: int = 10000
    block_size: int = 512
    n_embd: int = 384
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 2
    dropout: float = 0.10
    bias: bool = False
    rope_freq_base: float = 10000.0
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def intermediate_dim(self) -> int:
        # SwiGLU 8/3 multiplier convention
        return int(2 * (4 * self.n_embd) / 3)
