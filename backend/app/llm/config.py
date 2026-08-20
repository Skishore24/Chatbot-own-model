"""
backend/app/llm/config.py
----------------------------------------------------
Neural LLM Configuration and Hyperparameter dataclasses.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional


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

    def to_dict(self) -> Dict[str, Any]:
        """Serializes GPTConfig dataclass to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GPTConfig":
        """Constructs a GPTConfig instance from a dictionary, filtering unknown keys."""
        valid_keys = {
            "vocab_size", "block_size", "n_embd", "n_layer", "n_head",
            "n_kv_head", "dropout", "bias", "rope_freq_base",
            "pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id",
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def save_to_file(self, filepath: str) -> None:
        """Saves configuration parameters to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "GPTConfig":
        """Loads configuration from a JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at: {filepath}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
