"""
backend/app/llm/__init__.py
----------------------------------------------------
Public exports for Genkit AI LLM module.
"""

from app.llm.config import GPTConfig
from app.llm.model import EnterpriseGPTModel, TransformerBlock, SwiGLUFFN
from app.llm.tokenizer import ByteFallbackBPETokenizer, default_tokenizer
from app.llm.normalization import RMSNorm
from app.llm.positional import precompute_freqs_cis, apply_rotary_emb
from app.llm.attention import CausalGroupedQueryAttention
from app.llm.inference import load_model_and_tokenizer, get_inference_device
from app.llm.generation import GenerationEngine

__all__ = [
    "GPTConfig",
    "EnterpriseGPTModel",
    "TransformerBlock",
    "SwiGLUFFN",
    "ByteFallbackBPETokenizer",
    "default_tokenizer",
    "RMSNorm",
    "precompute_freqs_cis",
    "apply_rotary_emb",
    "CausalGroupedQueryAttention",
    "load_model_and_tokenizer",
    "get_inference_device",
    "GenerationEngine",
]
