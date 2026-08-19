"""
backend/app/ai/llm/model_loader.py
----------------------------------------------------
GENKIT AI v5.0 Enterprise Model Loader
Thread-safe singleton loader that auto-detects and loads trained checkpoints.
Falls back to a fresh random-weight model if no checkpoint exists.
"""

import threading
from typing import Optional, Tuple

import torch

from app.core.config import settings
from app.core.logger import logger
from app.ai.llm.ml_model import EnterpriseGPTModel, GPTConfig
from app.ai.tokenizer.tokenizer import ByteFallbackBPETokenizer

_lock = threading.Lock()
_model_instance: Optional[EnterpriseGPTModel] = None
_tokenizer_instance: Optional[ByteFallbackBPETokenizer] = None
_config_instance: Optional[GPTConfig] = None


def _default_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=settings.VOCAB_SIZE,
        block_size=settings.BLOCK_SIZE,
        n_embd=settings.EMBED_DIM,
        n_head=settings.NUM_HEADS,
        n_kv_head=settings.NUM_KV_HEADS,
        n_layer=settings.NUM_LAYERS,
        dropout=settings.DROPOUT,
        bias=settings.BIAS,
        page_size=settings.KV_CACHE_PAGE_SIZE,
        rope_freq_base=settings.ROPE_FREQ_BASE,
        rope_scale_factor=settings.ROPE_SCALE_FACTOR,
    )


def load_tokenizer() -> ByteFallbackBPETokenizer:
    global _tokenizer_instance
    with _lock:
        if _tokenizer_instance is not None:
            return _tokenizer_instance

        tokenizer_path = str(settings.TOKENIZER_CHECKPOINT_PATH)
        if settings.tokenizer_checkpoint_exists():
            logger.info(f"Loading trained BPE tokenizer from: {tokenizer_path}")
            _tokenizer_instance = ByteFallbackBPETokenizer.load(tokenizer_path)
        else:
            logger.warning(
                f"Tokenizer checkpoint not found at {tokenizer_path}. "
                "Using untrained base tokenizer (run train.py to generate a trained tokenizer)."
            )
            _tokenizer_instance = ByteFallbackBPETokenizer(vocab_size=settings.VOCAB_SIZE)

        return _tokenizer_instance


def load_model(tokenizer: Optional[ByteFallbackBPETokenizer] = None) -> Tuple[EnterpriseGPTModel, GPTConfig]:
    global _model_instance, _config_instance
    with _lock:
        if _model_instance is not None and _config_instance is not None:
            return _model_instance, _config_instance

        config = _default_config()
        if tokenizer is not None:
            config.vocab_size = tokenizer.vocab_size

        checkpoint_path = str(settings.MODEL_CHECKPOINT_PATH)

        if settings.model_checkpoint_exists():
            logger.info(f"Loading trained model checkpoint from: {checkpoint_path}")
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

                saved_config = checkpoint.get("config")
                if saved_config is not None and isinstance(saved_config, GPTConfig):
                    if tokenizer is None or saved_config.vocab_size == tokenizer.vocab_size:
                        config = saved_config
                        logger.info(f"Restored GPTConfig from checkpoint: {config}")
                    else:
                        logger.warning(
                            f"Checkpoint vocab_size ({saved_config.vocab_size}) != Tokenizer vocab_size ({tokenizer.vocab_size}). "
                            f"Re-aligning model vocab_size to {tokenizer.vocab_size}."
                        )
                        config.vocab_size = tokenizer.vocab_size

                model = EnterpriseGPTModel(config)
                # Load state dict if shapes match
                try:
                    model.load_state_dict(checkpoint["model_state_dict"])
                    logger.info(f"Checkpoint loaded successfully! Parameters: {model.count_parameters():,} | Device: {device}")
                except Exception as shape_err:
                    logger.warning(f"Checkpoint state_dict shape mismatch ({shape_err}). Initializing fresh model with matching vocab size {config.vocab_size}.")
                model.eval()
            except Exception as e:
                logger.error(f"Failed to load checkpoint ({e}). Initializing fresh model.")
                model = EnterpriseGPTModel(config)
        else:
            logger.warning(
                f"Model checkpoint not found at {checkpoint_path}. "
                "Using fresh random-weight model (run train.py to train first)."
            )
            model = EnterpriseGPTModel(config)

        _model_instance = model
        _config_instance = config
        return model, config


def get_model_and_tokenizer() -> Tuple[EnterpriseGPTModel, ByteFallbackBPETokenizer, GPTConfig]:
    tokenizer = load_tokenizer()
    model, config = load_model(tokenizer)
    return model, tokenizer, config
