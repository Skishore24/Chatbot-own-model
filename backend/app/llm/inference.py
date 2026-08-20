"""
backend/app/llm/inference.py
----------------------------------------------------
Model loader, strict checkpoint verification, and inference runtime setup.
Provides explicit ModelStatus states:
- MODEL_LOADING
- MODEL_READY
- MODEL_NOT_TRAINED
- MODEL_INCOMPATIBLE
- MODEL_ERROR
"""

from pathlib import Path
from typing import Optional, Tuple
import torch

from app.core.config import settings
from app.core.logger import logger
from app.llm.config import GPTConfig
from app.llm.model import EnterpriseGPTModel
from app.llm.tokenizer import ByteFallbackBPETokenizer, default_tokenizer


class ModelStatus:
    LOADING = "MODEL_LOADING"
    READY = "MODEL_READY"
    NOT_TRAINED = "MODEL_NOT_TRAINED"
    INCOMPATIBLE = "MODEL_INCOMPATIBLE"
    ERROR = "MODEL_ERROR"


NOT_TRAINED_MESSAGE = "Custom model is not trained yet. Please train/load a valid checkpoint."


def get_inference_device() -> torch.device:
    """Selects best available compute device for inference (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(
    model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[Optional[EnterpriseGPTModel], ByteFallbackBPETokenizer, Optional[GPTConfig], str]:
    """
    Loads model and tokenizer checkpoints with strict configuration verification.
    Returns: (model, tokenizer, config, model_status)
    Never runs untrained random weights for generation.
    """
    dev = torch.device(device) if device else get_inference_device()
    m_path = Path(model_path or settings.MODEL_CHECKPOINT_PATH)
    t_path = Path(tokenizer_path or settings.TOKENIZER_CHECKPOINT_PATH)

    # 1. Load Tokenizer
    tokenizer = ByteFallbackBPETokenizer(vocab_size=settings.VOCAB_SIZE)
    if t_path.exists():
        try:
            tokenizer.load(str(t_path))
            logger.info(f"Loaded Tokenizer: {t_path.name} (Vocab Size: {tokenizer.vocab_size:,})")
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {t_path}: {e}")
    else:
        logger.warning(f"Tokenizer checkpoint not found at {t_path}. Using base byte-fallback vocabulary.")

    # 2. Check if model checkpoint exists
    if not m_path.exists():
        logger.warning(
            f"Model checkpoint not found at {m_path}. "
            f"Status: {ModelStatus.NOT_TRAINED}. {NOT_TRAINED_MESSAGE}"
        )
        return None, tokenizer, None, ModelStatus.NOT_TRAINED

    # 3. Load & strictly validate model weights
    try:
        checkpoint = torch.load(str(m_path), map_location=dev, weights_only=False)

        # Extract configuration
        if isinstance(checkpoint, dict) and "config" in checkpoint and checkpoint["config"] is not None:
            saved_config = checkpoint["config"]
            if isinstance(saved_config, dict):
                config = GPTConfig.from_dict(saved_config)
            elif isinstance(saved_config, GPTConfig):
                config = saved_config
            else:
                config = GPTConfig(
                    vocab_size=tokenizer.vocab_size,
                    block_size=settings.BLOCK_SIZE,
                    n_embd=settings.EMBED_DIM,
                    n_layer=settings.NUM_LAYERS,
                    n_head=settings.NUM_HEADS,
                    n_kv_head=settings.NUM_KV_HEADS,
                    dropout=0.0,
                    bias=settings.BIAS,
                    rope_freq_base=settings.ROPE_FREQ_BASE,
                )
        else:
            config = GPTConfig(
                vocab_size=tokenizer.vocab_size,
                block_size=settings.BLOCK_SIZE,
                n_embd=settings.EMBED_DIM,
                n_layer=settings.NUM_LAYERS,
                n_head=settings.NUM_HEADS,
                n_kv_head=settings.NUM_KV_HEADS,
                dropout=0.0,
                bias=settings.BIAS,
                rope_freq_base=settings.ROPE_FREQ_BASE,
            )

        model = EnterpriseGPTModel(config)

        # Extract state dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and any(k.startswith("tok_embeddings") or k.startswith("layers") for k in checkpoint.keys()):
            state_dict = checkpoint
        else:
            raise ValueError("Checkpoint does not contain valid model weights.")

        # Strict weight loading to guarantee architecture match
        model.load_state_dict(state_dict, strict=True)
        model.to(dev)
        model.eval()

        logger.info(
            f"Loaded Trained Model Checkpoint: {m_path.name} "
            f"({model.count_parameters():,} parameters). Status: {ModelStatus.READY}"
        )
        return model, tokenizer, config, ModelStatus.READY

    except Exception as e:
        logger.error(
            f"MODEL_INCOMPATIBLE: checkpoint architecture does not match current configuration or corrupted at {m_path}: {e}. "
            f"Status: {ModelStatus.INCOMPATIBLE}."
        )
        return None, tokenizer, None, ModelStatus.INCOMPATIBLE
