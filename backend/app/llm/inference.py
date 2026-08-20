"""
backend/app/llm/inference.py
----------------------------------------------------
Model loader, strict checkpoint verification, and inference runtime setup.
Provides explicit ModelStatus states: READY, NOT_TRAINED, INCOMPATIBLE.
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
    READY = "MODEL_READY"
    NOT_TRAINED = "MODEL_NOT_TRAINED"
    INCOMPATIBLE = "MODEL_INCOMPATIBLE"
    ERROR = "MODEL_ERROR"


def get_inference_device() -> torch.device:
    """Selects best available compute device for inference."""
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
            "Status: MODEL_NOT_TRAINED. Responses will rely on verified RAG knowledge."
        )
        return None, tokenizer, None, ModelStatus.NOT_TRAINED

    # 3. Load & strictly validate model weights
    try:
        checkpoint = torch.load(str(m_path), map_location=dev, weights_only=False)
        
        # Extract configuration
        if isinstance(checkpoint, dict) and "config" in checkpoint and checkpoint["config"] is not None:
            saved_config = checkpoint["config"]
            if isinstance(saved_config, dict):
                config = GPTConfig(**saved_config)
            else:
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

        model = EnterpriseGPTModel(config)

        # Extract state dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Strict weight loading to guarantee architecture match
        model.load_state_dict(state_dict, strict=True)
        model.to(dev)
        model.eval()

        logger.info(
            f"Loaded Trained Model Checkpoint: {m_path.name} "
            f"({model.count_parameters():,} parameters). Status: MODEL_READY"
        )
        return model, tokenizer, config, ModelStatus.READY

    except Exception as e:
        logger.error(
            f"MODEL CHECKPOINT INCOMPATIBLE or corrupted at {m_path}: {e}. "
            "Status: MODEL_INCOMPATIBLE. Run 'python train.py' to re-train."
        )
        return None, tokenizer, None, ModelStatus.INCOMPATIBLE
