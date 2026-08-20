"""
backend/app/llm/inference.py
----------------------------------------------------
Model loader, checkpoint verification, and inference runtime setup.
"""

from pathlib import Path
from typing import Optional, Tuple
import torch

from app.core.config import settings
from app.core.logger import logger
from app.llm.config import GPTConfig
from app.llm.model import EnterpriseGPTModel
from app.llm.tokenizer import ByteFallbackBPETokenizer, default_tokenizer


def get_inference_device() -> torch.device:
    """Selects best available compute device for inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(
    model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[EnterpriseGPTModel, ByteFallbackBPETokenizer, GPTConfig]:
    """
    Loads model and tokenizer checkpoints from disk with strict verification.
    If no checkpoint exists, initializes an untrained model with logged status.
    """
    dev = torch.device(device) if device else get_inference_device()
    m_path = Path(model_path or settings.MODEL_CHECKPOINT_PATH)
    t_path = Path(tokenizer_path or settings.TOKENIZER_CHECKPOINT_PATH)

    # 1. Load Tokenizer
    tokenizer = ByteFallbackBPETokenizer(vocab_size=settings.VOCAB_SIZE)
    if t_path.exists():
        tokenizer.load(str(t_path))
        logger.info(f"Loaded Tokenizer: {t_path.name} (Vocab Size: {tokenizer.vocab_size:,})")
    else:
        logger.warning(f"Tokenizer checkpoint not found at {t_path}. Using base byte-fallback tokenizer.")

    # 2. Build Model Architecture
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=settings.BLOCK_SIZE,
        n_embd=settings.EMBED_DIM,
        n_layer=settings.NUM_LAYERS,
        n_head=settings.NUM_HEADS,
        n_kv_head=settings.NUM_KV_HEADS,
        dropout=0.0,  # Zero dropout during inference
        bias=settings.BIAS,
        rope_freq_base=settings.ROPE_FREQ_BASE,
    )

    model = EnterpriseGPTModel(config)

    # 3. Load Model Weights if available
    if m_path.exists():
        try:
            checkpoint = torch.load(str(m_path), map_location=dev, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded Trained Model Checkpoint: {m_path.name} ({model.count_parameters():,} parameters)")
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {m_path}: {e}")
    else:
        logger.warning(f"Model checkpoint not found at {m_path}. Initialized untrained model ({model.count_parameters():,} parameters).")

    model.to(dev)
    model.eval()

    if dev.type == "cuda":
        gpu_name = torch.cuda.get_device_name(dev)
        logger.info(f"Model allocated on GPU: {gpu_name}")
    else:
        logger.info("Model allocated on CPU")

    return model, tokenizer, config
