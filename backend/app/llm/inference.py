"""
backend/app/llm/inference.py
----------------------------------------------------
Model loader, strict checkpoint verification, and inference runtime setup.
Provides explicit ModelStatus states:
- MODEL_LOADING
- MODEL_READY
- MODEL_NOT_FOUND
- MODEL_INVALID
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
from app.llm.tokenizer import ByteFallbackBPETokenizer
from app.llm.checkpoint import CheckpointManager


class ModelStatus:
    LOADING = "MODEL_LOADING"
    READY = "MODEL_READY"
    NOT_FOUND = "MODEL_NOT_FOUND"
    NOT_TRAINED = "MODEL_NOT_TRAINED"  # Alias for backward compatibility
    INVALID = "MODEL_INVALID"
    INCOMPATIBLE = "MODEL_INCOMPATIBLE"
    ERROR = "MODEL_ERROR"


NOT_FOUND_MESSAGE = "Custom model checkpoint not found on disk. Please train the model."
INVALID_MESSAGE = "Custom model checkpoint is corrupted or invalid."
INCOMPATIBLE_MESSAGE = "Custom model checkpoint architecture does not match configuration."


def get_inference_device() -> torch.device:
    """Selects best available compute device for inference (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(
    model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    config_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[Optional[EnterpriseGPTModel], ByteFallbackBPETokenizer, Optional[GPTConfig], str]:
    """
    Loads model and tokenizer checkpoints with strict configuration and integrity verification.
    Returns: (model, tokenizer, config, model_status)
    Never runs untrained random weights or corrupted checkpoints for generation.
    """
    dev = torch.device(device) if device else get_inference_device()
    m_path = Path(model_path or settings.MODEL_CHECKPOINT_PATH)
    t_path = Path(tokenizer_path or settings.TOKENIZER_CHECKPOINT_PATH)
    c_path = Path(config_path or settings.CONFIG_CHECKPOINT_PATH)

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
        logger.warning(f"Model checkpoint not found at {m_path}. Status: {ModelStatus.NOT_FOUND}")
        return None, tokenizer, None, ModelStatus.NOT_FOUND

    # 3. Load Expected Configuration from config_v6.json if available
    expected_config: Optional[GPTConfig] = None
    if c_path.exists():
        try:
            expected_config = GPTConfig.load_from_file(str(c_path))
        except Exception as e:
            logger.warning(f"Could not load config file {c_path}: {e}")

    # 4. Verify Checkpoint File Integrity before full load
    is_valid, reason = CheckpointManager.verify_checkpoint(str(m_path), expected_config=expected_config)
    if not is_valid:
        if "corrupted" in reason.lower() or "zip" in reason.lower() or "failed" in reason.lower():
            logger.error(f"MODEL_INVALID: Checkpoint is corrupted at {m_path}: {reason}")
            return None, tokenizer, None, ModelStatus.INVALID
        else:
            logger.error(f"MODEL_INCOMPATIBLE: Checkpoint incompatibility at {m_path}: {reason}")
            return None, tokenizer, None, ModelStatus.INCOMPATIBLE

    # 5. Load and instantiate model
    try:
        checkpoint = torch.load(str(m_path), map_location=dev, weights_only=False)

        # Extract config
        if isinstance(checkpoint, dict) and "config" in checkpoint and checkpoint["config"]:
            raw_cfg = checkpoint["config"]
            if isinstance(raw_cfg, dict):
                config = GPTConfig.from_dict(raw_cfg)
            elif isinstance(raw_cfg, GPTConfig):
                config = raw_cfg
            else:
                config = expected_config or GPTConfig(vocab_size=tokenizer.vocab_size)
        elif expected_config is not None:
            config = expected_config
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

        # Verify tokenizer vocab matches config vocab
        if tokenizer.vocab_size != config.vocab_size:
            logger.error(
                f"MODEL_INCOMPATIBLE: Tokenizer vocab ({tokenizer.vocab_size}) != Model config vocab ({config.vocab_size})"
            )
            return None, tokenizer, None, ModelStatus.INCOMPATIBLE

        model = EnterpriseGPTModel(config)

        # Extract state dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and any(k.startswith("tok_embeddings") or k.startswith("layers") for k in checkpoint.keys()):
            state_dict = checkpoint
        else:
            raise ValueError("Checkpoint missing valid model_state_dict.")

        # Strict weight loading
        model.load_state_dict(state_dict, strict=True)
        model.to(dev)
        model.eval()

        logger.info(
            f"Loaded Trained Model Checkpoint: {m_path.name} "
            f"({model.count_parameters():,} parameters). Status: {ModelStatus.READY}"
        )
        return model, tokenizer, config, ModelStatus.READY

    except Exception as e:
        logger.error(f"MODEL_ERROR: Unexpected error loading model at {m_path}: {e}")
        return None, tokenizer, None, ModelStatus.ERROR
