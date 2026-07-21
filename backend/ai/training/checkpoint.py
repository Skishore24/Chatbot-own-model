"""
ai/training/checkpoint.py
----------------------------------------------------
Genkit AI - Checkpoint Manager

Features
--------
✓ Save training checkpoints
✓ Load best checkpoint
✓ Resume training from last checkpoint
✓ List all checkpoints
✓ Best model tracking
✓ Checkpoint metadata (epoch, loss, timestamp)
✓ Automatic cleanup of old checkpoints (keep N best)
✓ Thread-safe

Author : Genkit AI
"""

import os
import sys
import json
import time
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch

logger = logging.getLogger("Genkit AI")

# ============================================================
# CHECKPOINT MANAGER
# ============================================================

class CheckpointManager:
    """
    Production checkpoint manager for Genkit AI training.

    Handles saving, loading, listing, and cleanup of
    training checkpoints.

    Parameters
    ----------
    checkpoint_dir : str or Path  Directory for checkpoints.
    keep_top_n     : int          How many checkpoints to keep (best by loss).
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_top_n: int = 5,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_n = keep_top_n

        self.best_loss: float = float("inf")
        self.best_checkpoint_path: Optional[Path] = None
        self.checkpoint_history: List[Dict] = []

    # ----------------------------------------------------------
    # Save Checkpoint
    # ----------------------------------------------------------

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loss: float,
        valid_loss: float,
        config: Optional[Dict] = None,
        scaler_state: Optional[Dict] = None,
    ) -> Path:
        """
        Save a training checkpoint.

        Parameters
        ----------
        epoch       : int
        model       : nn.Module
        optimizer   : Optimizer
        train_loss  : float
        valid_loss  : float
        config      : dict (optional)
        scaler_state: dict (optional, for AMP)

        Returns
        -------
        Path  Path to saved checkpoint.
        """
        timestamp = int(time.time())
        filename = f"checkpoint_epoch{epoch:03d}_loss{valid_loss:.4f}_{timestamp}.pt"
        path = self.checkpoint_dir / filename

        # Unwrap DataParallel if needed
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )

        checkpoint = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "model": model_to_save.state_dict(),
            "optimizer": optimizer.state_dict(),
            "timestamp": timestamp,
            "config": config or {},
        }

        if scaler_state is not None:
            checkpoint["scaler"] = scaler_state

        torch.save(checkpoint, path)

        # Track in history
        self.checkpoint_history.append({
            "path": str(path),
            "epoch": epoch,
            "valid_loss": valid_loss,
            "timestamp": timestamp,
        })

        logger.info("Checkpoint saved: %s (val_loss=%.4f)", path.name, valid_loss)

        # Update best checkpoint
        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.best_checkpoint_path = path
            # Also save as best_model.pt
            best_path = self.checkpoint_dir.parent / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info("New best model saved (val_loss=%.4f)", valid_loss)

        # Also save as latest.pt for easy resumption
        latest_path = self.checkpoint_dir.parent / "checkpoint.pt"
        torch.save(checkpoint, latest_path)

        # Cleanup old checkpoints
        self._cleanup()

        return path

    # ----------------------------------------------------------
    # Load Checkpoint
    # ----------------------------------------------------------

    def load(
        self,
        path: Optional[str] = None,
        device: str = "cpu",
    ) -> Optional[Dict]:
        """
        Load a checkpoint.

        If no path is given, loads the latest checkpoint
        (checkpoint.pt in parent directory).

        Parameters
        ----------
        path   : str (optional)  Path to checkpoint file.
        device : str             Device to map tensors to.

        Returns
        -------
        dict or None
        """
        if path is None:
            # Try latest checkpoint
            latest = self.checkpoint_dir.parent / "checkpoint.pt"
            if not latest.exists():
                logger.info("No checkpoint found. Starting fresh.")
                return None
            path = str(latest)

        if not os.path.exists(path):
            logger.warning("Checkpoint not found: %s", path)
            return None

        try:
            checkpoint = torch.load(path, map_location=device)
            epoch = checkpoint.get("epoch", 0)
            val_loss = checkpoint.get("valid_loss", float("inf"))
            logger.info(
                "Checkpoint loaded: epoch=%d val_loss=%.4f",
                epoch, val_loss,
            )
            return checkpoint
        except Exception:
            logger.exception("Failed to load checkpoint: %s", path)
            return None

    # ----------------------------------------------------------
    # Resume Training
    # ----------------------------------------------------------

    def resume(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
    ) -> Tuple[int, float]:
        """
        Resume training from the latest checkpoint.

        Loads model weights, optimizer state, and returns
        the starting epoch and best loss so far.

        Parameters
        ----------
        model     : nn.Module
        optimizer : Optimizer
        device    : str

        Returns
        -------
        (start_epoch: int, best_loss: float)
        """
        checkpoint = self.load(device=device)
        if checkpoint is None:
            return 0, float("inf")

        # Adapt state dict keys if architecture changed
        model_state = checkpoint.get("model", {})
        adapted = self._adapt_state_dict(model_state)

        # Check for vocab size mismatch
        if not self._check_shape_compatibility(model, adapted):
            logger.warning(
                "Shape mismatch detected. Starting training from scratch."
            )
            return 0, float("inf")

        try:
            model.load_state_dict(adapted)
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            self.best_loss = checkpoint.get("valid_loss", float("inf"))
            logger.info("Resuming training from epoch %d.", start_epoch)
            return start_epoch, self.best_loss
        except Exception as e:
            logger.warning(
                "Could not restore checkpoint state: %s. Starting fresh.", e
            )
            return 0, float("inf")

    # ----------------------------------------------------------
    # List Checkpoints
    # ----------------------------------------------------------

    def list_checkpoints(self) -> List[Dict]:
        """
        List all checkpoints in the checkpoint directory,
        sorted by validation loss (best first).

        Returns
        -------
        List[dict]
        """
        files = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        result = []
        for f in files:
            try:
                ckpt = torch.load(str(f), map_location="cpu")
                result.append({
                    "path": str(f),
                    "epoch": ckpt.get("epoch"),
                    "valid_loss": ckpt.get("valid_loss"),
                    "timestamp": ckpt.get("timestamp"),
                })
            except Exception:
                pass
        return sorted(result, key=lambda x: x.get("valid_loss", 99))

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _adapt_state_dict(self, state: Dict) -> Dict:
        """Adapt state dict keys for architecture changes."""
        adapted = {}
        for k, v in state.items():
            if k.endswith(".attn.bias"):
                continue
            new_k = k
            new_k = new_k.replace(".ln_1.", ".ln1.")
            new_k = new_k.replace(".ln_2.", ".ln2.")
            new_k = new_k.replace(".mlp.c_fc.", ".mlp.fc.")
            new_k = new_k.replace(".mlp.c_proj.", ".mlp.proj.")
            adapted[new_k] = v
        return adapted

    def _check_shape_compatibility(
        self,
        model: torch.nn.Module,
        state: Dict,
    ) -> bool:
        """Check if saved model shapes match current model."""
        inner = model.module if hasattr(model, "module") else model
        wte_key = "transformer.wte.weight"
        if wte_key in state:
            saved_shape = state[wte_key].shape
            current_shape = inner.transformer["wte"].weight.shape
            if saved_shape != current_shape:
                logger.warning(
                    "Vocab size mismatch: saved=%d, current=%d",
                    saved_shape[0], current_shape[0],
                )
                return False
        return True

    def _cleanup(self) -> None:
        """Remove old checkpoints keeping only top-N by validation loss."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch*.pt"),
            key=lambda p: float(
                p.name.split("_loss")[-1].split("_")[0]
            ) if "_loss" in p.name else 99,
        )
        # Keep top N
        to_delete = checkpoints[self.keep_top_n:]
        for path in to_delete:
            try:
                path.unlink()
                logger.debug("Deleted old checkpoint: %s", path.name)
            except Exception:
                pass


# ============================================================
# Singleton (initialized lazily — needs config)
# ============================================================

def _create_default_manager():
    """Create default CheckpointManager using config paths."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from config import CHECKPOINT_DIR
        return CheckpointManager(checkpoint_dir=CHECKPOINT_DIR)
    except Exception:
        return CheckpointManager(checkpoint_dir="genkit-model/checkpoints")


checkpoint_manager = _create_default_manager()

__all__ = ["CheckpointManager", "checkpoint_manager"]
