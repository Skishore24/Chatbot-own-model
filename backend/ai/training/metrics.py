"""
ai/training/metrics.py
----------------------------------------------------
Genkit AI - Training Metrics Tracker

Features
--------
✓ Per-epoch loss tracking (train + validation)
✓ Perplexity computation
✓ Best epoch tracking
✓ Training summary
✓ Learning rate history
✓ Gradient norm history
✓ Export to JSON

Author : Genkit AI
"""

import json
import math
import time
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger("Genkit AI")

# ============================================================
# METRICS TRACKER
# ============================================================

class TrainingMetrics:
    """
    Tracks and summarizes training metrics for Genkit AI.

    Usage
    -----
    metrics = TrainingMetrics()
    metrics.record_epoch(epoch=1, train_loss=2.5, valid_loss=2.3, lr=1e-4)
    print(metrics.summary())
    """

    def __init__(self) -> None:
        self.epochs: List[Dict] = []
        self.best_epoch: Optional[int] = None
        self.best_valid_loss: float = float("inf")
        self.start_time: float = time.time()
        self.grad_norms: List[float] = []

    # ----------------------------------------------------------
    # Record epoch
    # ----------------------------------------------------------

    def record_epoch(
        self,
        epoch: int,
        train_loss: float,
        valid_loss: float,
        lr: float = 0.0,
        tokens_per_second: float = 0.0,
        elapsed_seconds: float = 0.0,
    ) -> None:
        """
        Record metrics for one training epoch.

        Parameters
        ----------
        epoch           : int
        train_loss      : float
        valid_loss      : float
        lr              : float  Current learning rate.
        tokens_per_second : float
        elapsed_seconds : float
        """
        train_perplexity = self.loss_to_perplexity(train_loss)
        valid_perplexity = self.loss_to_perplexity(valid_loss)

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "valid_loss": round(valid_loss, 4),
            "train_perplexity": round(train_perplexity, 2),
            "valid_perplexity": round(valid_perplexity, 2),
            "lr": lr,
            "tokens_per_second": round(tokens_per_second, 1),
            "elapsed_seconds": round(elapsed_seconds, 1),
            "timestamp": int(time.time()),
        }

        self.epochs.append(record)

        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            self.best_epoch = epoch

    # ----------------------------------------------------------
    # Record gradient norm
    # ----------------------------------------------------------

    def record_grad_norm(self, norm: float) -> None:
        """Record gradient norm (called per batch)."""
        self.grad_norms.append(norm)

    # ----------------------------------------------------------
    # Perplexity
    # ----------------------------------------------------------

    @staticmethod
    def loss_to_perplexity(loss: float) -> float:
        """
        Compute perplexity from cross-entropy loss.

        Perplexity = e^loss
        Lower is better. A random model over 10k vocab has ~10000 ppl.
        """
        try:
            return math.exp(min(loss, 20.0))  # Cap to avoid overflow
        except (ValueError, OverflowError):
            return float("inf")

    # ----------------------------------------------------------
    # Latest metrics
    # ----------------------------------------------------------

    @property
    def latest(self) -> Optional[Dict]:
        """Return metrics from the most recent epoch."""
        if not self.epochs:
            return None
        return self.epochs[-1]

    @property
    def total_elapsed(self) -> float:
        """Total training time in seconds."""
        return time.time() - self.start_time

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    def summary(self) -> str:
        """
        Print a formatted training summary.

        Returns
        -------
        str  Formatted text summary.
        """
        if not self.epochs:
            return "No epochs recorded yet."

        lines = [
            "=" * 60,
            "  GENKIT AI — TRAINING SUMMARY",
            "=" * 60,
        ]

        lines.append(
            f"  {'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | "
            f"{'Train PPL':>10} | {'Val PPL':>10} | {'LR':>10}"
        )
        lines.append("-" * 60)

        for ep in self.epochs:
            marker = " ★" if ep["epoch"] == self.best_epoch else ""
            lines.append(
                f"  {ep['epoch']:>5} | {ep['train_loss']:>10.4f} | "
                f"{ep['valid_loss']:>10.4f} | {ep['train_perplexity']:>10.2f} | "
                f"{ep['valid_perplexity']:>10.2f} | {ep['lr']:>10.2e}{marker}"
            )

        lines.append("=" * 60)
        lines.append(
            f"  Best Epoch     : {self.best_epoch}"
        )
        lines.append(
            f"  Best Val Loss  : {self.best_valid_loss:.4f}"
        )
        lines.append(
            f"  Best Val PPL   : {self.loss_to_perplexity(self.best_valid_loss):.2f}"
        )
        total_min = self.total_elapsed / 60
        lines.append(
            f"  Total Time     : {total_min:.1f} minutes"
        )

        if self.grad_norms:
            avg_norm = sum(self.grad_norms) / len(self.grad_norms)
            max_norm = max(self.grad_norms)
            lines.append(f"  Avg Grad Norm  : {avg_norm:.3f}")
            lines.append(f"  Max Grad Norm  : {max_norm:.3f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ----------------------------------------------------------
    # Loss history for early stopping check
    # ----------------------------------------------------------

    def is_improving(self, patience: int = 3) -> bool:
        """
        Returns True if there has been an improvement in val loss
        within the last `patience` epochs.
        """
        if len(self.epochs) < patience:
            return True
        recent_losses = [e["valid_loss"] for e in self.epochs[-patience:]]
        return recent_losses[-1] < recent_losses[0]

    # ----------------------------------------------------------
    # Export
    # ----------------------------------------------------------

    def export_json(self, path: str) -> None:
        """
        Export all metrics to a JSON file.

        Parameters
        ----------
        path : str  Output file path.
        """
        data = {
            "epochs": self.epochs,
            "best_epoch": self.best_epoch,
            "best_valid_loss": self.best_valid_loss,
            "total_epochs": len(self.epochs),
            "total_elapsed_seconds": round(self.total_elapsed, 1),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Metrics exported to %s", path)

    # ----------------------------------------------------------
    # Reset
    # ----------------------------------------------------------

    def reset(self) -> None:
        """Reset all metrics for a fresh training run."""
        self.epochs.clear()
        self.best_epoch = None
        self.best_valid_loss = float("inf")
        self.start_time = time.time()
        self.grad_norms.clear()


# ============================================================
# Singleton
# ============================================================
training_metrics = TrainingMetrics()

__all__ = ["TrainingMetrics", "training_metrics"]
