"""
backend/train.py
----------------------------------------------------
Genkit AI V6 Master Model Training Launcher.
"""

import sys
import argparse
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from training.train_model import train_pipeline


def main():
    parser = argparse.ArgumentParser(description="Genkit AI V6 Master Model Training Pipeline")
    parser.add_argument("--epochs", type=int, default=settings.EPOCHS, help="Training epochs (default: 60)")
    parser.add_argument("--batch-size", type=int, default=settings.BATCH_SIZE, help="Micro-batch size (default: 4)")
    parser.add_argument("--accum-steps", type=int, default=settings.GRADIENT_ACCUMULATION_STEPS, help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--block-size", type=int, default=settings.BLOCK_SIZE, help="Sequence block size (default: 512)")
    parser.add_argument("--vocab-size", type=int, default=settings.VOCAB_SIZE, help="Vocabulary size (default: 10000)")
    parser.add_argument("--lr", type=float, default=settings.LEARNING_RATE, help="Peak learning rate (default: 3e-4)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    train_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
