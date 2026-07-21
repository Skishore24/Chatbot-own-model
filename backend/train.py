"""
train.py
----------------------------------------------------
Genkit AI - Training Entry Point
Usage
-----
    cd backend
    python train.py

This script trains the custom GPT model on the
Genkit dataset. After training completes:
- model.pt     : best model weights
- vocab.json   : tokenizer vocabulary
- config.json  : model architecture config
- checkpoint.pt: latest training checkpoint

Note
----
On Windows, NUM_WORKERS is set to 0 automatically
to avoid multiprocessing DataLoader issues.
Author : Genkit AI
"""
import os
import sys
from pathlib import Path

# Ensure backend root is in path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import logger

def main():
    logger.info("=" * 70)
    logger.info("GENKIT AI - CUSTOM GPT TRAINER")
    logger.info("=" * 70)
    logger.info("Importing training pipeline...")
    try:
        from ai.llm.train import run_training, set_seed
        logger.info("Starting training...")
        set_seed()
        run_training()
        logger.info("=" * 70)
        logger.info("Training completed successfully!")
        logger.info("Next steps:")
        logger.info("  1. python evaluate.py   — validate the model")
        logger.info("  2. python predict.py    — test interactively")
        logger.info("  3. python main.py       — start the server")
        logger.info("=" * 70)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        sys.exit(0)
    except Exception:
        logger.exception("Training failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
