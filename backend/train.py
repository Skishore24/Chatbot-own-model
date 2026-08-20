"""
backend/train.py
----------------------------------------------------
Top-level entry point pointing to training/train.py
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.train import (
    TextDataset,
    CosineWarmupScheduler,
    ModelTrainer,
    load_corpus,
    main,
)

if __name__ == "__main__":
    main()
