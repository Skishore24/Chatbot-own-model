"""
backend/evaluate.py
----------------------------------------------------
Genkit AI V6 Master Benchmark Evaluation Launcher.
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from training.evaluate import run_benchmark

if __name__ == "__main__":
    run_benchmark()
