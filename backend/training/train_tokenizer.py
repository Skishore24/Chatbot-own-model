"""
backend/training/train_tokenizer.py
----------------------------------------------------
Standalone script to train Byte-Fallback BPE Tokenizer.
"""

import sys
import argparse
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.core.logger import logger
from app.llm.tokenizer import ByteFallbackBPETokenizer
from training.prepare import build_instruction_corpus


def train_tokenizer(vocab_size: int = settings.VOCAB_SIZE, output_path: str = None) -> ByteFallbackBPETokenizer:
    """Trains BPE Tokenizer on the domain instruction corpus and saves checkpoint."""
    save_path = output_path or str(settings.TOKENIZER_CHECKPOINT_PATH)
    logger.info("=" * 60)
    logger.info(f"Training Byte-Fallback BPE Tokenizer (Target Vocab: {vocab_size:,})")
    logger.info("=" * 60)

    corpus = build_instruction_corpus()
    if not corpus:
        raise ValueError("Training corpus is empty! Check backend/datasets/")

    tokenizer = ByteFallbackBPETokenizer(vocab_size=vocab_size)
    tokenizer.train_on_corpus(corpus, target_vocab_size=vocab_size)
    tokenizer.save(save_path)
    logger.info(f"Tokenizer successfully trained and saved to: {save_path}")
    logger.info(f"Final Vocabulary Size: {tokenizer.vocab_size:,}")
    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Byte-Fallback BPE Tokenizer")
    parser.add_argument("--vocab-size", type=int, default=settings.VOCAB_SIZE, help="Target vocabulary size")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    train_tokenizer(vocab_size=args.vocab_size, output_path=args.output)
