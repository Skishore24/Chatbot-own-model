"""
backend/train.py
----------------------------------------------------
GENKIT AI v5.0 Enterprise Training Launcher
Full training pipeline:
  1. Load domain corpus from backend/datasets/
  2. Train BPE tokenizer on corpus
  3. Encode all training pairs
  4. Train EnterpriseGPTModel with AMP + cosine LR + gradient accumulation
  5. Save model checkpoint + tokenizer to backend/genkit-model/
"""

import sys
import json
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.core.logger import logger
from app.ai.llm.ml_model import EnterpriseGPTModel, GPTConfig
from app.ai.llm.train import ModelTrainer, TextDataset, CosineWarmupScheduler
from app.ai.tokenizer.tokenizer import ByteFallbackBPETokenizer

import torch
from torch.utils.data import DataLoader


def load_corpus() -> list:
    """Loads all text from backend/datasets/ for training."""
    dataset_dir = settings.DATASET_DIR
    sentences = []

    if not dataset_dir.exists():
        logger.warning(f"Dataset dir not found: {dataset_dir}")
        return sentences

    for json_file in dataset_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        for field in ["instruction", "output", "content", "description", "answer", "text", "question"]:
                            val = item.get(field)
                            if val and isinstance(val, str) and len(val.strip()) > 5:
                                sentences.append(val.strip())
                    elif isinstance(item, str) and len(item.strip()) > 5:
                        sentences.append(item.strip())
            elif isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, str) and len(v.strip()) > 5:
                        sentences.append(f"{k}: {v.strip()}")

            logger.info(f"Loaded {json_file.name} — corpus size so far: {len(sentences)}")
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")

    return sentences


import argparse


def main():
    parser = argparse.ArgumentParser(description="Genkit AI Custom GPT Model Training")
    parser.add_argument("--epochs", type=int, default=settings.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=settings.BATCH_SIZE, help="Training batch size")
    parser.add_argument("--vocab-size", type=int, default=settings.VOCAB_SIZE, help="Target vocabulary size")
    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    vocab_size = args.vocab_size

    logger.info("=" * 70)
    logger.info("GENKIT AI v5.0 — ENTERPRISE MODEL TRAINING PIPELINE")
    logger.info(f"Target Epochs: {epochs} | Batch Size: {batch_size} | Vocab Size: {vocab_size}")
    logger.info("=" * 70)

    # ── Step 1: Load Corpus ──────────────────────────────────────────────
    logger.info("Step 1: Loading domain corpus...")
    corpus = load_corpus()
    if not corpus:
        logger.error("No training data found! Add JSON files to backend/datasets/")
        sys.exit(1)
    logger.info(f"Corpus loaded: {len(corpus):,} sentences")

    # ── Step 2: Train BPE Tokenizer ──────────────────────────────────────
    logger.info(f"Step 2: Training Byte-Fallback BPE Tokenizer (target vocab: {vocab_size})...")
    tokenizer = ByteFallbackBPETokenizer(vocab_size=vocab_size)
    tokenizer.train_on_corpus(corpus, target_vocab_size=vocab_size)
    tokenizer_path = str(settings.TOKENIZER_CHECKPOINT_PATH)
    tokenizer.save(tokenizer_path)
    logger.info(f"Tokenizer saved: {tokenizer_path}")

    # ── Step 3: Encode Training Data ─────────────────────────────────────
    logger.info("Step 3: Encoding corpus into token sequences...")
    encoded_sequences = []
    for text in corpus:
        try:
            ids = tokenizer.encode(text, add_special_tokens=True)
            if len(ids) > 2:
                encoded_sequences.append(ids)
        except Exception:
            pass
    logger.info(f"Encoded {len(encoded_sequences):,} sequences")

    if not encoded_sequences:
        logger.error("No sequences encoded! Check tokenizer.")
        sys.exit(1)

    # ── Step 4: Build Model ───────────────────────────────────────────────
    logger.info("Step 4: Building model architecture...")
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=settings.BLOCK_SIZE,
        n_embd=settings.EMBED_DIM,
        n_head=settings.NUM_HEADS,
        n_kv_head=settings.NUM_KV_HEADS,
        n_layer=settings.NUM_LAYERS,
        dropout=settings.DROPOUT,
        bias=settings.BIAS,
        page_size=settings.KV_CACHE_PAGE_SIZE,
        rope_freq_base=settings.ROPE_FREQ_BASE,
        rope_scale_factor=settings.ROPE_SCALE_FACTOR,
    )
    model = EnterpriseGPTModel(config)
    logger.info(f"Model parameters: {model.count_parameters():,}")

    # ── Step 5: Train ─────────────────────────────────────────────────────
    logger.info("Step 5: Starting training loop...")
    trainer = ModelTrainer(model, tokenizer)

    dataset = TextDataset(encoded_sequences, block_size=settings.BLOCK_SIZE)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    total_steps = len(dataloader) * epochs
    scheduler = CosineWarmupScheduler(
        trainer.optimizer,
        warmup_steps=min(settings.WARMUP_STEPS, max(1, total_steps // 5)),
        max_steps=max(total_steps, 10),
        max_lr=settings.LEARNING_RATE,
        min_lr=settings.MIN_LEARNING_RATE,
    )

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        avg_loss = trainer.train_epoch(dataloader, scheduler, epoch)
        if avg_loss < best_loss:
            best_loss = avg_loss
            trainer.save_checkpoint()
            logger.info(f"Best checkpoint saved at epoch {epoch} (loss={avg_loss:.4f})")

    # ── Step 6: Final Save ────────────────────────────────────────────────
    trainer.save_checkpoint()
    logger.info("=" * 70)
    logger.info(f"Training completed! Best loss: {best_loss:.4f}")
    logger.info(f"Model: {settings.MODEL_CHECKPOINT_PATH}")
    logger.info(f"Tokenizer: {settings.TOKENIZER_CHECKPOINT_PATH}")
    logger.info("Run: python main.py to start the inference server.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
