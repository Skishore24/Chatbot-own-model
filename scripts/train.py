"""
scripts/train.py
----------------------------------------------------
GENKIT AI v5.0 Master Training CLI Script
Usage:
    python scripts/train.py
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.core.logger import logger
from app.ai.llm.ml_model import EnterpriseGPTModel, GPTConfig
from app.ai.llm.train import ModelTrainer, TextDataset, CosineWarmupScheduler
from app.ai.tokenizer.tokenizer import default_tokenizer
from torch.utils.data import DataLoader


def main():
    logger.info("=" * 70)
    logger.info("GENKIT AI v5.0 — MASTER MODEL TRAINER CLI")
    logger.info("=" * 70)

    config = GPTConfig(
        vocab_size=settings.VOCAB_SIZE,
        block_size=settings.BLOCK_SIZE,
        n_embd=settings.EMBED_DIM,
        n_head=settings.NUM_HEADS,
        n_kv_head=settings.NUM_KV_HEADS,
        n_layer=settings.NUM_LAYERS,
    )
    model = EnterpriseGPTModel(config)
    trainer = ModelTrainer(model, default_tokenizer)

    sample_text = "Genkit AI provides enterprise AI and web development services."
    encoded = default_tokenizer.encode(sample_text)
    dataset = TextDataset([encoded], block_size=settings.BLOCK_SIZE)
    dataloader = DataLoader(dataset, batch_size=2)
    scheduler = CosineWarmupScheduler(trainer.optimizer, warmup_steps=10, max_steps=100)

    for epoch in range(1, 3):
        trainer.train_epoch(dataloader, scheduler, epoch)

    trainer.save_checkpoint()
    logger.info("=" * 70)
    logger.info("Training completed successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
