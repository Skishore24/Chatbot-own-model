"""
================================================================================
GENKIT AI v5.0 — UNIFIED ENTERPRISE MODEL TRAINING PIPELINE
================================================================================
Single definitive training pipeline located in backend/training/ containing:
  1. Corpus Loading & Extraction (backend/datasets/*.json)
  2. Byte-Fallback BPE Tokenizer Training & Encoding
  3. PyTorch Dataset with Dynamic Sequence Lengths
  4. Cosine Annealing Learning Rate Scheduler with Warmup
  5. GPU-Accelerated Trainer with Automatic Mixed Precision (AMP) & Gradient Accumulation
  6. Model & Tokenizer Checkpoint Exporter

Usage:
  python train.py
  python training/train.py
  python train.py --epochs 60 --batch-size 8 --accum-steps 4 --block-size 256
================================================================================
"""

import os
import sys
import json
import math
import time
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure backend root directory is in sys.path
BACKEND_DIR = (
    Path(__file__).resolve().parent.parent
    if Path(__file__).resolve().parent.name == "training"
    else Path(__file__).resolve().parent
)
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from app.core.config import settings
from app.core.logger import logger
from app.ai.llm.ml_model import EnterpriseGPTModel, GPTConfig
from app.ai.tokenizer.tokenizer import ByteFallbackBPETokenizer


# ==============================================================================
# 1. CORPUS LOADER
# ==============================================================================
def load_corpus() -> List[str]:
    """Loads all text from backend/datasets/ (*.json) for model training."""
    dataset_dir = settings.DATASET_DIR
    sentences = []

    if not dataset_dir.exists():
        logger.warning(f"Dataset directory not found: {dataset_dir}")
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

            logger.info(f"Loaded {json_file.name} — corpus size so far: {len(sentences):,}")
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")

    return sentences


# ==============================================================================
# 2. PYTORCH DATASET
# ==============================================================================
class TextDataset(Dataset):
    """PyTorch Dataset for Token Sequence Training."""

    def __init__(self, sequences: List[List[int]], block_size: int = 256, pad_id: int = 0):
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for seq in sequences:
            if len(seq) <= 1:
                continue
            if len(seq) > block_size + 1:
                seq = seq[: block_size + 1]
            else:
                seq = seq + [pad_id] * (block_size + 1 - len(seq))

            x = torch.tensor(seq[:-1], dtype=torch.long)
            y = torch.tensor(seq[1:], dtype=torch.long)
            self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


# ==============================================================================
# 3. COSINE WARMUP SCHEDULER
# ==============================================================================
class CosineWarmupScheduler:
    """Cosine Annealing Learning Rate Scheduler with Linear Warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 500,
        max_steps: int = 10000,
        max_lr: float = 3e-4,
        min_lr: float = 1e-5,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self) -> float:
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = self.max_lr * (self.current_step / max(1, self.warmup_steps))
        elif self.current_step > self.max_steps:
            lr = self.min_lr
        else:
            decay_ratio = (self.current_step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.min_lr + coeff * (self.max_lr - self.min_lr)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr


# ==============================================================================
# 4. GPU-ACCELERATED MODEL TRAINER
# ==============================================================================
class ModelTrainer:
    """Enterprise PyTorch Model Trainer with GPU Acceleration & AMP."""

    def __init__(
        self,
        model: EnterpriseGPTModel,
        tokenizer: ByteFallbackBPETokenizer,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer

        # Device selection: prefer CUDA if available
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda" and torch.cuda.is_available()

        if self.is_cuda:
            torch.cuda.empty_cache()
            gpu_name = torch.cuda.get_device_name(self.device)
            gpu_vram = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
            logger.info(f"Using GPU Acceleration: {gpu_name} ({gpu_vram:.2f} GB VRAM)")
        else:
            logger.warning("Using CPU for training (CUDA not available)")

        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=settings.LEARNING_RATE,
            weight_decay=settings.WEIGHT_DECAY,
            betas=(0.9, 0.95),
        )

        # Mixed Precision (AMP)
        self.use_amp = settings.USE_AMP and self.is_cuda
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # CrossEntropyLoss ignoring pad tokens
        pad_id = self.tokenizer.encoder.get("<pad>", 0)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    def train_epoch(
        self,
        dataloader: DataLoader,
        scheduler: CosineWarmupScheduler,
        epoch: int,
        grad_accum_steps: int = 4,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        self.optimizer.zero_grad()

        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device, non_blocking=self.is_cuda), y.to(self.device, non_blocking=self.is_cuda)

            # Forward pass with AMP
            if self.use_amp:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                with torch.amp.autocast("cuda", dtype=dtype):
                    logits, _ = self.model(x)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / grad_accum_steps
                self.scaler.scale(loss).backward()
            else:
                logits, _ = self.model(x)
                loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / grad_accum_steps
                loss.backward()

            # Gradient accumulation & optimizer step
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                if self.use_amp and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), settings.GRADIENT_CLIP)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), settings.GRADIENT_CLIP)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * grad_accum_steps

            # Periodic logging
            if (step + 1) % 50 == 0 or (step + 1) == len(dataloader):
                elapsed = time.time() - start_time
                current_lr = self.optimizer.param_groups[0]["lr"]
                step_loss = loss.item() * grad_accum_steps
                logger.info(
                    f"Epoch [{epoch}] Step [{step+1}/{len(dataloader)}] "
                    f"Loss: {step_loss:.4f} | LR: {current_lr:.6f} | Time: {elapsed:.2f}s"
                )

        avg_loss = total_loss / max(len(dataloader), 1)
        perplexity = math.exp(min(avg_loss, 20.0))
        logger.info(f"Epoch [{epoch}] Complete — Avg Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
        return avg_loss

    def save_checkpoint(self, filepath: Optional[str] = None) -> None:
        """Saves model weights and configuration to disk."""
        save_path = filepath or str(settings.MODEL_CHECKPOINT_PATH)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.model.config,
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Saved model checkpoint to: {save_path}")


# ==============================================================================
# 5. MAIN TRAINING PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Genkit AI Enterprise Model Training Pipeline")
    parser.add_argument("--epochs", type=int, default=settings.EPOCHS, help="Number of training epochs (default: 60)")
    parser.add_argument("--batch-size", type=int, default=8, help="Micro-batch size per step (default: 8, optimized for 6GB GPUs)")
    parser.add_argument("--block-size", type=int, default=256, help="Max sequence length (default: 256)")
    parser.add_argument("--accum-steps", type=int, default=4, help="Gradient accumulation steps (default: 4, effective batch = 32)")
    parser.add_argument("--vocab-size", type=int, default=settings.VOCAB_SIZE, help="Target vocabulary size (default: 16000)")
    parser.add_argument("--lr", type=float, default=settings.LEARNING_RATE, help="Peak learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    args = parser.parse_args()

    effective_batch = args.batch_size * args.accum_steps

    logger.info("=" * 70)
    logger.info("GENKIT AI v5.0 — ENTERPRISE MODEL TRAINING PIPELINE")
    logger.info(f"Epochs: {args.epochs} | Micro-Batch: {args.batch_size} | Accum Steps: {args.accum_steps} (Effective Batch: {effective_batch})")
    logger.info(f"Block Size: {args.block_size} | Vocab Size: {args.vocab_size} | Device: {args.device}")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU Hardware: {gpu_name} ({gpu_vram:.2f} GB VRAM)")
    logger.info("=" * 70)

    # ── Step 1: Load Domain Corpus ─────────────────────────────────────────
    logger.info("Step 1: Loading domain corpus...")
    corpus = load_corpus()
    if not corpus:
        logger.error("No training data found! Add JSON files to backend/datasets/")
        sys.exit(1)
    logger.info(f"Corpus loaded: {len(corpus):,} total sentences")

    # ── Step 2: Train Byte-Fallback BPE Tokenizer ──────────────────────────
    logger.info(f"Step 2: Training Byte-Fallback BPE Tokenizer (target vocab: {args.vocab_size})...")
    tokenizer = ByteFallbackBPETokenizer(vocab_size=args.vocab_size)
    tokenizer.train_on_corpus(corpus, target_vocab_size=args.vocab_size)
    tokenizer_path = str(settings.TOKENIZER_CHECKPOINT_PATH)
    tokenizer.save(tokenizer_path)
    logger.info(f"Tokenizer saved: {tokenizer_path}")

    # ── Step 3: Encode Training Sequences ──────────────────────────────────
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
        logger.error("No sequences encoded! Please check tokenizer.")
        sys.exit(1)

    # ── Step 4: Build Enterprise GPT Model Architecture ────────────────────
    logger.info("Step 4: Building model architecture...")
    config = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = EnterpriseGPTModel(config)
    logger.info(f"Model parameters: {model.count_parameters():,}")

    # ── Step 5: Start Training Loop ────────────────────────────────────────
    logger.info("Step 5: Starting model training loop...")
    trainer = ModelTrainer(model, tokenizer, device=args.device)

    dataset = TextDataset(encoded_sequences, block_size=args.block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=trainer.is_cuda,
    )

    total_steps = len(dataloader) * args.epochs
    scheduler = CosineWarmupScheduler(
        trainer.optimizer,
        warmup_steps=min(settings.WARMUP_STEPS, max(1, total_steps // 5)),
        max_steps=max(total_steps, 10),
        max_lr=args.lr,
        min_lr=settings.MIN_LEARNING_RATE,
    )

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        avg_loss = trainer.train_epoch(
            dataloader=dataloader,
            scheduler=scheduler,
            epoch=epoch,
            grad_accum_steps=args.accum_steps,
        )
        if avg_loss < best_loss:
            best_loss = avg_loss
            trainer.save_checkpoint()
            logger.info(f"★ Best checkpoint updated at epoch {epoch} (loss={avg_loss:.4f})")

    # ── Step 6: Final Save ─────────────────────────────────────────────────
    trainer.save_checkpoint()
    logger.info("=" * 70)
    logger.info(f"Training completed successfully! Best loss: {best_loss:.4f}")
    logger.info(f"Model Checkpoint : {settings.MODEL_CHECKPOINT_PATH}")
    logger.info(f"Tokenizer Config : {settings.TOKENIZER_CHECKPOINT_PATH}")
    logger.info("Run 'python main.py' to start the Genkit AI server.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
