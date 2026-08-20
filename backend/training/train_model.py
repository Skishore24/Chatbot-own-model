"""
backend/training/train_model.py
----------------------------------------------------
Production PyTorch Model Trainer for Genkit AI V6.
- Hardware-accelerated training (NVIDIA RTX 3050 / CUDA / AMP)
- Dynamic token sequence datasets with padding mask handling
- AdamW + Cosine Warmup Learning Rate Scheduler
- Gradient Clipping & Accumulation
- Pre-training smoke test
- Checkpoint persistence (model.pt, tokenizer.json, config.json)
"""

import os
import sys
import json
import math
import time
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from app.core.config import settings
from app.core.logger import logger
from app.llm.config import GPTConfig
from app.llm.model import EnterpriseGPTModel
from app.llm.tokenizer import ByteFallbackBPETokenizer
from training.prepare import build_instruction_corpus
from training.train_tokenizer import train_tokenizer


# ==============================================================================
# 1. TEXT DATASET
# ==============================================================================
class TextDataset(Dataset):
    """PyTorch Dataset for Token Sequences."""

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
# 2. COSINE WARMUP SCHEDULER
# ==============================================================================
class CosineWarmupScheduler:
    """Cosine Annealing Learning Rate Scheduler with Linear Warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 200,
        max_steps: int = 5000,
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
# 3. MODEL TRAINER
# ==============================================================================
class ModelTrainer:
    """Production PyTorch Model Trainer."""

    def __init__(
        self,
        model: EnterpriseGPTModel,
        tokenizer: ByteFallbackBPETokenizer,
        device: Optional[str] = None,
        lr: float = settings.LEARNING_RATE,
        weight_decay: float = settings.WEIGHT_DECAY,
    ):
        self.model = model
        self.tokenizer = tokenizer
        dev_str = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev_str)
        self.is_cuda = self.device.type == "cuda" and torch.cuda.is_available()

        if self.is_cuda:
            torch.cuda.empty_cache()
            gpu_name = torch.cuda.get_device_name(self.device)
            gpu_vram = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
            logger.info(f"Training on GPU: {gpu_name} ({gpu_vram:.2f} GB VRAM)")
        else:
            logger.warning("Training on CPU (CUDA not detected)")

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

        self.use_amp = settings.USE_AMP and self.is_cuda
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)

    def smoke_test(self, batch_size: int = 2, block_size: int = 64) -> None:
        """Performs 1-batch smoke test to verify forward and backward pass stability."""
        logger.info("Running pre-flight model smoke test...")
        self.model.train()
        dummy_x = torch.randint(0, self.tokenizer.vocab_size, (batch_size, block_size), device=self.device)
        dummy_y = torch.randint(0, self.tokenizer.vocab_size, (batch_size, block_size), device=self.device)

        self.optimizer.zero_grad()
        if self.use_amp:
            with torch.amp.autocast("cuda"):
                logits, _ = self.model(dummy_x)
                loss = self.criterion(logits.view(-1, logits.size(-1)), dummy_y.view(-1))
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits, _ = self.model(dummy_x)
            loss = self.criterion(logits.view(-1, logits.size(-1)), dummy_y.view(-1))
            loss.backward()
            self.optimizer.step()

        self.optimizer.zero_grad()
        logger.info("Pre-flight smoke test passed successfully.")

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
        logger.info(f"Epoch [{epoch}] Finished — Avg Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
        return avg_loss

    def save_checkpoint(self, model_path: Optional[str] = None) -> None:
        """Saves model weights, tokenizer, and configuration to disk."""
        save_path = model_path or str(settings.MODEL_CHECKPOINT_PATH)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": {
                "vocab_size": self.model.config.vocab_size,
                "block_size": self.model.config.block_size,
                "n_embd": self.model.config.n_embd,
                "n_layer": self.model.config.n_layer,
                "n_head": self.model.config.n_head,
                "n_kv_head": self.model.config.n_kv_head,
            },
        }
        torch.save(checkpoint, save_path)

        # Save config.json
        config_path = str(settings.CONFIG_CHECKPOINT_PATH)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint["config"], f, indent=2)

        logger.info(f"Saved model checkpoint: {save_path}")


# ==============================================================================
# 4. TRAINING PIPELINE ENTRYPOINT
# ==============================================================================
def train_pipeline(
    epochs: int = settings.EPOCHS,
    batch_size: int = settings.BATCH_SIZE,
    accum_steps: int = settings.GRADIENT_ACCUMULATION_STEPS,
    block_size: int = settings.BLOCK_SIZE,
    vocab_size: int = settings.VOCAB_SIZE,
    lr: float = settings.LEARNING_RATE,
    device: Optional[str] = None,
) -> float:
    """Executes the complete Genkit AI V6 training workflow."""
    effective_batch = batch_size * accum_steps
    logger.info("=" * 70)
    logger.info("GENKIT AI v6.0 — NEURAL MODEL TRAINING PIPELINE")
    logger.info(f"Target Epochs: {epochs} | Micro-Batch: {batch_size} | Accum Steps: {accum_steps} (Effective Batch: {effective_batch})")
    logger.info(f"Block Size: {block_size} | Vocab Size: {vocab_size} | LR: {lr}")
    logger.info("=" * 70)

    # 1. Prepare Corpus
    corpus = build_instruction_corpus()
    if not corpus:
        raise ValueError("No training data found in backend/datasets/")

    # 2. Train / Load Tokenizer
    tokenizer = train_tokenizer(vocab_size=vocab_size)

    # 3. Encode Sequences
    logger.info("Encoding instruction corpus into token sequences...")
    encoded_sequences = []
    for text in corpus:
        ids = tokenizer.encode(text, add_special_tokens=True)
        if len(ids) > 2:
            encoded_sequences.append(ids)

    logger.info(f"Encoded {len(encoded_sequences):,} sequences.")

    # 4. Initialize Model Architecture
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_embd=settings.EMBED_DIM,
        n_head=settings.NUM_HEADS,
        n_kv_head=settings.NUM_KV_HEADS,
        n_layer=settings.NUM_LAYERS,
        dropout=settings.DROPOUT,
        bias=settings.BIAS,
        rope_freq_base=settings.ROPE_FREQ_BASE,
    )
    model = EnterpriseGPTModel(config)
    logger.info(f"Initialized EnterpriseGPTModel: {model.count_parameters():,} parameters")

    # 5. Trainer Setup
    trainer = ModelTrainer(model, tokenizer, device=device, lr=lr)
    trainer.smoke_test(batch_size=min(batch_size, 2), block_size=min(block_size, 64))

    dataset = TextDataset(encoded_sequences, block_size=block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=trainer.is_cuda,
    )

    total_steps = len(dataloader) * epochs
    scheduler = CosineWarmupScheduler(
        trainer.optimizer,
        warmup_steps=min(settings.WARMUP_STEPS, max(1, total_steps // 5)),
        max_steps=max(total_steps, 10),
        max_lr=lr,
        min_lr=settings.MIN_LEARNING_RATE,
    )

    # 6. Training Loop
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        avg_loss = trainer.train_epoch(dataloader, scheduler, epoch, grad_accum_steps=accum_steps)
        if avg_loss < best_loss:
            best_loss = avg_loss
            trainer.save_checkpoint()
            logger.info(f"★ Checkpoint saved at epoch {epoch} (loss={avg_loss:.4f})")

    trainer.save_checkpoint()
    logger.info("=" * 70)
    logger.info(f"Training completed successfully! Best loss: {best_loss:.4f}")
    logger.info(f"Model Checkpoint : {settings.MODEL_CHECKPOINT_PATH}")
    logger.info(f"Tokenizer Checkpoint: {settings.TOKENIZER_CHECKPOINT_PATH}")
    logger.info("=" * 70)
    return best_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Genkit AI V6 Model")
    parser.add_argument("--epochs", type=int, default=settings.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=settings.BATCH_SIZE)
    parser.add_argument("--accum-steps", type=int, default=settings.GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--block-size", type=int, default=settings.BLOCK_SIZE)
    parser.add_argument("--vocab-size", type=int, default=settings.VOCAB_SIZE)
    parser.add_argument("--lr", type=float, default=settings.LEARNING_RATE)
    parser.add_argument("--device", type=str, default=None)
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
