"""
backend/app/ai/llm/train.py
----------------------------------------------------
GENKIT AI v5.0 Enterprise Model Trainer
Pure PyTorch implementation with AMP (Automatic Mixed Precision), Gradient Accumulation,
AdamW, Cosine Annealing with Warmup, Loss Tracking & Checkpointing.
"""

import os
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from app.core.logger import logger
from app.core.config import settings
from app.ai.llm.ml_model import EnterpriseGPTModel, GPTConfig
from app.ai.tokenizer.tokenizer import ByteFallbackBPETokenizer


class TextDataset(Dataset):
    """PyTorch Dataset for Token Sequence Training."""

    def __init__(self, sequences: List[List[int]], block_size: int = 2048, pad_id: int = 0):
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


class CosineWarmupScheduler:
    """Cosine Annealing Learning Rate Scheduler with Warmup."""

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

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        elif self.current_step > self.max_steps:
            lr = self.min_lr
        else:
            decay_ratio = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.min_lr + coeff * (self.max_lr - self.min_lr)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr


class ModelTrainer:
    """Enterprise PyTorch Model Trainer."""

    def __init__(
        self,
        model: EnterpriseGPTModel,
        tokenizer: ByteFallbackBPETokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=settings.LEARNING_RATE,
            weight_decay=settings.WEIGHT_DECAY,
            betas=(0.9, 0.95),
        )

        self.is_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self.use_amp = settings.USE_AMP and self.is_cuda

        if self.is_cuda and self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.encoder.get("<pad>", 0))

    def train_epoch(self, dataloader: DataLoader, scheduler: CosineWarmupScheduler, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        self.optimizer.zero_grad()

        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)

            if self.use_amp and self.is_cuda:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                with torch.amp.autocast("cuda", dtype=dtype):
                    logits, _ = self.model(x)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / settings.GRADIENT_ACCUMULATION_STEPS
                self.scaler.scale(loss).backward()
            else:
                logits, _ = self.model(x)
                loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / settings.GRADIENT_ACCUMULATION_STEPS
                loss.backward()

            if (step + 1) % settings.GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(dataloader):
                if self.use_amp and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), settings.GRADIENT_CLIP)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), settings.GRADIENT_CLIP)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                curr_lr = scheduler.step()

            total_loss += loss.item() * settings.GRADIENT_ACCUMULATION_STEPS

            if (step + 1) % 50 == 0 or (step + 1) == len(dataloader):
                elapsed = time.time() - start_time
                logger.info(
                    f"Epoch [{epoch}] Step [{step+1}/{len(dataloader)}] "
                    f"Loss: {loss.item()*settings.GRADIENT_ACCUMULATION_STEPS:.4f} "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f} Time: {elapsed:.2f}s"
                )

        avg_loss = total_loss / max(len(dataloader), 1)
        perplexity = math.exp(min(avg_loss, 20.0))
        logger.info(f"Epoch [{epoch}] Finished. Avg Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
        return avg_loss

    def save_checkpoint(self, filepath: Optional[str] = None) -> None:
        save_path = filepath or str(settings.MODEL_DIR / "model_v5.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.model.config,
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Saved model checkpoint to {save_path}")
