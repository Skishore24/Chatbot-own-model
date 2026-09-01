"""
backend/training/train_model.py
----------------------------------------------------
Production PyTorch Model Trainer for Genkit AI V6.1.
- Hardware-accelerated training (NVIDIA CUDA / AMP / RTX 3050)
- Train / Validation Split (90/10) with validation loss & perplexity tracking
- Attention Mask support for padding tokens
- Real-data pre-training smoke test
- AdamW + Cosine Warmup Learning Rate Scheduler
- Gradient Clipping & Accumulation
- CheckpointManager atomic persistence and immediate torch.load verification
- Comprehensive training report generation (reports/training_report.json)
"""

import os
import sys
import json
import math
import time
import random
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from app.llm.checkpoint import CheckpointManager
from training.prepare import build_instruction_corpus
from training.train_tokenizer import train_tokenizer


# ==============================================================================
# 1. TEXT DATASET WITH ATTENTION MASK
# ==============================================================================
class TextDataset(Dataset):
    """PyTorch Dataset for Token Sequences with explicit Attention Masks."""

    def __init__(self, sequences: List[List[int]], block_size: int = 512, pad_id: int = 0):
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for seq in sequences:
            if len(seq) <= 1:
                continue
            if len(seq) > block_size + 1:
                seq = seq[: block_size + 1]
            else:
                seq = seq + [pad_id] * (block_size + 1 - len(seq))

            x_list = seq[:-1]
            y_list = seq[1:]

            x = torch.tensor(x_list, dtype=torch.long)
            y = torch.tensor(y_list, dtype=torch.long)
            mask = (x != pad_id).long()

            self.samples.append((x, y, mask))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
# 3. ENTERPRISE MODEL TRAINER
# ==============================================================================
class EnterpriseTrainer:
    """Hardware-accelerated PyTorch Trainer with atomic checkpoint verification."""

    def __init__(
        self,
        model: EnterpriseGPTModel,
        tokenizer: ByteFallbackBPETokenizer,
        config: GPTConfig,
        lr: float = 3e-4,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

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

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=settings.WEIGHT_DECAY,
            betas=(0.9, 0.95),
        )

        self.use_amp = settings.USE_AMP and self.is_cuda
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)

    def run_real_smoke_test(self, sample_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> bool:
        """Verifies full forward + backward training step with real tokenized dataset sample."""
        logger.info("Executing Pre-Training Real-Data Smoke Test...")
        try:
            self.model.train()
            self.optimizer.zero_grad()

            x, y, mask = sample_batch
            x = x.to(self.device)
            y = y.to(self.device)
            mask = mask.to(self.device)

            logits, _ = self.model(x, attention_mask=mask)
            loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError("Smoke test computed NaN/Inf loss!")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), settings.GRADIENT_CLIP)
            self.optimizer.step()
            self.optimizer.zero_grad()

            logger.info(f"[OK] Real-data smoke test passed! Initial loss: {loss.item():.4f}")
            return True
        except Exception as e:
            logger.error(f"[FAIL] Smoke test failed: {e}")
            raise e

    def train_epoch(
        self,
        dataloader: DataLoader,
        scheduler: CosineWarmupScheduler,
        epoch: int,
        grad_accum_steps: int = 8,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        start_time = time.time()
        self.optimizer.zero_grad()

        for step, (x, y, mask) in enumerate(dataloader):
            x = x.to(self.device, non_blocking=self.is_cuda)
            y = y.to(self.device, non_blocking=self.is_cuda)
            mask = mask.to(self.device, non_blocking=self.is_cuda)

            if self.use_amp:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                with torch.amp.autocast("cuda", dtype=dtype):
                    logits, _ = self.model(x, attention_mask=mask)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / grad_accum_steps
                self.scaler.scale(loss).backward()
            else:
                logits, _ = self.model(x, attention_mask=mask)
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

        avg_loss = total_loss / max(len(dataloader), 1)
        elapsed = time.time() - start_time
        logger.info(f"Epoch [{epoch}] Training Loss: {avg_loss:.4f} (Time: {elapsed:.2f}s)")
        return avg_loss

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Evaluates model loss and perplexity on held-out validation split."""
        self.model.eval()
        total_loss = 0.0

        for x, y, mask in val_loader:
            x = x.to(self.device, non_blocking=self.is_cuda)
            y = y.to(self.device, non_blocking=self.is_cuda)
            mask = mask.to(self.device, non_blocking=self.is_cuda)

            logits, _ = self.model(x, attention_mask=mask)
            loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()

        avg_val_loss = total_loss / max(len(val_loader), 1)
        val_perplexity = math.exp(min(avg_val_loss, 20.0))
        return avg_val_loss, val_perplexity

    def save_checkpoint(
        self,
        filepath: Optional[str] = None,
        epoch: Optional[int] = None,
        val_loss: Optional[float] = None,
    ) -> bool:
        """Saves model checkpoint safely via CheckpointManager."""
        save_path = filepath or str(settings.MODEL_CHECKPOINT_PATH)
        training_meta = {
            "epoch": epoch,
            "val_loss": val_loss,
            "device": str(self.device),
        }
        tokenizer_meta = {
            "vocab_size": self.tokenizer.vocab_size,
        }

        return CheckpointManager.save_atomic(
            filepath=save_path,
            model_state_dict=self.model.state_dict(),
            config=self.config,
            vocab_size=self.tokenizer.vocab_size,
            training_metadata=training_meta,
            tokenizer_metadata=tokenizer_meta,
        )


# ==============================================================================
# 4. MASTER TRAINING PIPELINE
# ==============================================================================
def train_pipeline(
    epochs: int = 60,
    batch_size: int = 4,
    accum_steps: int = 8,
    block_size: int = 512,
    vocab_size: int = 2084,
    lr: float = 3e-4,
    device: Optional[str] = None,
    retrain_tokenizer: bool = False,
) -> Dict[str, Any]:
    start_train_time = time.time()
    logger.info("=" * 70)
    logger.info("GENKIT AI v6.1 — ENTERPRISE MODEL TRAINING PIPELINE")
    logger.info("=" * 70)

    # 1. Corpus Preparation & Validation
    corpus = build_instruction_corpus()
    if not corpus:
        logger.error("No training data found! Check backend/datasets/")
        sys.exit(1)
    logger.info(f"Loaded training corpus with {len(corpus):,} instruction pairs")

    # 2. Tokenizer Loading or Training
    tokenizer = ByteFallbackBPETokenizer(vocab_size=vocab_size)
    if settings.tokenizer_checkpoint_exists() and not retrain_tokenizer:
        tokenizer.load(str(settings.TOKENIZER_CHECKPOINT_PATH))
        logger.info(f"Loaded existing Tokenizer: {settings.TOKENIZER_CHECKPOINT_PATH} (Vocab: {tokenizer.vocab_size:,})")
    else:
        logger.info("Training Byte-Fallback BPE Tokenizer...")
        tokenizer.train_on_corpus(corpus, target_vocab_size=vocab_size)
        tokenizer.save(str(settings.TOKENIZER_CHECKPOINT_PATH))
        logger.info(f"Tokenizer saved: {settings.TOKENIZER_CHECKPOINT_PATH} (Vocab: {tokenizer.vocab_size:,})")

    # 3. Token Encoding & Dataset Preparation
    logger.info("Encoding instruction sequences...")
    encoded_sequences = []
    for text in corpus:
        ids = tokenizer.encode(text, add_special_tokens=True)
        if len(ids) > 4:
            encoded_sequences.append(ids)

    logger.info(f"Encoded {len(encoded_sequences):,} total valid sequences")
    random.shuffle(encoded_sequences)

    # 4. 90/10 Train / Validation Split
    split_idx = int(len(encoded_sequences) * 0.90)
    train_seqs = encoded_sequences[:split_idx]
    val_seqs = encoded_sequences[split_idx:] if split_idx < len(encoded_sequences) else train_seqs[:50]

    train_dataset = TextDataset(train_seqs, block_size=block_size, pad_id=tokenizer.pad_id)
    val_dataset = TextDataset(val_seqs, block_size=block_size, pad_id=tokenizer.pad_id)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Dataset split: {len(train_dataset):,} train | {len(val_dataset):,} validation")

    # 5. Build Model Architecture
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_embd=settings.EMBED_DIM,
        n_layer=settings.NUM_LAYERS,
        n_head=settings.NUM_HEADS,
        n_kv_head=settings.NUM_KV_HEADS,
        dropout=settings.DROPOUT,
        bias=settings.BIAS,
        pad_token_id=tokenizer.pad_id,
        rope_freq_base=settings.ROPE_FREQ_BASE,
    )

    model = EnterpriseGPTModel(config)
    param_count = model.count_parameters()
    logger.info(f"Model architecture initialized: {param_count:,} parameters")

    trainer = EnterpriseTrainer(model, tokenizer, config, lr=lr, device=device)

    # 6. Real-Data Pre-Training Smoke Test
    sample_batch = next(iter(train_loader))
    trainer.run_real_smoke_test(sample_batch)

    # 7. Training Loop with Periodic Verification
    total_steps = len(train_loader) * epochs
    scheduler = CosineWarmupScheduler(
        trainer.optimizer,
        warmup_steps=min(settings.WARMUP_STEPS, max(1, total_steps // 5)),
        max_steps=max(total_steps, 10),
        max_lr=lr,
        min_lr=settings.MIN_LEARNING_RATE,
    )

    best_val_loss = float("inf")
    epoch_logs = []

    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(train_loader, scheduler, epoch, grad_accum_steps=accum_steps)
        val_loss, val_ppl = trainer.evaluate(val_loader)
        logger.info(f"Epoch [{epoch}/{epochs}] Val Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_perplexity": round(val_ppl, 2),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(epoch=epoch, val_loss=val_loss)
            logger.info(f"[BEST] Best model checkpoint verified & saved at epoch {epoch} (Val Loss: {val_loss:.4f})")

    # 8. Save final config & verify final production checkpoint
    config.save_to_file(str(settings.CONFIG_CHECKPOINT_PATH))
    is_valid, verify_status = CheckpointManager.verify_checkpoint(str(settings.MODEL_CHECKPOINT_PATH))

    if not is_valid:
        raise RuntimeError(f"Final checkpoint verification failed: {verify_status}")

    total_training_time = time.time() - start_train_time
    logger.info("=" * 70)
    logger.info(f"Training successfully completed in {total_training_time:.2f}s! Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"Model Checkpoint: {settings.MODEL_CHECKPOINT_PATH} (Verified: PASS)")
    logger.info("=" * 70)

    # 9. Generate Training Report
    report_data = {
        "model_name": "Genkit Enterprise GPT v6.1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "parameters": param_count,
        "vocab_size": tokenizer.vocab_size,
        "block_size": block_size,
        "dataset_samples": len(encoded_sequences),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accum_steps,
        "learning_rate": lr,
        "best_val_loss": round(best_val_loss, 4),
        "training_duration_seconds": round(total_training_time, 2),
        "checkpoint_path": str(settings.MODEL_CHECKPOINT_PATH),
        "checkpoint_verified": is_valid,
        "checkpoint_status": verify_status,
        "epoch_history": epoch_logs,
    }

    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = settings.REPORTS_DIR / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"Saved training report: {report_path}")

    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genkit AI V6 Master Model Training Pipeline")
    parser.add_argument("--epochs", type=int, default=settings.EPOCHS, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=settings.BATCH_SIZE, help="Micro-batch size")
    parser.add_argument("--accum-steps", type=int, default=settings.GRADIENT_ACCUMULATION_STEPS, help="Gradient accumulation steps")
    parser.add_argument("--block-size", type=int, default=settings.BLOCK_SIZE, help="Sequence block size")
    parser.add_argument("--vocab-size", type=int, default=settings.VOCAB_SIZE, help="Vocabulary size")
    parser.add_argument("--lr", type=float, default=settings.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--retrain-tokenizer", action="store_true", help="Force retraining BPE tokenizer")
    args = parser.parse_args()

    train_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        lr=args.lr,
        device=args.device,
        retrain_tokenizer=args.retrain_tokenizer,
    )
