"""
ai/llm/train.py
----------------------------------------------------
Genkit AI - Production GPT Trainer
Features
--------
✔ Pure PyTorch
✔ Own GPT Model
✔ CUDA Training
✔ AMP Mixed Precision
✔ Gradient Accumulation
✔ Validation
✔ Early Stopping
✔ Resume Training
✔ Best Model Saving
✔ Checkpoint Saving
✔ Learning Rate Scheduler
✔ Automatic GPU Detection
Author : Genkit AI
"""
import os
import sys
import json
import math
import time
import random
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
)
from torch.cuda.amp import (
    autocast,
    GradScaler,
)
# ---------------------------------------------------------
# Backend Path
# ---------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
from config import (
    MODEL_DIR,
    DATASET_DIR,
    logger,
    BLOCK_SIZE,
    EMBED_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    LEARNING_RATE,
    BATCH_SIZE,
    EPOCHS,
)
from ai.llm.ml_model import (
    GPT,
    GPTConfig,
    SimpleWordTokenizer,
)
# ==========================================================
# TRAINING CONFIG
# ==========================================================
SEED = 42
MAX_LENGTH = BLOCK_SIZE  # Expanded context length from config
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION = 2
VALIDATION_SPLIT = 0.10
EARLY_STOPPING = 8
SAVE_EVERY = 5
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()
USE_AMP = torch.cuda.is_available()
# ==========================================================
# RANDOM SEED
# ==========================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ==========================================================
# DEVICE
# ==========================================================
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
logger.info("=" * 60)
logger.info("Genkit AI Production Trainer")
logger.info("=" * 60)
logger.info(f"Device : {DEVICE}")
if DEVICE.type == "cuda":
    logger.info(
        f"GPU : {torch.cuda.get_device_name(0)}"
    )
    logger.info(
        f"CUDA : {torch.version.cuda}"
    )
    logger.info(
        f"VRAM : {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB"
    )
# ==========================================================
# DATA AUGMENTATION
# ==========================================================
def expand_dataset(data):
    expanded = list(data)
    templates = [
        "Explain {}",
        "Tell me about {}",
        "What is {}",
        "Can you explain {}",
        "Give information about {}",
        "Describe {}",
    ]
    for item in data:
        question = item["instruction"].strip()
        answer = item["output"]
        for template in templates:
            expanded.append({
                "instruction": template.format(question),
                "output": answer,
            })
    reject_answer = (
        "I can answer only questions related to Genkit. "
        "Please ask about our services, products, support or company."
    )
    rejection_questions = [
        "Who is Elon Musk?",
        "Tell me a joke",
        "What is IPL?",
        "What is Bitcoin?",
        "Who is Prime Minister?",
        "What is the weather?",
        "Write a poem",
        "Solve my homework",
        "Who won FIFA?",
        "What is ChatGPT?",
        "What is Python?",
        "Sing a song",
    ]
    for question in rejection_questions:
        expanded.append({
            "instruction": question,
            "output": reject_answer,
        })
    random.shuffle(expanded)
    logger.info(
        f"Expanded Dataset : {len(expanded)} samples"
    )
    return expanded
# ==========================================================
# CHAT DATASET
# ==========================================================
class ChatDataset(Dataset):
    """
    Dataset for causal language modeling.
    Format:
    <s>
    user: ...
    assistant: ...
    </s>
    Loss is calculated only on assistant tokens.
    """
    def __init__(
        self,
        data,
        tokenizer,
        max_length=MAX_LENGTH,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        logger.info("Building training dataset...")
        for item in data:
            instruction = str(
                item.get("instruction", "")
            ).strip()
            output = str(
                item.get("output", "")
            ).strip()
            # Use [INST]...[/INST] structured prompt format.
            # The model is trained on this format and will learn
            # to generate text between [/INST] and </s>.
            prompt = (
                f"[INST] {instruction} [/INST]"
            )
            prompt_ids = (
                [tokenizer.bos_token_id]
                + tokenizer.encode(
                    prompt,
                    add_special_tokens=False,
                )
            )
            answer_ids = (
                tokenizer.encode(
                    output,
                    add_special_tokens=False,
                )
                + [tokenizer.eos_token_id]
            )
            input_ids = prompt_ids + answer_ids
            labels = (
                [-100] * len(prompt_ids)
                + answer_ids
            )
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            self.samples.append({
                "input_ids": input_ids,
                "labels": labels,
            })
        logger.info(
            f"Dataset ready ({len(self.samples)} samples)"
        )
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]
# ==========================================================
# COLLATE FUNCTION
# ==========================================================
def collate_fn(batch):
    max_len = max(
        len(x["input_ids"])
        for x in batch
    )
    input_ids = []
    labels = []
    attention_mask = []
    for sample in batch:
        ids = sample["input_ids"]
        lbl = sample["labels"]
        pad = max_len - len(ids)
        ids = ids + (
            [0] * pad
        )
        lbl = lbl + (
            [-100] * pad
        )
        mask = (
            [1] * len(sample["input_ids"])
            + [0] * pad
        )
        input_ids.append(ids)
        labels.append(lbl)
        attention_mask.append(mask)
    return {
        "input_ids": torch.tensor(
            input_ids,
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            labels,
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            attention_mask,
            dtype=torch.long,
        ),
    }
# ==========================================================
# LOAD DATASET
# ==========================================================
def load_dataset():
    file_path = DATASET_DIR / "dataset.json"
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found:\n{file_path}"
        )
    raw_data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            file_data = json.load(f)
            if isinstance(file_data, list):
                for item in file_data:
                    inst = item.get("instruction", "").strip()
                    out = item.get("output", "").strip()
                    intent = item.get("intent", "general")
                    if inst and out:
                        raw_data.append({
                            "instruction": inst,
                            "output": out,
                            "intent": intent
                        })
    except Exception as e:
        logger.error(f"Error loading dataset.json: {e}")
    logger.info(
        f"Loaded {len(raw_data)} prompt records from dataset.json"
    )
    # Shuffling directly to preserve variety in train/validation split
    random.shuffle(raw_data)
    return raw_data

# ==========================================================
# ML INTENT CLASSIFIER TRAINER
# ==========================================================
def train_intent_classifier(data):
    """
    Fits and serializes a scikit-learn intent classifier on the queries.
    """
    logger.info("=" * 60)
    logger.info("Training ML Intent Classifier...")
    logger.info("=" * 60)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import joblib
    
    texts = []
    labels = []
    
    for item in data:
        inst = item.get("instruction", "")
        # Parse out user query text
        query = inst.split("Question:\n")[-1].strip()
        texts.append(query)
        labels.append(item.get("intent", "out_of_domain"))
        
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = vectorizer.fit_transform(texts)
        
        clf = LogisticRegression(C=1.0, max_iter=300)
        clf.fit(X, labels)
        
        clf_path = os.path.join(MODEL_DIR, "intent_classifier.joblib")
        vec_path = os.path.join(MODEL_DIR, "intent_vectorizer.joblib")
        
        joblib.dump(clf, clf_path)
        joblib.dump(vectorizer, vec_path)
        
        logger.info(f"Intent Classifier trained successfully on {len(texts)} samples.")
        logger.info(f"Classes: {clf.classes_}")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Failed to train intent classifier: {e}")

# ==========================================================
# TOKENIZER
# ==========================================================
def build_tokenizer(data):
    texts = []
    for item in data:
        texts.append(item["instruction"])
        texts.append(item["output"])
    tokenizer = SimpleWordTokenizer.train_on_texts(
        texts
    )
    tokenizer.save_pretrained(
        MODEL_DIR
    )
    logger.info(
        f"Vocabulary Size : {len(tokenizer)}"
    )
    return tokenizer
# ==========================================================
# DATALOADERS
# ==========================================================
def create_dataloaders():
    data = load_dataset()
    tokenizer = build_tokenizer(data)
    dataset = ChatDataset(
        data,
        tokenizer,
        MAX_LENGTH,
    )
    train_size = int(
        len(dataset)
        * (1 - VALIDATION_SPLIT)
    )
    valid_size = (
        len(dataset)
        - train_size
    )
    train_dataset, valid_dataset = random_split(
        dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(
            SEED
        ),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
    )
    logger.info(
        f"Train Samples : {len(train_dataset)}"
    )
    logger.info(
        f"Validation Samples : {len(valid_dataset)}"
    )
    return (
        tokenizer,
        train_loader,
        valid_loader,
    )
# ==========================================================
# MODEL
# ==========================================================
def build_model(tokenizer):
    config = GPTConfig(
        vocab_size=len(tokenizer),
        block_size=BLOCK_SIZE,
        n_embd=EMBED_DIM,
        n_head=NUM_HEADS,
        n_layer=NUM_LAYERS,
        dropout=0.1,
        gradient_checkpointing=True
    )
    model = GPT(config)
    model = model.to(DEVICE)
    logger.info("=" * 60)
    logger.info("GPT MODEL")
    logger.info("=" * 60)
    logger.info(
        f"Parameters : {model.get_num_params():,}"
    )
    if torch.cuda.device_count() > 1:
        logger.info(
            f"Using {torch.cuda.device_count()} GPUs"
        )
        model = nn.DataParallel(model)
    return model, config
# ==========================================================
# OPTIMIZER
# ==========================================================
def build_optimizer(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
            },
        ],
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    return optimizer
# ==========================================================
# LR SCHEDULER
# ==========================================================
class CosineWarmupScheduler:
    def __init__(
        self,
        optimizer,
        warmup_steps,
        total_steps,
        min_lr=1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = LEARNING_RATE
        self.step_num = 0
    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = (
                self.base_lr
                * self.step_num
                / self.warmup_steps
            )
        else:
            progress = (
                self.step_num - self.warmup_steps
            ) / (
                self.total_steps - self.warmup_steps
            )
            lr = self.min_lr + (
                self.base_lr - self.min_lr
            ) * (
                0.5
                * (
                    1
                    + math.cos(
                        math.pi * progress
                    )
                )
            )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]
# ==========================================================
# AMP
# ==========================================================
scaler = GradScaler(
    enabled=USE_AMP
)
# ==========================================================
# CHECKPOINT
# ==========================================================
CHECKPOINT_FILE = os.path.join(
    MODEL_DIR,
    "checkpoint.pt",
)
BEST_MODEL = os.path.join(
    MODEL_DIR,
    "best_model.pt",
)

def resume_training(
    model,
    optimizer,
):
    start_epoch = 0
    best_loss = 1e9
    if not os.path.exists(CHECKPOINT_FILE):
        return (
            start_epoch,
            best_loss,
        )
    logger.info(
        "Loading previous checkpoint..."
    )
    try:
        checkpoint = torch.load(
            CHECKPOINT_FILE,
            map_location=DEVICE,
        )
        model_state = checkpoint["model"]
        new_model_state = {}
        for k, v in model_state.items():
            if k.endswith(".attn.bias"):
                continue
            new_key = k
            new_key = new_key.replace(".ln_1.", ".ln1.")
            new_key = new_key.replace(".ln_2.", ".ln2.")
            new_key = new_key.replace(".mlp.c_fc.", ".mlp.fc.")
            new_key = new_key.replace(".mlp.c_proj.", ".mlp.proj.")
            new_model_state[new_key] = v
        
        # Check for shape mismatch (e.g., if vocabulary size changed)
        wte_key = "transformer.wte.weight"
        if wte_key in new_model_state:
            checkpoint_shape = new_model_state[wte_key].shape
            inner_model = model.module if hasattr(model, 'module') else model
            model_shape = inner_model.transformer.wte.weight.shape
            if checkpoint_shape != model_shape:
                logger.warning(
                    f"Checkpoint vocabulary size {checkpoint_shape[0]} does not match "
                    f"current model vocabulary size {model_shape[0]} (dataset expanded). "
                    "Starting training from scratch."
                )
                return start_epoch, best_loss

        model.load_state_dict(new_model_state)
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        logger.info(
            f"Resuming from epoch {start_epoch}"
        )
    except Exception as e:
        logger.warning(
            f"Failed to load checkpoint: {e}. Starting training from scratch."
        )
    return (
        start_epoch,
        best_loss,
    )
    # ==========================================================
# SAVE
# ==========================================================
def save_checkpoint(
    epoch,
    model,
    optimizer,
    best_loss,
):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
        },
        CHECKPOINT_FILE,
    )
# ==========================================================
# TRAIN ONE EPOCH
# ==========================================================
def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    epoch,
):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(
            DEVICE,
            non_blocking=True,
        )
        labels = batch["labels"].to(
            DEVICE,
            non_blocking=True,
        )
        with autocast(enabled=USE_AMP):
            _, loss = model(
                input_ids,
                targets=labels,
            )
            loss = loss / GRADIENT_ACCUMULATION
        scaler.scale(loss).backward()
        if (
            (step + 1) % GRADIENT_ACCUMULATION == 0
            or
            (step + 1) == len(train_loader)
        ):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        total_loss += loss.item() * GRADIENT_ACCUMULATION
        if step % 20 == 0:
            logger.info(
                f"Epoch {epoch:02d} "
                f"Step {step:04d}/{len(train_loader)} "
                f"Loss {loss.item()*GRADIENT_ACCUMULATION:.4f} "
                f"LR {scheduler.get_lr():.6f}"
            )
    return total_loss / len(train_loader)
# ==========================================================
# VALIDATION
# ==========================================================
def compute_bleu(ref: str, cand: str) -> float:
    r_words = ref.lower().split()
    c_words = cand.lower().split()
    if not r_words or not c_words:
        return 0.0
    overlap = set(r_words) & set(c_words)
    return len(overlap) / len(set(c_words))

def compute_rouge(ref: str, cand: str) -> float:
    r_words = ref.lower().split()
    c_words = cand.lower().split()
    if not r_words or not c_words:
        return 0.0
    overlap = set(r_words) & set(c_words)
    return len(overlap) / len(r_words)

def simple_decode(ids, tokenizer) -> str:
    words = []
    for tid in ids:
        val = tid.item() if hasattr(tid, "item") else tid
        if val in {tokenizer.bos_token_id, tokenizer.eos_token_id}:
            continue
        word = tokenizer.inverse_vocab.get(val, "")
        if word:
            if word in {".", ",", "!", "?", ":", ";", ")", "]", "}"}:
                words.append(word)
            else:
                words.append(" " + word)
    return "".join(words).strip()

@torch.no_grad()
def validate(
    model,
    valid_loader,
    tokenizer=None,
):
    model.eval()
    total_loss = 0.0
    for batch in valid_loader:
        input_ids = batch["input_ids"].to(
            DEVICE,
            non_blocking=True,
        )
        labels = batch["labels"].to(
            DEVICE,
            non_blocking=True,
        )
        with autocast(enabled=USE_AMP):
            _, loss = model(
                input_ids,
                targets=labels,
            )
        total_loss += loss.item()
    
    # Compute BLEU & ROUGE on 10 random validation samples for logging
    bleu_scores = []
    rouge_scores = []
    
    if tokenizer is not None:
        samples_evaluated = 0
        for batch in valid_loader:
            if samples_evaluated >= 10:
                break
            input_ids_batch = batch["input_ids"]
            labels_batch = batch["labels"]
            for i in range(len(input_ids_batch)):
                if samples_evaluated >= 10:
                    break
                input_ids = input_ids_batch[i]
                labels = labels_batch[i]
                
                # Extract prompt and target
                prompt_ids = []
                target_ids = []
                for tid, lid in zip(input_ids, labels):
                    if lid.item() == -100:
                        prompt_ids.append(tid.item())
                    else:
                        target_ids.append(tid.item())
                
                target_str = simple_decode(target_ids, tokenizer)
                if not target_str:
                    continue
                
                # Greedy text generation
                gen_ids = []
                x = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
                for _ in range(80):
                    if x.size(1) > model.config.block_size:
                        x_cond = x[:, -model.config.block_size:]
                    else:
                        x_cond = x
                    logits, _ = model(x_cond)
                    logits = logits[:, -1, :]
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    token_id = next_token.item()
                    if token_id == tokenizer.eos_token_id:
                        break
                    gen_ids.append(token_id)
                    x = torch.cat((x, next_token), dim=1)
                
                gen_str = simple_decode(gen_ids, tokenizer)
                
                bleu_scores.append(compute_bleu(target_str, gen_str))
                rouge_scores.append(compute_rouge(target_str, gen_str))
                samples_evaluated += 1
                
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
        logger.info(
            f"Validation Metrics - BLEU-1: {avg_bleu:.4f} | ROUGE-1: {avg_rouge:.4f}"
        )
        
    return total_loss / len(valid_loader)
# ==========================================================
# TRAIN MODEL
# ==========================================================
def train_model():
    set_seed()
    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )
    tokenizer, train_loader, valid_loader = (
        create_dataloaders()
    )
    
    # Train Intent Classifier
    raw_data = load_dataset()
    train_intent_classifier(raw_data)

    model, config = build_model(
        tokenizer
    )
    optimizer = build_optimizer(
        model
    )
    total_steps = (
        len(train_loader)
        * EPOCHS
    ) // GRADIENT_ACCUMULATION
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=max(
            50,
            total_steps // 20,
        ),
        total_steps=total_steps,
    )
    start_epoch, best_loss = (
        resume_training(
            model,
            optimizer,
        )
    )
    patience = 0
    logger.info("=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)
    for epoch in range(
        start_epoch,
        EPOCHS,
    ):
        start = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            epoch + 1,
        )
        valid_loss = validate(
            model,
            valid_loader,
            tokenizer=tokenizer,
        )
        elapsed = (
            time.time() - start
        )
        logger.info(
            f"Epoch {epoch+1}/{EPOCHS}"
        )
        logger.info(
            f"Train Loss : {train_loss:.4f}"
        )
        logger.info(
            f"Valid Loss : {valid_loss:.4f}"
        )
        logger.info(
            f"Time : {elapsed:.1f} sec"
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            patience = 0
            torch.save(
                model.state_dict(),
                BEST_MODEL,
            )
            logger.info(
                "New best model saved."
            )
        else:
            patience += 1
            logger.info(
                f"No improvement ({patience}/{EARLY_STOPPING})"
            )
        save_checkpoint(
            epoch,
            model,
            optimizer,
            best_loss,
        )
        if (
            (epoch + 1)
            % SAVE_EVERY
            == 0
        ):
            torch.save(
                model.state_dict(),
                os.path.join(
                    MODEL_DIR,
                    f"epoch_{epoch+1}.pt",
                ),
            )
        if patience >= EARLY_STOPPING:
            logger.info(
                "Early stopping activated."
            )
            break
    logger.info("=" * 60)
    logger.info("Training Finished")
    logger.info("=" * 60)
# ==========================================================
# EXPORT FINAL MODEL
# ==========================================================
def export_model(model, tokenizer, config):
    """
    Save the final model, tokenizer and configuration
    required for inference.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    final_model = (
        model.module
        if isinstance(model, nn.DataParallel)
        else model
    )
    # Save weights
    torch.save(
        final_model.state_dict(),
        os.path.join(MODEL_DIR, "model.pt"),
    )
    # Save tokenizer
    tokenizer.save_pretrained(MODEL_DIR)
    # Save config
    config_data = {
        "vocab_size": config.vocab_size,
        "block_size": config.block_size,
        "n_embd": config.n_embd,
        "n_head": config.n_head,
        "n_layer": config.n_layer,
        "dropout": config.dropout,
    }
    with open(
        os.path.join(MODEL_DIR, "config.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config_data,
            f,
            indent=4,
            ensure_ascii=False,
        )
    logger.info("=" * 60)
    logger.info("Model Export Completed")
    logger.info("=" * 60)
    logger.info(f"Model     : {os.path.join(MODEL_DIR, 'model.pt')}")
    logger.info(f"Tokenizer : {os.path.join(MODEL_DIR, 'vocab.json')}")
    logger.info(f"Config    : {os.path.join(MODEL_DIR, 'config.json')}")
# ==========================================================
# RUN TRAINING
# ==========================================================
def run_training():
    logger.info("=" * 70)
    logger.info("GENKIT AI CUSTOM GPT TRAINER")
    logger.info("=" * 70)
    tokenizer, train_loader, valid_loader = create_dataloaders()
    
    # Train Intent Classifier
    raw_data = load_dataset()
    train_intent_classifier(raw_data)

    model, config = build_model(tokenizer)
    optimizer = build_optimizer(model)
    total_steps = (
        len(train_loader)
        * EPOCHS
    ) // GRADIENT_ACCUMULATION
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=max(50, total_steps // 20),
        total_steps=total_steps,
    )
    start_epoch, best_loss = resume_training(
        model,
        optimizer,
    )
    patience = 0
    for epoch in range(start_epoch, EPOCHS):
        logger.info("-" * 70)
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}")
        logger.info("-" * 70)
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            epoch + 1,
        )
        valid_loss = validate(
            model,
            valid_loader,
            tokenizer=tokenizer,
        )
        logger.info(
            f"Train Loss : {train_loss:.6f}"
        )
        logger.info(
            f"Valid Loss : {valid_loss:.6f}"
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            patience = 0
            logger.info(
                "Best model improved."
            )
            export_model(
                model,
                tokenizer,
                config,
            )
        else:
            patience += 1
            logger.info(
                f"No Improvement ({patience}/{EARLY_STOPPING})"
            )
        save_checkpoint(
            epoch,
            model,
            optimizer,
            best_loss,
        )
        if patience >= EARLY_STOPPING:
            logger.info(
                "Early stopping activated."
            )
            break
    logger.info("=" * 70)
    logger.info("Training Finished Successfully")
    logger.info("=" * 70)
# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    try:
        set_seed()
        run_training()
    except KeyboardInterrupt:
        logger.warning(
            "Training interrupted by user."
        )
    except Exception:
        logger.exception(
            "Training failed."
        )
        raise
