import torch
import os
import shutil
import sys

# ─────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import MODEL_DIR, DATASET_PATH, logger
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)


def run_training():
    """
    Trains ONLY your Genkit model.
    ❌ No GPT fallback
    """

    # ─────────────────────────────────────────
    # CHECK BASE MODEL
    # ─────────────────────────────────────────
    if not os.path.exists(MODEL_DIR):
        raise RuntimeError(
            "❌ No base model found.\n"
            "👉 You must place your pretrained model inside ml/genkit-model/"
        )

    logger.info(f"Loading YOUR model from: {MODEL_DIR}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

        model.config.use_cache = False

    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return

    # ─────────────────────────────────────────
    # DATASET CHECK
    # ─────────────────────────────────────────
    if not os.path.exists(DATASET_PATH):
        raise RuntimeError(f"❌ Dataset not found at {DATASET_PATH}")

    logger.info("Loading dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH)

    # ─────────────────────────────────────────
    # TOKENIZATION
    # ─────────────────────────────────────────
    def tokenize(example):

        prompt = f"### Instruction: {example['instruction']}\n### Response: "
        full_text = prompt + example["output"] + tokenizer.eos_token

        tokens = tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=256
        )

        labels = tokens["input_ids"].copy()

        # mask instruction
        prompt_len = len(tokenizer(prompt)["input_ids"])

        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        # mask padding
        for i in range(len(labels)):
            if tokens["input_ids"][i] == tokenizer.pad_token_id:
                labels[i] = -100

        tokens["labels"] = labels
        return tokens

    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset["train"].map(
        tokenize,
        remove_columns=dataset["train"].column_names
    )

    # ─────────────────────────────────────────
    # TRAINING CONFIG
    # ─────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=20,
        learning_rate=3e-5,
        logging_steps=10,
        save_strategy="no",
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    # ─────────────────────────────────────────
    # TRAIN
    # ─────────────────────────────────────────
    logger.info("🚀 Training started...")

    try:
        trainer.train()

        logger.info("💾 Saving model...")
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)

        logger.info("✅ Training completed successfully")

    except Exception as e:
        logger.error(f"Training failed: {e}")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_training()