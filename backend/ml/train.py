import torch
import os
import sys
import json
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import MODEL_DIR, DATASET_PATH, logger
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)

# ─────────────────────────────────────────────
# DATASET EXPANSION (SMART)
# ─────────────────────────────────────────────
def expand_dataset(data):
    templates = [
        "Tell me about {}",
        "Explain {}",
        "Give details about {}",
        "What is {}",
        "Can you explain {}?",
        "I want to know about {}"
    ]

    expanded = []

    for item in data:
        expanded.append(item)

        for t in templates:
            expanded.append({
                "instruction": t.format(item["instruction"].lower()),
                "output": item["output"]
            })

    # 🔥 OUT OF SCOPE TRAINING
    expanded += [
        {"instruction": "Who is Elon Musk?", "output": "I can help only with Genkit services."},
        {"instruction": "Tell me a joke", "output": "I can help only with Genkit services."},
        {"instruction": "Weather today?", "output": "I can help only with Genkit services."}
    ]

    random.shuffle(expanded)
    return expanded[:800]


# ─────────────────────────────────────────────
# TRAIN FUNCTION
# ─────────────────────────────────────────────
def run_training():

    BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    logger.info(f"🚀 Loading base model: {BASE_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.config.use_cache = False

    # ─────────────────────────────────────────
    # LOAD DATA
    # ─────────────────────────────────────────
    if not os.path.exists(DATASET_PATH):
        raise RuntimeError("Dataset not found")

    with open(DATASET_PATH, "r") as f:
        raw_data = json.load(f)

    expanded_data = expand_dataset(raw_data)

    dataset = Dataset.from_list(expanded_data)
    dataset = dataset.train_test_split(test_size=0.1)

    # ─────────────────────────────────────────
    # TOKENIZE
    # ─────────────────────────────────────────
    def tokenize(example):

        prompt = f"""### SYSTEM:
You are Genkit AI assistant.
Rules:
- Answer ONLY about Genkit
- Use simple words
- Max 3 bullet points

### USER:
{example['instruction']}

### ASSISTANT:
"""

        full_text = prompt + example["output"] + tokenizer.eos_token

        tokens = tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=256
        )

        labels = tokens["input_ids"].copy()

        prompt_len = len(tokenizer(prompt)["input_ids"])

        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        tokens["labels"] = labels
        return tokens

    tokenized = dataset.map(
        tokenize,
        remove_columns=dataset["train"].column_names
    )

    # ─────────────────────────────────────────
    # TRAIN CONFIG (FIXED)
    # ─────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"]
    )

    # ─────────────────────────────────────────
    # TRAIN
    # ─────────────────────────────────────────
    logger.info("🚀 Training started...")
    trainer.train()

    logger.info("💾 Saving model...")

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    logger.info("✅ Training completed")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_training()