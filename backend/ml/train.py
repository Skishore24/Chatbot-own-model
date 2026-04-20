import torch, os, sys, json, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import MODEL_DIR, DATASET_PATH, logger
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 🔥 REDUCE MEMORY PRESSURE
MAX_LENGTH = 256

def expand_dataset(data):
    templates = [
        "Tell me about {}",
        "Explain {}",
        "What is {}",
        "Give details about {}",
    ]

    expanded = list(data)

    for item in data:
        base = item["instruction"].lower()
        for t in templates:
            expanded.append({
                "instruction": t.format(base),
                "output": item["output"]
            })

    # rejection training
    expanded.extend([
        {"instruction": "Who is Elon Musk?", "output": "I can help only with Genkit services."},
        {"instruction": "Tell me a joke", "output": "I can help only with Genkit services."}
    ])

    random.shuffle(expanded)
    return expanded[:800]  # 🔥 REDUCED

def run_training():

    logger.info(f"Loading model: {BASE_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 🔥 CPU SAFE LOAD
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    model.config.use_cache = False

    # ─────────────────────────
    # LOAD DATA
    # ─────────────────────────
    if not os.path.exists(DATASET_PATH):
        raise RuntimeError("Dataset not found")

    with open(DATASET_PATH, "r") as f:
        raw = json.load(f)

    data = expand_dataset(raw)

    logger.info(f"Training on {len(data)} samples")

    dataset = Dataset.from_list(data).train_test_split(test_size=0.1)

    # ─────────────────────────
    # TOKENIZE
    # ─────────────────────────
    def tokenize(example):

        prompt = f"""### SYSTEM:
You are Genkit AI.
Answer ONLY about Genkit.

### USER:
{example['instruction']}

### ASSISTANT:
"""

        full = prompt + example["output"] + tokenizer.eos_token

        tokens = tokenizer(
            full,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

        labels = tokens["input_ids"].copy()

        prompt_len = len(tokenizer(prompt)["input_ids"])

        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        tokens["labels"] = labels
        return tokens

    tokenized = dataset.map(tokenize)

    # ─────────────────────────
    # TRAIN CONFIG (CPU SAFE)
    # ─────────────────────────
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,

        per_device_train_batch_size=1,   # 🔥 IMPORTANT
        gradient_accumulation_steps=16,  # simulate batch

        num_train_epochs=3,
        learning_rate=2e-5,

        logging_steps=20,
        save_strategy="epoch",

        # ❌ REMOVE eval (saves memory)
        do_eval=False,

        fp16=False,  # ❌ CPU → no fp16

        dataloader_num_workers=0,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    logger.info("Training started...")
    trainer.train()

    logger.info("Saving model...")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    logger.info("✅ Training completed")

if __name__ == "__main__":
    run_training()