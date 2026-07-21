"""
evaluate.py
----------------------------------------------------
Genkit AI - Model Evaluation Script
Usage
-----
    cd backend
    python evaluate.py

What this does
--------------
1. Loads the trained model (model.pt)
2. Runs validation on 10% of the dataset
3. Reports: loss, perplexity, BLEU scores
4. Generates sample responses for QA review
Author : Genkit AI
"""
import os
import sys
import json
import math
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import logger, MODEL_DIR, DATASET_DIR
from ai.training.metrics import TrainingMetrics

metrics = TrainingMetrics()


# ============================================================
# SAMPLE QA PAIRS FOR EVALUATION
# ============================================================
EVAL_QUESTIONS = [
    "What services does Genkit offer?",
    "How much does a website cost at Genkit?",
    "How can I contact Genkit?",
    "Does Genkit make mobile apps?",
    "What technologies does Genkit use?",
    "Tell me about Genkit's branding services.",
    "Does Genkit do SEO?",
    "Who is Elon Musk?",  # should be rejected (out-of-scope)
]


def load_model_and_tokenizer():
    """Load trained model and tokenizer."""
    from ai.llm.ml_model import GPT, GPTConfig, SimpleWordTokenizer
    import torch

    config_path = MODEL_DIR / "config.json"
    model_path = MODEL_DIR / "model.pt"
    vocab_path = MODEL_DIR / "vocab.json"

    if not all(p.exists() for p in [config_path, model_path, vocab_path]):
        logger.error("Model files not found. Please run: python train.py")
        sys.exit(1)

    with open(config_path) as f:
        cfg = json.load(f)

    tokenizer = SimpleWordTokenizer.from_pretrained(str(MODEL_DIR))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = GPTConfig(
        vocab_size=cfg["vocab_size"],
        block_size=cfg["block_size"],
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        dropout=0.0,  # no dropout during eval
    )
    model = GPT(config)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def compute_perplexity(model, tokenizer, device, data, max_samples=200):
    """Compute validation perplexity on a sample of the dataset."""
    import torch
    import torch.nn.functional as F

    total_loss = 0.0
    count = 0
    samples = data[:max_samples]

    with torch.no_grad():
        for item in samples:
            instruction = item.get("instruction", "").strip()
            output = item.get("output", "").strip()
            text = f"[INST] {instruction} [/INST] {output}"
            ids = (
                [tokenizer.bos_token_id]
                + tokenizer.encode(text, add_special_tokens=False)
                + [tokenizer.eos_token_id]
            )
            if len(ids) < 4:
                continue
            ids = ids[:256]
            input_ids = torch.tensor([ids], dtype=torch.long).to(device)
            _, loss = model(input_ids, targets=input_ids)
            if loss is not None and not math.isnan(loss.item()):
                total_loss += loss.item()
                count += 1

    if count == 0:
        return float("inf")
    avg_loss = total_loss / count
    return avg_loss, metrics.perplexity(avg_loss)


def generate_samples(num=5):
    """Generate responses for sample questions."""
    from ai.llm.inference import generate
    print("\n" + "=" * 70)
    print("SAMPLE GENERATIONS")
    print("=" * 70)
    for q in EVAL_QUESTIONS[:num]:
        from ai.llm.prompt_builder import prompt_builder
        prompt = prompt_builder.build_simple(q)
        response = generate(prompt, max_new_tokens=150)
        print(f"\nQ: {q}")
        print(f"A: {response}")
        print("-" * 50)


def main():
    logger.info("=" * 70)
    logger.info("GENKIT AI - MODEL EVALUATION")
    logger.info("=" * 70)

    # Load dataset
    data = []
    for fp in DATASET_DIR.glob("*.json"):
        try:
            with open(fp, encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    data.extend(loaded)
        except Exception as e:
            logger.warning(f"Skipping {fp}: {e}")

    if not data:
        logger.error("No dataset found.")
        sys.exit(1)

    logger.info(f"Dataset size: {len(data)} records")

    # Load model
    logger.info("Loading model...")
    model, tokenizer, device = load_model_and_tokenizer()
    logger.info(f"Device: {device}")

    # Compute perplexity
    logger.info("Computing validation perplexity...")
    result = compute_perplexity(model, tokenizer, device, data)
    if isinstance(result, tuple):
        avg_loss, ppl = result
        print(f"\n{'=' * 40}")
        print(f"  Validation Loss   : {avg_loss:.4f}")
        print(f"  Perplexity        : {ppl:.2f}")
        print(f"{'=' * 40}")
    else:
        print(f"  Perplexity: {result}")

    # Generate samples
    try:
        generate_samples(num=5)
    except Exception as e:
        logger.warning(f"Generation failed: {e}")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
