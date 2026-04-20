import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import os
from threading import Thread
from app.config import MODEL_DIR, logger

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Loading Genkit model on {device}...")

# HARD FAIL
if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    raise RuntimeError("❌ Train your model first: python ml/train.py")

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

model.eval()

logger.info("✅ Model loaded successfully")


# ─────────────────────────────────────────────
# CLEAN OUTPUT (VERY IMPORTANT)
# ─────────────────────────────────────────────
def clean_output(text: str, prompt: str) -> str:
    if "ANSWER:" in text:
        text = text.split("ANSWER:")[-1]

    text = text.replace(prompt, "").strip()

    # remove repeated lines
    lines = []
    for l in text.split("\n"):
        l = l.strip()
        if l and l not in lines:
            lines.append(l)

    text = "\n".join(lines)

    # limit length
    return text[:200]


# ─────────────────────────────────────────────
# REAL STREAMING (TOKEN BY TOKEN)
# ─────────────────────────────────────────────
def generate_stream(prompt):

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.6,          # 🔥 improved
        top_p=0.9,
        repetition_penalty=1.3,   # 🔥 reduce repetition
        no_repeat_ngram_size=3,   # 🔥 critical
        eos_token_id=tokenizer.eos_token_id,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_text = ""

    for token in streamer:
        partial_text += token

        # 🔥 STOP early if structure detected
        if partial_text.count("•") >= 3:
            break

        # 🔥 STOP if too long
        if len(partial_text) > 200:
            break

        yield token


# ─────────────────────────────────────────────
# FULL RESPONSE (NON-STREAM)
# ─────────────────────────────────────────────
def generate_response(prompt: str) -> str:

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return clean_output(text, prompt)