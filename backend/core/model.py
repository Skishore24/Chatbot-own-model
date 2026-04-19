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

# ❌ HARD FAIL (as you want)
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
# CLEAN RESPONSE
# ─────────────────────────────────────────────
def clean_output(text: str, prompt: str) -> str:
    """
    Extract clean answer from model output
    """

    if "ANSWER:" in text:
        text = text.split("ANSWER:")[-1]

    # remove prompt echo
    text = text.replace(prompt, "")

    # remove extra lines
    text = text.strip().split("\n")[0]

    # remove weird repetition
    if len(text) > 300:
        text = text[:300]

    return text.strip()


# ─────────────────────────────────────────────
# STREAM GENERATION (REAL STREAM)
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
        temperature=0.6,
        top_p=0.85,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for token in streamer:
        if "\n" in token:
            break
        yield token

# ─────────────────────────────────────────────
# SYNC GENERATION (OPTIONAL)
# ─────────────────────────────────────────────
def generate_response(prompt: str) -> str:
    """
    Full response (non-stream)
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.6,
        top_p=0.85,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return clean_output(text, prompt)