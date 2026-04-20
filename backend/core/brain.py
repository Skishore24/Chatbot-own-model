import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread, Lock
import os
from app.config import MODEL_DIR, logger

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {device}")

_tokenizer = None
_model = None
_model_lock = Lock()


def _load_model():
    global _tokenizer, _model

    if _model is not None:
        return True

    with _model_lock:
        if _model is not None:
            return True

        config_path = os.path.join(MODEL_DIR, "config.json")

        if not os.path.exists(config_path):
            raise RuntimeError(f"Model not found at {MODEL_DIR}. Run training first.")

        try:
            logger.info("Loading Genkit model...")

            _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token

            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            ).to(device)

            _model.eval()

            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.exception("Model load failed")
            raise e


# lazy load
_load_model()


def generate_stream(prompt: str):
    """Production-safe streaming generator"""

    if _model is None or _tokenizer is None:
        yield "Model not available."
        return

    try:
        torch.set_grad_enabled(False)

        inputs = _tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        streamer = TextIteratorStreamer(
            _tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=20.0,
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=_tokenizer.pad_token_id,
        )

        thread = Thread(target=_model.generate, kwargs=generation_kwargs, daemon=True)
        thread.start()

        collected = ""

        for token in streamer:
            collected += token
            yield token

            # safe stop
            if len(collected) > 500:
                break

        thread.join(timeout=3)

    except Exception as e:
        logger.exception("Generation error")
        yield "⚠️ Model error. Please try again."