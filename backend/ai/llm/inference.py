"""
ai/llm/inference.py
----------------------------------------------------
Genkit AI - Inference Engine

Loads the trained custom GPT model.

Provides
--------
✓ load_model()
✓ warmup()
✓ is_model_loaded()

Author : Genkit AI
"""

import os
import sys
import json
from threading import Lock
from typing import Optional

import torch

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)

from config import (
    MODEL_DIR,
    DEVICE,
    TEMPERATURE,
    TOP_K,
    logger,
)

from ai.llm.ml_model import (
    GPT,
    GPTConfig,
    SimpleWordTokenizer,
)

# ============================================================
# Singleton Objects
# ============================================================

_tokenizer: Optional[SimpleWordTokenizer] = None
_model: Optional[GPT] = None

_model_lock = Lock()

MODEL_READY = False

# ============================================================
# Utilities
# ============================================================

def is_model_loaded() -> bool:
    """
    Returns True if model is already loaded.
    """
    return _model is not None


def get_model():
    return _model


def get_tokenizer():
    return _tokenizer


# ============================================================
# Load Model
# ============================================================

def load_model(force_reload: bool = False) -> bool:
    """
    Load tokenizer + GPT model.

    Returns
    -------
    bool
        True if successful.
    """

    global _model
    global _tokenizer
    global MODEL_READY

    if MODEL_READY and not force_reload:
        return True

    with _model_lock:

        if MODEL_READY and not force_reload:
            return True

        config_file = os.path.join(MODEL_DIR, "config.json")
        model_file = os.path.join(MODEL_DIR, "model.pt")
        vocab_file = os.path.join(MODEL_DIR, "vocab.json")

        required = [
            config_file,
            model_file,
            vocab_file,
        ]

        missing = [
            f for f in required
            if not os.path.exists(f)
        ]

        if missing:

            logger.warning(
                "Model files missing:"
            )

            for f in missing:
                logger.warning(" - %s", f)

            MODEL_READY = False
            return False

        try:

            logger.info(
                "Loading tokenizer..."
            )

            _tokenizer = SimpleWordTokenizer.from_pretrained(
                MODEL_DIR
            )

            logger.info(
                "Loading config..."
            )

            with open(
                config_file,
                "r",
                encoding="utf-8"
            ) as f:

                cfg = json.load(f)

            config = GPTConfig(

                vocab_size=cfg["vocab_size"],

                block_size=cfg["block_size"],

                n_embd=cfg["n_embd"],

                n_head=cfg["n_head"],

                n_layer=cfg["n_layer"],

                dropout=cfg.get(
                    "dropout",
                    0.1,
                ),
            )

            logger.info(
                "Creating GPT model..."
            )

            _model = GPT(config)

            logger.info(
                "Loading weights..."
            )

            state = torch.load(
                model_file,
                map_location=DEVICE,
            )

            # Adapt keys from trained model to match current class architecture
            new_state = {}
            for k, v in state.items():
                if k.endswith(".attn.bias"):
                    continue  # ignore mask buffer
                new_key = k
                new_key = new_key.replace(".ln_1.", ".ln1.")
                new_key = new_key.replace(".ln_2.", ".ln2.")
                new_key = new_key.replace(".mlp.c_fc.", ".mlp.fc.")
                new_key = new_key.replace(".mlp.c_proj.", ".mlp.proj.")
                new_state[new_key] = v
            state = new_state

            _model.load_state_dict(state)

            _model.to(DEVICE)

            _model.eval()

            MODEL_READY = True

            logger.info(
                "Model loaded successfully."
            )

            logger.info(
                "Device : %s",
                DEVICE,
            )

            logger.info(
                "Parameters : %d",
                _model.get_num_params(),
            )

            return True

        except Exception:

            logger.exception(
                "Failed loading model."
            )

            MODEL_READY = False

            _model = None
            _tokenizer = None

            return False


# ============================================================
# Warmup
# ============================================================

def warmup() -> bool:
    """
    Runs one dummy forward pass so the
    first user response is faster.
    """

    if not load_model():
        return False

    try:

        dummy = torch.tensor(
            [[_tokenizer.bos_token_id]],
            dtype=torch.long,
            device=DEVICE,
        )

        with torch.no_grad():
            _model(dummy)

        logger.info(
            "Inference warmup completed."
        )

        return True

    except Exception:

        logger.exception(
            "Warmup failed."
        )

        return False
    
    # ============================================================
# Generate IDs
# ============================================================

@torch.no_grad()
def generate_ids(
    input_ids,
    max_new_tokens: int = 150,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    repetition_penalty: float = 1.15,
):
    ids, _ = generate_ids_with_prob(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    return ids

@torch.no_grad()
def generate_ids_with_prob(
    input_ids,
    max_new_tokens: int = 150,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    repetition_penalty: float = 1.15,
):
    """
    Generate token ids and track probabilities.
    """
    if not load_model():
        return input_ids, 0.0

    generated = list(input_ids)
    probs_list = []

    x = torch.tensor(
        [generated],
        dtype=torch.long,
        device=DEVICE,
    )

    for _ in range(max_new_tokens):
        if x.size(1) > _model.config.block_size:
            x_cond = x[:, -_model.config.block_size:]
        else:
            x_cond = x

        logits, _ = _model(x_cond)
        logits = logits[:, -1, :]

        if repetition_penalty > 1:
            unique_tokens = set(generated)
            for token in unique_tokens:
                logits[0, token] /= repetition_penalty

        logits = logits / max(temperature, 1e-6)

        if top_k is not None and top_k > 0:
            values, _ = torch.topk(
                logits,
                min(top_k, logits.size(-1)),
            )
            logits[logits < values[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_id = next_token.item()
        
        prob = probs[0, token_id].item()
        probs_list.append(prob)

        generated.append(token_id)
        x = torch.cat((x, next_token), dim=1)

        if token_id == _tokenizer.eos_token_id:
            break

    avg_prob = sum(probs_list) / len(probs_list) if probs_list else 1.0
    return generated, avg_prob


# ============================================================
# Streaming Generation
# ============================================================

@torch.no_grad()
def generate_stream(
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    repetition_penalty: float = 1.15,
):
    """
    Stream generated text token-by-token.
    """
    if not load_model():
        yield "⚠️ Model not loaded."
        return

    try:
        input_ids = [_tokenizer.bos_token_id]
        input_ids.extend(
            _tokenizer.encode(
                prompt,
                add_special_tokens=False,
            )
        )
        generated, _ = generate_ids_with_prob(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        new_tokens = generated[len(input_ids):]

        for token in new_tokens:
            if token == _tokenizer.eos_token_id:
                break

            word = _tokenizer.inverse_vocab.get(token, "<unk>")
            if word in {".", ",", "!", "?", ":", ";", ")", "]", "}"}:
                yield word
            elif word.startswith("'"):
                yield word
            else:
                yield " " + word

    except Exception:
        logger.exception("Generation failed.")
        yield "⚠️ Error while generating response."

# ============================================================
# Full Text Generation
# ============================================================

def generate(
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    repetition_penalty: float = 1.15,
) -> str:
    """
    Generate a complete response as a string.
    """
    output = []
    for token in generate_stream(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    ):
        output.append(token)
    return "".join(output).strip()


def generate_with_confidence(
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    repetition_penalty: float = 1.15,
) -> tuple[str, float]:
    """
    Generate response and return average token probability.
    """
    if not load_model():
        return "⚠️ Model not loaded.", 0.0

    input_ids = [_tokenizer.bos_token_id]
    input_ids.extend(
        _tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )
    )

    generated, confidence = generate_ids_with_prob(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    new_tokens = generated[len(input_ids):]
    decoded_words = []
    for token in new_tokens:
        if token == _tokenizer.eos_token_id:
            break
        word = _tokenizer.inverse_vocab.get(token, "<unk>")
        if word in {".", ",", "!", "?", ":", ";", ")", "]", "}"}:
            decoded_words.append(word)
        elif word.startswith("'"):
            decoded_words.append(word)
        else:
            decoded_words.append(" " + word)

    text = "".join(decoded_words).strip()
    return text, confidence


# ============================================================
# Chat Interface
# ============================================================

def chat(
    prompt: str,
    system_prompt: str = "",
    max_new_tokens: int = 150,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
) -> str:
    """
    High-level chatbot interface.
    """

    if system_prompt.strip():

        final_prompt = (
            system_prompt.strip()
            + "\n\nUser: "
            + prompt.strip()
            + "\nAssistant:"
        )

    else:

        final_prompt = prompt.strip()

    return generate(
        prompt=final_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )


# ============================================================
# Reload Model
# ============================================================

def reload_model() -> bool:
    """
    Reload the model from disk.
    """

    unload_model()

    return load_model(force_reload=True)


# ============================================================
# Unload Model
# ============================================================

def unload_model():
    """
    Free model memory.
    Useful when retraining or shutting down.
    """

    global _model
    global _tokenizer
    global MODEL_READY

    _model = None
    _tokenizer = None
    MODEL_READY = False

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Model unloaded.")


# ============================================================
# Model Information
# ============================================================

def model_info() -> dict:
    """
    Returns runtime information.
    """

    info = {
        "loaded": MODEL_READY,
        "device": str(DEVICE),
    }

    if _model is not None:

        info["parameters"] = _model.get_num_params()

        info["block_size"] = _model.config.block_size

        info["embedding_size"] = _model.config.n_embd

        info["layers"] = _model.config.n_layer

        info["heads"] = _model.config.n_head

    else:

        info["parameters"] = 0

    return info


# ============================================================
# Auto Load
# ============================================================

try:

    load_model()

    if MODEL_READY:
        warmup()

except Exception:

    logger.exception(
        "Automatic model loading failed."
    )
