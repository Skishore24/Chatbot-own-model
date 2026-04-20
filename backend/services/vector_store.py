import os
import json
import re
from typing import List
import numpy as np
from threading import Lock

from sentence_transformers import SentenceTransformer
from app.config import DATASET_PATH, logger

_model = SentenceTransformer("all-MiniLM-L6-v2")

_documents: List[str] = []
_embeddings = None
_initialized = False
_lock = Lock()


# ─────────────────────────────
# CLEAN
# ─────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# ─────────────────────────────
# LOAD DATASET
# ─────────────────────────────
def _load_dataset():
    if not os.path.exists(DATASET_PATH):
        logger.warning("Dataset not found")
        return []

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = []
    for item in data:
        q = clean_text(item.get("instruction", ""))
        a = clean_text(item.get("output", ""))

        if len(q) < 5 or len(a) < 5:
            continue

        chunks.append(f"{q}\n{a}")

    logger.info(f"Loaded {len(chunks)} chunks")
    return chunks


# ─────────────────────────────
# INIT (AUTO FIX FOR YOUR PROBLEM)
# ─────────────────────────────
def _init():
    global _documents, _embeddings, _initialized

    if _initialized:
        return

    with _lock:
        if _initialized:
            return

        docs = _load_dataset()

        if not docs:
            logger.warning("No dataset loaded")
            return

        _documents = docs

        _embeddings = _model.encode(
            docs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32
        )

        _initialized = True
        logger.info("Vector store initialized")


# ─────────────────────────────
# SEARCH
# ─────────────────────────────
def search(query: str, top_k: int = 3) -> List[str]:
    _init()

    if _embeddings is None or not _documents:
        return []

    q_vec = _model.encode(
        [clean_text(query)],
        normalize_embeddings=True
    )[0]

    sims = np.dot(_embeddings, q_vec)

    top_idx = np.argsort(sims)[::-1][:top_k * 3]

    results = []

    for i in top_idx:
        score = float(sims[i])

        if score < 0.30:
            continue

        results.append((score, _documents[i]))

    results.sort(key=lambda x: x[0], reverse=True)

    return [doc[:300] for _, doc in results[:top_k]]


# ─────────────────────────────
# SESSION MEMORY (SMART)
# ─────────────────────────────
_memory = {}


def save_memory(session_id: str, text: str):
    if session_id not in _memory:
        _memory[session_id] = []

    _memory[session_id].append(text)
    _memory[session_id] = _memory[session_id][-5:]


def search_memory(query: str, session_id: str) -> str:
    if session_id not in _memory:
        return ""

    q = clean_text(query)

    scored = []

    for m in _memory[session_id]:
        score = sum(1 for w in q.split() if w in m.lower())
        if score > 0:
            scored.append((score, m))

    scored.sort(reverse=True)

    return "\n".join([m for _, m in scored[:3]])


# ─────────────────────────────
# COMPATIBILITY (FOR MAIN.PY)
# ─────────────────────────────
def load_and_split():
    """Compatibility layer for main.py"""
    return _load_dataset()

def add_documents(docs: List[str]):
    """Compatibility layer for main.py"""
    global _documents, _embeddings, _initialized
    if not docs: return
    
    with _lock:
        _documents = docs
        _embeddings = _model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
        _initialized = True
        logger.info(f"Vector store loaded with {len(docs)} documents")