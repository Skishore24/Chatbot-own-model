import os
import json
import re
from typing import List
import numpy as np

from app.config import DATASET_PATH, logger

# ─────────────────────────────────────────────
# EMBEDDING MODEL (LOCAL, FAST, GOOD QUALITY)
# ─────────────────────────────────────────────
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ─────────────────────────────────────────────
# GLOBAL STORAGE
# ─────────────────────────────────────────────
documents: List[str] = []
embeddings = None

# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─────────────────────────────────────────────
# LOAD + SPLIT DATASET (BETTER STRUCTURE)
# ─────────────────────────────────────────────
def load_and_split():
    if not os.path.exists(DATASET_PATH):
        logger.warning("Dataset not found")
        return []

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = []

    for item in data:
        q = clean_text(item.get("instruction", ""))
        a = clean_text(item.get("output", ""))

        # 🔥 keep structured format
        chunk = f"Q: {q}\nA: {a}"

        if len(chunk) < 25:
            continue

        chunks.append(chunk)

    logger.info(f"✅ Loaded {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────────
# CREATE EMBEDDINGS
# ─────────────────────────────────────────────
def add_documents(docs: List[str]):
    global documents, embeddings

    if not docs:
        logger.warning("No documents to embed")
        return

    documents = docs

    embeddings = embedding_model.encode(
        docs,
        convert_to_numpy=True,
        normalize_embeddings=True  # 🔥 important for cosine
    )

    logger.info("✅ Embeddings created successfully")


# ─────────────────────────────────────────────
# FAST COSINE SIMILARITY
# ─────────────────────────────────────────────
def cosine_sim_matrix(query_vec):
    return np.dot(embeddings, query_vec)


# ─────────────────────────────────────────────
# 🔥 SMART SEARCH (PRODUCTION)
# ─────────────────────────────────────────────
def search(query: str, top_k: int = 3):

    if embeddings is None or len(documents) == 0:
        return ""

    query_clean = clean_text(query)

    # 🔥 embedding search
    query_vec = embedding_model.encode(
        [query_clean],
        normalize_embeddings=True
    )[0]

    sims = cosine_sim_matrix(query_vec)

    # get top indices
    top_idx = np.argsort(sims)[::-1][:top_k * 2]

    results = []

    for i in top_idx:
        score = sims[i]

        # 🔥 HARD FILTER (kills hallucination)
        if score < 0.45:
            continue

        doc = documents[i]

        # 🔥 keyword filter (strong)
        if not any(word in doc for word in query_clean.split()):
            continue

        results.append((score, doc))

    # sort final
    results = sorted(results, key=lambda x: x[0], reverse=True)

    # take best
    final_docs = [doc for _, doc in results[:top_k]]

    return clean_results(final_docs)


# ─────────────────────────────────────────────
# CLEAN OUTPUT (VERY IMPORTANT)
# ─────────────────────────────────────────────
def clean_results(results: List[str]) -> str:
    final = []

    seen = set()

    for r in results:
        if not r or r in seen:
            continue

        seen.add(r)

        # remove Q/A labels for model clarity
        r = r.replace("Q:", "").replace("A:", "").strip()

        # limit size
        if len(r) > 200:
            r = r[:200]

        final.append(r)

    return "\n".join(final)


# ─────────────────────────────────────────────
# 🔥 MEMORY (IMPROVED)
# ─────────────────────────────────────────────
memory_store = {}

def save_memory(session_id: str, text: str):
    if session_id not in memory_store:
        memory_store[session_id] = []

    memory_store[session_id].append(text)

    # keep last 5
    memory_store[session_id] = memory_store[session_id][-5:]


def search_memory(query: str, session_id: str):
    if session_id not in memory_store:
        return ""

    query_clean = clean_text(query)

    results = []

    for item in memory_store[session_id]:
        if any(word in item.lower() for word in query_clean.split()):
            results.append(item)

    return "\n".join(results[-2:])