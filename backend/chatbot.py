"""
chatbot.py
----------------------------------------------------
Genkit AI - Production Chat Pipeline
Architecture
1. Query Validation
2. Spell Correction
3. Intent Detection
4. User Profile Update
5. Hybrid RAG Retrieval
6. Re-ranking
7. Context Builder
8. Memory Retrieval
9. LLM Generation
10. Grounding Check
11. Response Formatting
12. Lead Detection
13. Database Storage
Author : Genkit AI
"""
import os
import sys
import json
import re
from typing import Iterator, List, Dict, Optional, Tuple
sys.path.insert(
    0,
    os.path.dirname(
        os.path.abspath(__file__)
    ),
)
from config import (
    DATASET_PATH,
    logger,
)
from database import (
    update_user_profile,
    get_user_profile,
    increment_chat_count,
    save_chat_to_db,
    save_lead_to_db,
)
from ai.embeddings.embedding import (
    _store,
    save_memory,
    search_memory,
)
from ai.rag.retriever import Retriever
from ai.rag.reranker import Reranker
from ai.rag.context_builder import ContextBuilder
from ai.llm.inference import generate
from ai.preprocessing.cleaner import cleaner
from ai.preprocessing.spell import spell_checker
from utils.helper import (
    clean_query,
    detect_intent,
    detect_lead,
    is_valid_query,
    format_response,
    is_grounded,
)
# ============================================================
# INITIALIZE COMPONENTS
# ============================================================
retriever = Retriever()
reranker = Reranker()
context_builder = ContextBuilder()
_store.init()
# ============================================================
# DEFAULT RESPONSES
# ============================================================
WELCOME_MESSAGE = (
    "Hello! Welcome to Genkit AI."
)
INVALID_QUERY = (
    "Please enter a valid question."
)
OUT_OF_SCOPE = (
    "I can assist only with Genkit services, products and business information."
)
NO_MATCH = (
    "I couldn't find information about that in the Genkit knowledge base."
)
SYSTEM_PROMPT = """
You are Genkit AI.
Rules
- Answer ONLY from the provided context.
- Never invent information.
- Never answer unrelated questions.
- If context is empty, politely say you don't know.
- Keep answers professional.
- Use bullet points when appropriate.
- Mention Genkit naturally.
"""
# ============================================================
# DATASET CACHE
# ============================================================
_DATASET = []
_DATASET_READY = False
# ============================================================
# DATASET LOADER
# ============================================================
def load_dataset():
    """
    Load Genkit dataset into memory.
    """
    global _DATASET
    global _DATASET_READY
    if _DATASET_READY:
        return
    if not os.path.exists(DATASET_PATH):
        logger.warning(
            "Dataset not found."
        )
        _DATASET_READY = True
        return
    try:
        with open(
            DATASET_PATH,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)
        _DATASET = []
        for item in data:
            instruction = item.get(
                "instruction",
                ""
            ).strip()
            output = item.get(
                "output",
                ""
            ).strip()
            if not instruction:
                continue
            if not output:
                continue
            _DATASET.append({
                "instruction": instruction,
                "output": output
            })
        logger.info(
            f"Loaded {len(_DATASET)} QA pairs."
        )
        _DATASET_READY = True
    except Exception:
        logger.exception(
            "Dataset loading failed."
        )
        _DATASET_READY = True

# ============================================================
# TOKEN OVERLAP SCORE
# ============================================================
def overlap_score(
    query,
    text
):
    query_words = set(
        re.findall(
            r"\w+",
            query.lower()
        )
    )
    text_words = set(
        re.findall(
            r"\w+",
            text.lower()
        )
    )
    if not query_words:
        return 0.0
    return len(
        query_words & text_words
    ) / len(
        query_words
    )

# ============================================================
# EXACT DATASET MATCH
# ============================================================
def exact_dataset_match(
    query
):
    load_dataset()
    query = clean_query(query)
    for item in _DATASET:
        if clean_query(
            item["instruction"]
        ) == query:
            return item["output"]
    return None

# ============================================================
# FUZZY DATASET MATCH
# ============================================================
def fuzzy_dataset_match(
    query
):
    load_dataset()
    best_score = 0
    best_answer = None
    query = clean_query(query)
    for item in _DATASET:
        score = overlap_score(
            query,
            item["instruction"]
        )
        if score > best_score:
            best_score = score
            best_answer = item["output"]
    if best_score >= 0.60:
        return best_answer
    return None

# ============================================================
# HYBRID RETRIEVAL
# ============================================================
def retrieve_documents(
    query,
    top_k=5
):
    try:
        documents = retriever.retrieve(
            query,
            top_k=top_k
        )
        documents = reranker.rerank(
            query,
            documents,
            top_k=top_k
        )
        return documents
    except Exception:
        logger.exception(
            "Retriever failed."
        )
        return []

# ============================================================
# CONTEXT
# ============================================================
def build_context(
    query,
    documents,
    history
):
    return context_builder.build(
        query=query,
        documents=documents,
        history=history
    )
# ============================================================
# MAIN CHAT PIPELINE
# ============================================================
def get_answer(
    query: str,
    session_id: str
) -> Iterator[str]:
    """
    Main Genkit AI pipeline.
    """
    # --------------------------------------------------------
    # Query Validation
    # --------------------------------------------------------
    query = cleaner.clean(query)
    query = spell_checker.correct(query)
    if not is_valid_query(query):
        yield INVALID_QUERY
        return
    # --------------------------------------------------------
    # Intent Detection
    # --------------------------------------------------------
    intent = detect_intent(query)
    if intent.get("is_out_of_scope", False):
        yield OUT_OF_SCOPE
        return
    # --------------------------------------------------------
    # Update User Profile
    # --------------------------------------------------------
    try:
        update_user_profile(
            session_id=session_id,
            last_query=query
        )
        increment_chat_count(session_id)
        profile = get_user_profile(session_id)
    except Exception:
        logger.exception(
            "User profile update failed."
        )
        profile = {}
    # --------------------------------------------------------
    # Exact Dataset Match
    # --------------------------------------------------------
    answer = exact_dataset_match(query)
    if answer:
        answer = format_response(answer)
        save_chat_to_db(
            session_id=session_id,
            question=query,
            answer=answer,
            intent=intent["intent"],
            source="dataset",
            confidence=1.0
        )
        yield answer
        return
    # --------------------------------------------------------
    # Fuzzy Dataset Match
    # --------------------------------------------------------
    answer = fuzzy_dataset_match(query)
    if answer:
        answer = format_response(answer)
        save_chat_to_db(
            session_id=session_id,
            question=query,
            answer=answer,
            intent=intent["intent"],
            source="dataset",
            confidence=0.90
        )
        yield answer
        return
    # --------------------------------------------------------
    # Retrieve Documents
    # --------------------------------------------------------
    documents = retrieve_documents(query)
    # --------------------------------------------------------
    # Conversation Memory
    # --------------------------------------------------------
    history = []
    try:
        memory = search_memory(
            query,
            session_id
        )
        if memory:
            history = memory
    except Exception:
        logger.exception(
            "Memory retrieval failed."
        )
    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------
    prompt = build_context(
        query=query,
        documents=documents,
        history=history
    )
    if not prompt:
        yield NO_MATCH
        return
    # --------------------------------------------------------
    # Generate Response
    # --------------------------------------------------------
    try:
        answer = generate(
            prompt,
            max_new_tokens=200
        )
    except Exception:
        logger.exception(
            "LLM generation failed."
        )
        yield NO_MATCH
        return
    # --------------------------------------------------------
    # Grounding Check
    # --------------------------------------------------------
    context = "\n".join(
        d["text"]
        for d in documents
    )
    if not is_grounded(
        answer,
        context
    ):
        answer = NO_MATCH
    answer = format_response(answer)
    # --------------------------------------------------------
    # Store Memory
    # --------------------------------------------------------
    try:
        save_memory(
            session_id,
            f"{query}\n{answer}"
        )
    except Exception:
        logger.exception(
            "Memory save failed."
        )
    # --------------------------------------------------------
    # Save Chat
    # --------------------------------------------------------
    try:
        save_chat_to_db(
            session_id=session_id,
            question=query,
            answer=answer,
            intent=intent["intent"],
            source="rag",
            confidence=intent["confidence"]
        )
    except Exception:
        logger.exception(
            "Chat save failed."
        )
    # --------------------------------------------------------
    # Lead Detection
    # --------------------------------------------------------
    try:
        if detect_lead(query):
            save_lead_to_db(
                name=profile.get(
                    "name",
                    "Visitor"
                ),
                email=profile.get(
                    "email",
                    "visitor@genkit.in"
                )
            )
    except Exception:
        logger.exception(
            "Lead save failed."
        )
    # --------------------------------------------------------
    # Stream Response
    # --------------------------------------------------------
    for line in answer.split("\n"):
        line = line.strip()
        if line:
            yield line + "\n"
# ============================================================
# STARTUP
# ============================================================
def initialize_chatbot():
    """
    Initialize chatbot components.
    Safe to call multiple times.
    """
    try:
        logger.info("=" * 60)
        logger.info("Initializing Genkit AI Chatbot...")
        load_dataset()
        _store.init()
        logger.info(
            "Knowledge Base Ready."
        )
        logger.info(
            f"Dataset Size : {len(_DATASET)}"
        )
        logger.info("=" * 60)
    except Exception:
        logger.exception(
            "Chatbot initialization failed."
        )

# ============================================================
# HEALTH CHECK
# ============================================================
def chatbot_health():
    return {
        "dataset_loaded": _DATASET_READY,
        "dataset_size": len(_DATASET),
        "vector_store": _store._initialized,
        "retriever": retriever is not None,
        "reranker": reranker is not None,
        "context_builder": context_builder is not None
    }

# ============================================================
# RELOAD DATASET
# ============================================================
def reload_dataset():
    global _DATASET
    global _DATASET_READY
    _DATASET = []
    _DATASET_READY = False
    load_dataset()
    logger.info(
        "Dataset reloaded."
    )

# ============================================================
# CLEAR MEMORY
# ============================================================
def clear_memory(session_id: str):
    try:
        save_memory(
            session_id,
            ""
        )
    except Exception:
        pass

# ============================================================
# VERSION
# ============================================================
CHATBOT_VERSION = "2.0.0"

# ============================================================
# EXPORTS
# ============================================================
__all__ = [
    "get_answer",
    "initialize_chatbot",
    "chatbot_health",
    "reload_dataset",
    "clear_memory",
    "CHATBOT_VERSION"
]

# ============================================================
# AUTO INITIALIZATION
# ============================================================
try:
    initialize_chatbot()
except Exception:
    logger.exception(
        "Automatic chatbot initialization failed."
    )
