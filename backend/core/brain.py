from typing import Iterator
from services.vector_store import search, search_memory, save_memory
from core.model import generate_stream
from core.intent import detect_intent
from services.memory import (
    update_user_info,
    get_user_info,
    save_lead_to_db,
)
from utils.helpers import detect_lead
from app.config import logger
from db.database import get_connection


# ─────────────────────────────────────────────
# VALIDATION (HALLUCINATION KILLER)
# ─────────────────────────────────────────────
def is_valid_response(response: str, context: str) -> bool:
    if not response:
        return False

    if len(response.split()) < 3:
        return False

    context_words = set(context.lower().split())
    response_words = set(response.lower().split())

    overlap = len(context_words & response_words)

    return overlap >= 2


# ─────────────────────────────────────────────
# CLEAN CONTEXT
# ─────────────────────────────────────────────
def clean_context(context: str) -> str:
    lines = context.split("\n")
    clean = []

    for l in lines:
        l = l.strip()

        if len(l) < 20:
            continue

        clean.append(l)

        if len(clean) >= 2:
            break

    return "\n".join(clean)


# ─────────────────────────────────────────────
# CLEAN MEMORY
# ─────────────────────────────────────────────
def clean_memory(memory: str) -> str:
    if not memory:
        return ""

    lines = memory.split("\n")
    return "\n".join(lines[-2:])


# ─────────────────────────────────────────────
# CLEAN RESPONSE
# ─────────────────────────────────────────────
def clean_response(text: str) -> str:
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if not line.startswith("•"):
            line = "• " + line.lstrip("-* ")

        words = line.split()[:10]
        cleaned.append(" ".join(words))

        if len(cleaned) >= 3:
            break

    return "\n".join(cleaned)


# ─────────────────────────────────────────────
# AUTO DATASET SAVE
# ─────────────────────────────────────────────
def save_training_data(question: str, answer: str):
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO feedback (question, answer, rating) VALUES (?, ?, ?)",
                (question, answer, 5)
            )
            conn.commit()
    except:
        pass


# ─────────────────────────────────────────────
# CORE AI
# ─────────────────────────────────────────────
def get_answer(query: str, session_id: str) -> Iterator[str]:

    # 1. User memory
    update_user_info(session_id, query)
    user = get_user_info(session_id)

    # 2. Intent check
    guard = detect_intent(query)
    if guard["is_out_of_scope"]:
        yield "I can help only with Genkit services like web, video, and design."
        return

    # 3. Get context (RAG)
    context = search(query)

    if not context:
        yield "I can help only with Genkit services like web, video, and design."
        return

    context = clean_context(context)

    # 4. Get memory
    memory = clean_memory(search_memory(query, session_id))

    # 5. Prompt (STRICT)
    prompt = f"""
You are Genkit AI.

STRICT RULES:
- Answer ONLY from CONTEXT
- DO NOT add extra info
- If not found → say EXACTLY:
  "I can help only with Genkit services"

FORMAT:
• Max 3 bullet points
• Each point 5–10 words

CONTEXT:
{context}

MEMORY:
{memory}

QUESTION:
{query}

ANSWER:
"""

    try:
        # 6. Generate FULL response
        full_response = ""

        for chunk in generate_stream(prompt):
            full_response += chunk

        # 7. Clean
        full_response = full_response[:300]
        full_response = clean_response(full_response)

        # 8. HARD FAIL SAFE
        if "•" not in full_response:
            yield "I can help only with Genkit services."
            return

        # 9. Validate
        if not is_valid_response(full_response, context):
            logger.warning("⚠️ Invalid response blocked")
            yield "I can help only with Genkit services."
            return

        # 10. Output
        for line in full_response.split("\n"):
            yield line + "\n"

        # 11. Save memory
        save_memory(session_id, f"User: {query} | Assistant: {full_response}")

        # 12. Auto dataset learning
        save_training_data(query, full_response)

    except Exception as e:
        logger.error(f"Error: {e}")
        yield "⚠️ Something went wrong."

    # 13. Lead capture
    if detect_lead(query) or guard["intent"] in ["pricing", "contact"]:
        save_lead_to_db(user.get("name", "Client"), "Interested via chatbot")