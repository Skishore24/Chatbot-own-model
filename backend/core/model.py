from typing import Iterator
from rag.retriever import retrieve
from rag.reranker import rerank
from services.vector_store import search_memory, save_memory

from core.brain import generate_stream
from core.intent import detect_intent
from services.memory import update_user_info, get_user_info, save_lead_to_db
from utils.helpers import is_valid_query, detect_lead
from app.config import logger

FALLBACK = "I can help only with Genkit services like web, video, and design."


def build_context(docs):
    return "\n".join(docs[:5]) if docs else ""


def trim_memory(memory: str):
    return "\n".join(memory.split("\n")[-3:]) if memory else ""


def clean_response(text: str):
    lines = text.strip().split("\n")
    result = []
    seen = set()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if not line.startswith("•"):
            line = "• " + line

        key = line.lower()[:40]
        if key in seen:
            continue

        seen.add(key)
        result.append(" ".join(line.split()[:25]))

        if len(result) >= 3:
            break

    return "\n".join(result)


def is_grounded(response: str, context: str):
    if not response:
        return False

    ctx = set(context.lower().split())
    res = set(response.lower().split())

    return len(ctx & res) >= 2


def get_answer(query: str, session_id: str) -> Iterator[str]:

    if not is_valid_query(query):
        yield FALLBACK
        return

    update_user_info(session_id, query)
    user = get_user_info(session_id)

    guard = detect_intent(query)
    if guard["is_out_of_scope"]:
        yield FALLBACK
        return

    docs = rerank(query, retrieve(query))
    memory = trim_memory(search_memory(query, session_id))

    if not docs and not memory:
        yield FALLBACK
        return

    context = build_context(docs)

    prompt = f"""### SYSTEM:
You are Genkit AI assistant.

STRICT RULES:
- Answer ONLY from CONTEXT
- Use 2–3 bullet points

### CONTEXT:
{context}

### MEMORY:
{memory}

### USER:
{query}

### ASSISTANT:
"""

    try:
        raw = ""

        for chunk in generate_stream(prompt):
            raw += chunk

        cleaned = clean_response(raw)

        if not cleaned or not is_grounded(cleaned, context):
            yield FALLBACK
            return

        for line in cleaned.split("\n"):
            yield line + "\n"

        save_memory(session_id, f"{query} → {cleaned[:80]}")

        if detect_lead(query):
            name = user.get("name") or "Visitor"
            email = user.get("email") or "lead@genkit.in"
            save_lead_to_db(name, email)

    except Exception as e:
        logger.exception("Pipeline error")
        yield "⚠️ Something went wrong."