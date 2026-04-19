from typing import Iterator
from services.vector_store import search, search_memory, save_memory
from core.model import generate_stream
from core.intent import detect_intent
from core.reasoning import build_reasoning
from services.memory import (
    update_user_info,
    get_history,
    get_user_info,
    save_lead_to_db,
    summarize_history
)
from utils.helpers import detect_lead
from app.config import logger

# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
def is_valid_response(response: str, context: str) -> bool:
    """Checks if the response has minimal grounding in the context."""
    if not response or len(response) < 5:
        return False
    
    # Check for word overlap (simplified grounding check)
    r_words = set(response.lower().split())
    c_words = set(context.lower().split())
    overlap = len(r_words & c_words)
    
    # If the response is short but shares context words, or it's long enough
    return overlap >= 2 or len(response) > 50

# ─────────────────────────────────────────────
# CORE AI LOGIC
# ─────────────────────────────────────────────
def get_answer(query: str, session_id: str) -> Iterator[str]:
    """
    Production-grade AI pipeline:
    1. Memory & User Profile retrieval
    2. Intent & Domain Guard
    3. RAG (Knowledge + Semantic Memory)
    4. Reasoning & Prompting
    5. Streaming Generation & Validation
    """
    
    # 1. Update Profile & State
    update_user_info(session_id, query)
    user = get_user_info(session_id)
    history = get_history(session_id)

    # 2. Domain Guard
    guard = detect_intent(query)
    if guard["is_out_of_scope"]:
        yield "I am the Genkit AI assistant. I only answer questions related to Genkit's services (Web, Video, Design). How can I help you with those?"
        return

    # 3. RAG Context (Knowledge Base)
    context = search(query)
    if not context:
        # If no specific context found, use a general professional fallback
        context = "Genkit is a tech and creative solutions provider specializing in Web Development, Video Editing, and Graphic Design."

    # 4. Memory Retrieval (Conversational Context)
    history_text = "\n".join([f"{h['role']}: {h['message']}" for h in history[-3:]])
    summary = summarize_history(history)
    memory_context = search_memory(query, session_id)

    # 5. Reasoning Layer
    reasoning = build_reasoning(query, context)

    # 6. Prompt Construction
    user_context = f"User Name: {user.get('name', 'Client')}\nInterest: {user.get('interest', 'General')}"
    
    prompt = f"""You are the official Genkit AI Assistant. 
STRICT RULES:
- Answer ONLY using the CONTEXT provided.
- If the answer isn't in the CONTEXT, say you only help with Genkit's specific services.
- Be professional, concise (1-2 lines), and friendly.
- Use USER context for personalization.

USER PROFILE:
{user_context}

SUMMARY OF PAST CHATS:
{summary}

RELEVANT MEMORY:
{memory_context}

RECENT HISTORY:
{history_text}

THINKING STEPS:
{reasoning}

CONTEXT (KNOWLEDGE BASE):
{context}

QUESTION:
{query}

ANSWER:
"""

    # 7. Generation Loop
    full_response = ""
    try:
        for chunk in generate_stream(prompt):
            full_response += chunk
            yield chunk
            
        # Post-generation Validation
        if not is_valid_response(full_response, context):
            logger.warning(f"Response rejected by validation: {full_response[:50]}...")
            # We don't yield here since we've already streamed, 
            # but we can log for quality monitoring.

        # 8. Persistent Semantic Memory
        save_memory(session_id, f"User: {query} | Assistant: {full_response}")

    except Exception as e:
        logger.error(f"Generation error: {e}")
        yield "⚠️ I'm sorry, I encountered an error. Please try rephrasing your question."

    # 9. Business Value: Lead Detection
    if detect_lead(query) or guard["intent"] in ["pricing", "contact"]:
        save_lead_to_db(user.get("name", "Interested Client"), "Identified via chat")