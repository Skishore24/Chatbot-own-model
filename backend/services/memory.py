import re
from collections import OrderedDict
from typing import List, Dict, Any
from app.config import logger
from db.database import get_connection

# In-memory session stores
chat_history = OrderedDict()
user_profiles = {}

MAX_MESSAGES = 30

# ───────── CHAT MEMORY ─────────
def add_message(session_id: str, role: str, message: str):
    """Adds a message to the in-memory session history."""
    if session_id not in chat_history:
        chat_history[session_id] = []

    chat_history[session_id].append({
        "role": role,
        "message": message
    })

    # Keep history within limits
    if len(chat_history[session_id]) > MAX_MESSAGES:
        chat_history[session_id].pop(0)

def get_history(session_id: str) -> List[Dict[str, str]]:
    """Retrieves message history for a session."""
    return chat_history.get(session_id, [])

# ───────── USER MEMORY ─────────
def update_user_info(session_id: str, text: str):
    """Extracts user information (name, interest) from text."""
    if session_id not in user_profiles:
        user_profiles[session_id] = {"name": "", "interest": ""}

    t = text.lower()

    # Extract Name
    match = re.search(r"(my name is|i am|call me)\s+([\w]+)", t)
    if match:
        user_profiles[session_id]["name"] = match.group(2).capitalize()

    # Extract Interest
    if any(x in t for x in ["website", "app", "ai", "chatbot", "development", "pricing"]):
        user_profiles[session_id]["interest"] = text

def get_user_info(session_id: str) -> Dict[str, str]:
    """Retrieves stored user profile for a session."""
    return user_profiles.get(session_id, {})

# ───────── DATABASE STORAGE ─────────
def save_chat_to_db(session_id: str, question: str, answer: str):
    """Persists a chat interaction to the SQLite database."""
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO chats (session_id, question, answer) VALUES (?, ?, ?)",
                (session_id, question, answer)
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to save chat to DB: {e}")

def save_lead_to_db(name: str, email: str):
    """Saves a lead (name, email) to the database."""
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO leads (name, email) VALUES (?, ?)",
                (name, email)
            )
            conn.commit()
            logger.info(f"Lead saved successfully: {email}")
    except Exception as e:
        logger.error(f"Failed to save lead to DB: {e}")

def summarize_history(history):

    if len(history) < 6:
        return ""

    summary = []

    for h in history[-6:]:
        summary.append(f"{h['role']}: {h['message']}")

    return " | ".join(summary[:3])