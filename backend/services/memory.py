import re
from collections import OrderedDict
from typing import List, Dict
from threading import Lock

from app.config import logger
from db.database import get_connection

chat_history = OrderedDict()
user_profiles = {}

MAX_MESSAGES = 30
_lock = Lock()


# ─────────────────────────────
# CHAT MEMORY
# ─────────────────────────────
def add_message(session_id: str, role: str, message: str):
    with _lock:
        if session_id not in chat_history:
            chat_history[session_id] = []

        chat_history[session_id].append({
            "role": role,
            "message": message
        })

        if len(chat_history[session_id]) > MAX_MESSAGES:
            chat_history[session_id].pop(0)


def get_history(session_id: str) -> List[Dict[str, str]]:
    return chat_history.get(session_id, [])


# ─────────────────────────────
# USER MEMORY
# ─────────────────────────────
def update_user_info(session_id: str, text: str):
    with _lock:
        if session_id not in user_profiles:
            user_profiles[session_id] = {"name": "", "interest": ""}

        t = text.lower()

        match = re.search(r"(my name is|i am|call me)\s+([\w]+)", t)
        if match:
            user_profiles[session_id]["name"] = match.group(2).capitalize()

        if any(x in t for x in ["website", "app", "ai", "chatbot", "pricing"]):
            user_profiles[session_id]["interest"] = text


def get_user_info(session_id: str) -> Dict[str, str]:
    return user_profiles.get(session_id, {})


# ─────────────────────────────
# DATABASE
# ─────────────────────────────
def save_chat_to_db(session_id: str, question: str, answer: str):
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO chats (session_id, question, answer) VALUES (?, ?, ?)",
                (session_id, question, answer)
            )
    except Exception as e:
        logger.error(f"DB chat save error: {e}")


def save_lead_to_db(name: str, email: str):
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO leads (name, email) VALUES (?, ?)",
                (name, email)
            )
            logger.info(f"Lead saved: {email}")
    except Exception as e:
        logger.error(f"Lead save error: {e}")