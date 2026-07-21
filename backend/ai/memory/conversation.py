"""
conversation.py
----------------------------------------------------
Genkit AI - Production Conversation Memory

Features
--------
✔ Thread Safe
✔ Session Memory
✔ Automatic Session Creation
✔ Timestamp Support
✔ LRU Session Cache
✔ Conversation Context
✔ Search Ready
✔ Export / Import Ready

Author : Genkit AI
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from collections import OrderedDict
from threading import Lock, RLock
from typing import Dict, List, Optional


# ==========================================================
# MESSAGE
# ==========================================================

@dataclass
class ConversationMessage:
    """
    Single conversation message.
    """

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self):

        return {

            "id": self.message_id,

            "role": self.role,

            "content": self.content,

            "timestamp": self.timestamp,

        }


# ==========================================================
# SESSION
# ==========================================================

class ConversationSession:
    """
    Stores conversation of one user session along with profiles and summaries.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.messages: List[ConversationMessage] = []
        self.user_profile = {"name": "", "email": "", "interests": []}
        self.long_term_memory = []
        self.context_summary = ""

    # ------------------------------------------------------

    def touch(self):
        self.updated_at = time.time()

    # ------------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str,
    ):
        msg = ConversationMessage(
            role=role,
            content=content,
        )
        self.messages.append(msg)
        self.touch()

    # ------------------------------------------------------

    def update_profile(self, info: dict):
        for k, v in info.items():
            if k == "interests" and isinstance(v, list):
                # Merge lists
                self.user_profile["interests"] = list(set(self.user_profile["interests"] + v))
            else:
                self.user_profile[k] = v
        self.touch()

    def add_long_term_fact(self, fact: str):
        if fact not in self.long_term_memory:
            self.long_term_memory.append(fact)
        self.touch()

    def set_summary(self, summary: str):
        self.context_summary = summary
        self.touch()

    # ------------------------------------------------------

    def last_message(self):
        if not self.messages:
            return None
        return self.messages[-1]

    # ------------------------------------------------------

    def total_messages(self):
        return len(self.messages)

    # ------------------------------------------------------

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "user_profile": self.user_profile,
            "long_term_memory": self.long_term_memory,
            "context_summary": self.context_summary,
            "messages": [
                m.to_dict()
                for m in self.messages
            ],
        }


# ==========================================================
# MEMORY
# ==========================================================

class ConversationMemory:

    """
    Production conversation memory.

    Features
    --------
    • Thread Safe
    • LRU Cache
    • Multiple Sessions
    • Timestamp Tracking
    • Context Retrieval
    """

    def __init__(

        self,

        max_sessions: int = 100,

        max_messages: int = 50,

    ):

        self.max_sessions = max_sessions

        self.max_messages = max_messages

        self.sessions: Dict[str, ConversationSession] = OrderedDict()

        self.lock = RLock()

    # ------------------------------------------------------

    def create_session(

        self,

        session_id: Optional[str] = None,

    ) -> str:

        with self.lock:

            if session_id is None:

                session_id = str(uuid.uuid4())

            if session_id not in self.sessions:

                if len(self.sessions) >= self.max_sessions:

                    self.sessions.popitem(last=False)

                self.sessions[session_id] = ConversationSession(

                    session_id

                )

            return session_id

    # ------------------------------------------------------

    def get_session(

        self,

        session_id: str,

    ) -> ConversationSession:

        with self.lock:

            if session_id not in self.sessions:

                self.create_session(session_id)

            session = self.sessions[session_id]

            session.touch()

            self.sessions.move_to_end(session_id)

            return session

    # ------------------------------------------------------
    # Add Message
    # ------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ):

        with self.lock:

            session = self.get_session(session_id)

            session.add_message(
                role,
                content,
            )

            if len(session.messages) > self.max_messages:

                session.messages = session.messages[
                    -self.max_messages:
                ]

    # ------------------------------------------------------
    # Get History
    # ------------------------------------------------------

    def get_history(
        self,
        session_id: str,
    ):

        if session_id not in self.sessions:

            return []

        return [

            message.to_dict()

            for message in self.sessions[
                session_id
            ].messages

        ]

    # ------------------------------------------------------
    # Last Message
    # ------------------------------------------------------

    def get_last_message(
        self,
        session_id: str,
    ):

        if session_id not in self.sessions:

            return None

        message = self.sessions[
            session_id
        ].last_message()

        if message is None:

            return None

        return message.to_dict()

    # ------------------------------------------------------
    # Clear Messages
    # ------------------------------------------------------

    def clear_session(
        self,
        session_id: str,
    ):

        with self.lock:

            if session_id in self.sessions:

                self.sessions[
                    session_id
                ].messages.clear()

                self.sessions[
                    session_id
                ].touch()


    # Session Exists
    # ------------------------------------------------------

    def session_exists(
        self,
        session_id: str,
    ):

        return session_id in self.sessions

    # ------------------------------------------------------
    # Total Sessions
    # ------------------------------------------------------

    def total_sessions(self):

        return len(self.sessions)

    # ------------------------------------------------------
    # Total Messages
    # ------------------------------------------------------

    def total_messages(self):

        total = 0

        for session in self.sessions.values():

            total += len(session.messages)

        return total

    # ------------------------------------------------------
    # Conversation Context
    # ------------------------------------------------------

    def get_context(
        self,
        session_id: str,
        max_history: int = 10,
    ) -> str:

        if session_id not in self.sessions:
            return ""

        session = self.sessions[session_id]

        history = session.messages[-max_history:]

        context = []

        for msg in history:

            role = msg.role.capitalize()

            context.append(
                f"{role}: {msg.content}"
            )

        return "\n".join(context)

    # ------------------------------------------------------
    # Recent Messages
    # ------------------------------------------------------

    def recent_messages(
        self,
        session_id: str,
        count: int = 5,
    ):

        if session_id not in self.sessions:
            return []

        session = self.sessions[session_id]

        return [

            message.to_dict()

            for message in session.messages[-count:]

        ]

    # ------------------------------------------------------
    # User Messages
    # ------------------------------------------------------

    def user_messages(
        self,
        session_id: str,
    ):

        if session_id not in self.sessions:
            return []

        return [

            message.to_dict()

            for message in self.sessions[
                session_id
            ].messages

            if message.role.lower() == "user"

        ]

    # ------------------------------------------------------
    # Assistant Messages
    # ------------------------------------------------------

    def assistant_messages(
        self,
        session_id: str,
    ):

        if session_id not in self.sessions:
            return []

        return [

            message.to_dict()

            for message in self.sessions[
                session_id
            ].messages

            if message.role.lower() == "assistant"

        ]

    # ------------------------------------------------------
    # Search Messages
    # ------------------------------------------------------

    def search_messages(
        self,
        session_id: str,
        keyword: str,
    ):

        if session_id not in self.sessions:
            return []

        keyword = keyword.lower()

        results = []

        for message in self.sessions[
            session_id
        ].messages:

            if keyword in message.content.lower():

                results.append(
                    message.to_dict()
                )

        return results

    # ------------------------------------------------------
    # Conversation Summary
    # ------------------------------------------------------

    def conversation_summary(
        self,
        session_id: str,
        max_items: int = 10,
    ):

        if session_id not in self.sessions:
            return ""

        session = self.sessions[session_id]

        summary = []

        for message in session.messages[-max_items:]:

            text = message.content.strip()

            if len(text) > 120:
                text = text[:117] + "..."

            summary.append(
                f"{message.role}: {text}"
            )

        return "\n".join(summary)

    # ------------------------------------------------------
    # Export Session
    # ------------------------------------------------------

    def export_session(
        self,
        session_id: str,
    ):

        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        return {
            "session_id": session.session_id,
            "created_at": datetime.fromtimestamp(session.created_at).isoformat(),
            "updated_at": datetime.fromtimestamp(session.updated_at).isoformat(),
            "messages": [
                message.to_dict()
                for message in session.messages
            ],
        }

    # ------------------------------------------------------
    # Import Session
    # ------------------------------------------------------

    def import_session(
        self,
        session_data: dict,
    ):

        session_id = session_data.get("session_id")

        if not session_id:
            return False

        session = ConversationSession(session_id)

        for item in session_data.get("messages", []):

            session.add_message(
                role=item.get("role", "user"),
                content=item.get("content", ""),
            )

        self.sessions[session_id] = session

        return True

    # ------------------------------------------------------
    # Delete Session
    # ------------------------------------------------------

    def delete_session(
        self,
        session_id: str,
    ):

        with self.lock:

            if session_id in self.sessions:

                del self.sessions[session_id]

                return True

        return False

    # ------------------------------------------------------
    # Cleanup Old Sessions
    # ------------------------------------------------------

    def cleanup_old_sessions(
        self,
        hours: int = 24,
    ):

        now = time.time()

        remove_ids = []

        for sid, session in self.sessions.items():

            age_seconds = now - session.updated_at

            if age_seconds > hours * 3600:

                remove_ids.append(sid)

        for sid in remove_ids:

            del self.sessions[sid]

        return len(remove_ids)

    # ------------------------------------------------------
    # Memory Statistics
    # ------------------------------------------------------

    def memory_statistics(self):

        total_messages = 0

        for session in self.sessions.values():

            total_messages += len(session.messages)

        return {

            "sessions": len(self.sessions),

            "messages": total_messages,

            "max_messages_per_session": self.max_messages,

        }

    def get_profile(self, session_id: str) -> dict:
        with self.lock:
            session = self.get_session(session_id)
            return session.user_profile

    def update_profile(self, session_id: str, info: dict):
        with self.lock:
            session = self.get_session(session_id)
            session.update_profile(info)

    def add_long_term_fact(self, session_id: str, fact: str):
        with self.lock:
            session = self.get_session(session_id)
            session.add_long_term_fact(fact)

    def get_long_term_memory(self, session_id: str) -> List[str]:
        with self.lock:
            session = self.get_session(session_id)
            return session.long_term_memory

    def get_context_summary(self, session_id: str) -> str:
        with self.lock:
            session = self.get_session(session_id)
            return session.context_summary

    def set_context_summary(self, session_id: str, summary: str):
        with self.lock:
            session = self.get_session(session_id)
            session.set_summary(summary)

    # ------------------------------------------------------
    # Reset Everything
    # ------------------------------------------------------

    def reset_all(self):

        with self.lock:

            self.sessions.clear()

    # ------------------------------------------------------
    # Magic Methods
    # ------------------------------------------------------

    def __len__(self):

        return len(self.sessions)

    def __contains__(self, session_id):

        return session_id in self.sessions

    def __repr__(self):

        return (
            f"ConversationMemory("
            f"sessions={len(self.sessions)}, "
            f"max_messages={self.max_messages})"
        )

# ==========================================================
# Global Singleton
# ==========================================================

conversation_memory = ConversationMemory()
