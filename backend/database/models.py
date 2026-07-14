"""
database/models.py
----------------------------------------------------
Genkit AI - Database Models
Plain Python dataclasses representing MySQL tables.
Database:
    MySQL Only
No SQLite
No SQLAlchemy
No ORM
Author : Genkit AI
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# ============================================================
# CHAT
# ============================================================
@dataclass(slots=True)
class Chat:
    session_id: str
    question: str
    answer: str
    intent: str = "general"
    source: str = "rag"
    confidence: float = 0.0
    id: Optional[int] = None
    created_at: Optional[datetime] = None

# ============================================================
# LEAD
# ============================================================
@dataclass(slots=True)
class Lead:
    name: str
    email: str
    phone: str = ""
    company: str = ""
    message: str = ""
    status: str = "New"
    id: Optional[int] = None
    created_at: Optional[datetime] = None

# ============================================================
# FEEDBACK
# ============================================================
@dataclass(slots=True)
class Feedback:
    session_id: str
    question: str
    answer: str
    rating: int
    comments: str = ""
    id: Optional[int] = None
    created_at: Optional[datetime] = None

# ============================================================
# USER PROFILE
# ============================================================
@dataclass(slots=True)
class UserProfile:
    session_id: str
    name: str = ""
    email: str = ""
    phone: str = ""
    company: str = ""
    interest: str = ""
    last_query: str = ""
    total_chats: int = 0
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

# ============================================================
# EXPORTS
# ============================================================
__all__ = [
    "Chat",
    "Lead",
    "Feedback",
    "UserProfile",
]
