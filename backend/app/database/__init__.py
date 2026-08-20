"""
backend/app/database/__init__.py
----------------------------------------------------
Database exports for Genkit AI V6.
"""

from app.database.connection import db_manager
from app.database.repository import ChatRepository, LeadRepository

__all__ = [
    "db_manager",
    "ChatRepository",
    "LeadRepository",
]
