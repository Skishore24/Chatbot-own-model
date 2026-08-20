"""
backend/app/database/repository.py
----------------------------------------------------
Data Access Repository for MySQL (Chat Sessions, Messages, Leads, Feedback).
All queries are 100% parameterized against SQL injection.
"""

from typing import List, Optional
from app.database.connection import db_manager


class ChatRepository:
    """Repository managing chat messages and sessions in MySQL."""

    @staticmethod
    def record_session(session_id: str) -> None:
        """Upserts a chat session record in MySQL."""
        query = "INSERT IGNORE INTO chat_sessions (session_id) VALUES (%s)"
        db_manager.execute_write(query, (session_id,))

    @staticmethod
    def save_message(
        session_id: str,
        role_or_sender: str,
        content: str,
        intent: Optional[str] = None,
        confidence_score: Optional[float] = None,
        tokens_generated: Optional[int] = None,
        latency_ms: Optional[float] = None,
    ) -> Optional[int]:
        """
        Saves a chat message into MySQL chat_messages table.
        Normalizes role to 'user' or 'assistant'.
        """
        ChatRepository.record_session(session_id)
        
        # Normalize role value
        role_normalized = "assistant" if role_or_sender.lower() in ("bot", "assistant", "ai") else "user"

        query = (
            "INSERT INTO chat_messages "
            "(session_id, role, content, intent, confidence_score, tokens_generated, latency_ms) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)"
        )
        return db_manager.execute_write(
            query,
            (
                session_id,
                role_normalized,
                content,
                intent,
                confidence_score,
                tokens_generated,
                latency_ms,
            ),
        )

    @staticmethod
    def get_session_history(session_id: str, limit: int = 50) -> List[dict]:
        """Retrieves chronological message history for a session."""
        query = (
            "SELECT id, role as sender, content as text, created_at "
            "FROM chat_messages WHERE session_id = %s "
            "ORDER BY id ASC LIMIT %s"
        )
        return db_manager.execute_read(query, (session_id, limit))


class LeadRepository:
    """Repository managing captured business leads in MySQL."""

    @staticmethod
    def create_lead(
        name: str,
        email: str,
        phone: Optional[str] = None,
        company: Optional[str] = None,
        message: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[int]:
        """Captures a new lead or updates existing lead with parameterized inputs into MySQL."""
        query = (
            "INSERT INTO leads (name, email, phone, company, message, session_id) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE "
            "name = VALUES(name), "
            "phone = VALUES(phone), "
            "company = VALUES(company), "
            "message = VALUES(message), "
            "session_id = VALUES(session_id), "
            "status = 'New'"
        )
        return db_manager.execute_write(
            query,
            (name, email, phone or "", company or "", message or "", session_id or ""),
        )

    @staticmethod
    def get_leads(limit: int = 100) -> List[dict]:
        """Fetches recent leads from MySQL."""
        query = "SELECT id, name, email, phone, company, message, status, session_id, created_at FROM leads ORDER BY id DESC LIMIT %s"
        return db_manager.execute_read(query, (limit,))


class FeedbackRepository:
    """Repository managing user feedback in MySQL."""

    @staticmethod
    def record_feedback(
        session_id: str,
        rating: int,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        comments: Optional[str] = None,
    ) -> Optional[int]:
        """Records user rating and comments into MySQL feedback table."""
        query = (
            "INSERT INTO feedback (session_id, rating, question, answer, comments) "
            "VALUES (%s, %s, %s, %s, %s)"
        )
        return db_manager.execute_write(
            query,
            (session_id, rating, question or "", answer or "", comments or ""),
        )
