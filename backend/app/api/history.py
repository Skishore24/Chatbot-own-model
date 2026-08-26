"""
backend/app/api/history.py
----------------------------------------------------
Chat history retrieval endpoint for Genkit AI V6.
"""

from fastapi import APIRouter, Query
from app.schemas.chat import HistoryResponse, HistoryMessage
from app.database.repository import ChatRepository

router = APIRouter(tags=["History"])


@router.get("/history", response_model=HistoryResponse)
async def get_history_endpoint(session_id: str = Query(..., min_length=1, description="Chat session ID")):
    """Retrieves previous chat messages for a session."""
    rows = ChatRepository.get_session_history(session_id)
    messages = [
        HistoryMessage(
            id=r["id"],
            sender=r["sender"],
            text=r["text"],
            created_at=str(r["created_at"]),
        )
        for r in rows
    ]
    return HistoryResponse(session_id=session_id, messages=messages)


@router.get("/sessions")
async def get_sessions_endpoint(limit: int = 50):
    """Retrieves active chat sessions list for dashboard viewing."""
    sessions = ChatRepository.get_all_sessions(limit=limit)
    return {"success": True, "count": len(sessions), "sessions": sessions}

