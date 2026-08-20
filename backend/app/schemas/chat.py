"""
backend/app/schemas/chat.py
----------------------------------------------------
Pydantic schemas for chat requests and responses.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatSource(BaseModel):
    id: str
    title: str
    category: str
    score: float


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User input query")
    session_id: Optional[str] = Field(default=None, description="Optional session identifier for history tracking")
    stream: bool = Field(default=False, description="Whether to stream response tokens")


class ChatResponse(BaseModel):
    success: bool = True
    session_id: str
    query: str
    answer: str
    intent: str
    confidence: float
    grounded: bool
    sources: List[ChatSource] = []
    latency_ms: float


class HistoryMessage(BaseModel):
    id: int
    sender: str
    text: str
    created_at: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: List[HistoryMessage]
