from pydantic import BaseModel, Field, EmailStr
from typing import Optional

class ChatRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=500)
    session_id: Optional[str] = None


class LeadRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    session_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    session_id: Optional[str] = None
    question: str
    answer: str
    rating: int = Field(..., ge=1, le=5)
