"""
backend/app/api/feedback.py
----------------------------------------------------
User feedback and rating endpoint for Genkit AI V6.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.database.repository import FeedbackRepository
from app.core.logger import logger

router = APIRouter(tags=["Feedback"])


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Chat session ID")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    question: Optional[str] = Field(default=None, description="User question")
    answer: Optional[str] = Field(default=None, description="Assistant answer")
    comments: Optional[str] = Field(default=None, max_length=1000, description="Optional comments")


class FeedbackResponse(BaseModel):
    success: bool = True
    message: str = "Feedback received. Thank you!"
    feedback_id: Optional[int] = None


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback_endpoint(req: FeedbackRequest):
    """Saves user rating and feedback for assistant response quality."""
    try:
        feedback_id = FeedbackRepository.record_feedback(
            session_id=req.session_id,
            rating=req.rating,
            question=req.question,
            answer=req.answer,
            comments=req.comments,
        )
        logger.info(f"Recorded feedback ID {feedback_id} for session {req.session_id} (Rating: {req.rating})")
        return FeedbackResponse(
            success=True,
            message="Thank you for your feedback! It helps improve Genkit AI.",
            feedback_id=feedback_id,
        )
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback.",
        )
