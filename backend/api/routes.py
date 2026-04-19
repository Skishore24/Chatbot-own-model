import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional

from core.brain import get_answer
from services.memory import (
    add_message,
    save_chat_to_db,
    save_lead_to_db,
    update_user_info
)
from db.database import get_connection
from app.config import logger

router = APIRouter()

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    q: str = Field(..., min_length=1)
    session_id: Optional[str] = None

class LeadRequest(BaseModel):
    name: str
    email: str
    session_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int = Field(..., ge=1, le=5)

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@router.get("/health")
def health():
    return {"status": "ok", "message": "Genkit AI Service is healthy"}

@router.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    
    try:
        add_message(session_id, "user", req.q)
        update_user_info(session_id, req.q)
    except Exception as e:
        logger.error(f"[SESSION ERROR] {session_id}: {e}")

    def generator():
        full_reply = ""
        try:
            for chunk in get_answer(req.q, session_id):
                full_reply += chunk
                yield chunk
            
            if full_reply.strip():
                add_message(session_id, "assistant", full_reply)
                save_chat_to_db(session_id, req.q, full_reply)
        except Exception as e:
            logger.error(f"[PIPELINE ERROR] {session_id}: {e}")
            yield "\n⚠️ Internal error. Please try again."

    return StreamingResponse(
        generator(),
        media_type="text/plain",
        headers={"X-Session-Id": session_id}
    )

@router.post("/lead")
def lead(req: LeadRequest):
    try:
        save_lead_to_db(req.name, req.email)
        logger.info(f"📩 Lead captured: {req.email}")
        return {"status": "success", "message": "Lead saved"}
    except Exception as e:
        logger.error(f"[LEAD ERROR]: {e}")
        raise HTTPException(status_code=500, detail="Failed to save lead")

@router.post("/feedback")
def feedback(req: FeedbackRequest):
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO feedback (question, answer, rating) VALUES (?, ?, ?)",
                (req.question, req.answer, req.rating)
            )
            conn.commit()
        logger.info(f"⭐ Feedback saved: {req.rating}/5")
        return {"status": "success", "message": "Feedback saved"}
    except Exception as e:
        logger.error(f"[FEEDBACK ERROR]: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback")