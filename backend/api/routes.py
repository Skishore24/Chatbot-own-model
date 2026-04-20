import uuid
import re
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse


from core.model import get_answer
from .models import ChatRequest, LeadRequest, FeedbackRequest
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
# ROUTES
# ─────────────────────────────────────────────

@router.get("/health")
def health():
    return {"status": "ok", "message": "Genkit AI Service is healthy"}


# ───────────────── CHAT ─────────────────
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
        buffer = ""

        try:
            for chunk in get_answer(req.q, session_id):

                # accumulate
                buffer += chunk
                full_reply += chunk

                # stream line-by-line (cleaner UI)
                if "\n" in buffer:
                    yield buffer
                    buffer = ""

            if buffer:
                yield buffer

            # save final response
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


# ───────────────── LEAD ─────────────────
@router.post("/lead")
def lead(req: LeadRequest):
    try:
        save_lead_to_db(req.name, req.email)
        logger.info(f"📩 Lead captured: {req.email}")

        return {
            "status": "success",
            "message": "Lead saved"
        }

    except Exception as e:
        logger.error(f"[LEAD ERROR]: {e}")
        raise HTTPException(status_code=500, detail="Failed to save lead")


# ───────────────── FEEDBACK ─────────────────
@router.post("/feedback")
def feedback(req: FeedbackRequest):
    try:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO feedback (session_id, question, answer, rating)
                VALUES (?, ?, ?, ?)
                """,
                (req.session_id, req.question, req.answer, req.rating)
            )
            conn.commit()

        logger.info(f"⭐ Feedback saved: {req.rating}/5")

        return {
            "status": "success",
            "message": "Feedback saved"
        }

    except Exception as e:
        logger.error(f"[FEEDBACK ERROR]: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback")