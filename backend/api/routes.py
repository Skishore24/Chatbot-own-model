"""
api/routes.py
----------------------------------------------------
Genkit AI - Production API Routes

Features
--------
• Health API
• Streaming Chat
• Lead Management
• Feedback
• Session Management
• Version API
• Favicon

Author : Genkit AI
"""

import os
import sys
import time
import traceback
from typing import Optional

from fastapi import (
    APIRouter,
    HTTPException,
    Depends,
    status,
)

from fastapi.responses import (
    StreamingResponse,
    FileResponse,
    JSONResponse,
)

from pydantic import (
    BaseModel,
    Field,
    EmailStr,
)

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
)

from config import (
    FRONTEND_DIR,
    DEVICE,
    MODEL_READY,
    logger,
)

from chatbot import get_answer, chatbot_health

from database import (
    save_chat_to_db,
    save_lead_to_db,
    save_feedback_to_db,
)

from api.auth import (
    verify_api_key,
    ensure_session,
)

from ai.memory.conversation import conversation_memory

router = APIRouter()

APP_VERSION = "2.0.0"

# ============================================================
# REQUEST SCHEMAS
# ============================================================

class ChatRequest(BaseModel):

    q: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User Question"
    )

    session_id: Optional[str] = Field(
        default=None
    )


class LeadRequest(BaseModel):

    name: str = Field(
        ...,
        min_length=2,
        max_length=100
    )

    email: EmailStr

    phone: Optional[str] = None

    company: Optional[str] = None

    message: Optional[str] = None


class FeedbackRequest(BaseModel):

    session_id: Optional[str] = None

    question: str

    answer: str

    rating: int = Field(
        ...,
        ge=1,
        le=5
    )

    comments: Optional[str] = None

# ============================================================
# HEALTH API
# ============================================================

@router.get(
    "/health",
    tags=["Health"]
)
def health():

    return {

        "status": "online",

        "service": "Genkit AI",

        "version": APP_VERSION,

        "device": DEVICE,

        "time": int(time.time())

    }


# ============================================================
# VERSION API
# ============================================================

@router.get(
    "/version",
    tags=["Health"]
)
def version():

    return {

        "application": "Genkit AI",

        "version": APP_VERSION,

        "python": sys.version,

        "device": DEVICE

    }

# ============================================================
# CHAT API
# ============================================================

@router.post(
    "/chat",
    tags=["Chat"],
    dependencies=[Depends(verify_api_key)]
)
async def chat(req: ChatRequest):

    session_id = ensure_session(req.session_id)

    try:

        conversation_memory.add_message(
            session_id,
            "user",
            req.q
        )

    except Exception as e:

        logger.exception(e)

    def stream():

        full_answer = ""

        try:

            for token in get_answer(
                req.q,
                session_id
            ):

                if token is None:
                    continue

                token = str(token)

                full_answer += token

                yield token

            if full_answer.strip():

                try:

                    conversation_memory.add_message(
                        session_id,
                        "assistant",
                        full_answer
                    )

                except Exception:

                    logger.exception(
                        "Conversation save failed."
                    )

                try:

                    save_chat_to_db(

                        session_id=session_id,

                        question=req.q,

                        answer=full_answer

                    )

                except Exception:

                    logger.exception(
                        "Database save failed."
                    )

        except Exception as e:

            logger.error(
                traceback.format_exc()
            )

            yield (
                "\n\n⚠️ Sorry, an internal "
                "server error occurred."
            )

    return StreamingResponse(

        stream(),

        media_type="text/plain",

        headers={

            "X-Session-ID": session_id,

            "Cache-Control": "no-cache, no-store, must-revalidate",

            "Connection": "keep-alive",

            "X-Accel-Buffering": "no",

        }

    )

# ============================================================
# LEAD API
# ============================================================

@router.post(
    "/lead",
    tags=["Lead"],
    dependencies=[Depends(verify_api_key)]
)
def create_lead(req: LeadRequest):

    try:

        save_lead_to_db(

            name=req.name,

            email=req.email,

            phone=req.phone,

            company=req.company,

            message=req.message

        )

        logger.info(
            f"[Lead] New Lead : {req.email}"
        )

        return {

            "success": True,

            "message": "Lead saved successfully.",

            "data": {

                "name": req.name,

                "email": req.email

            }

        }

    except ValueError as e:

        logger.error(e)

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=str(e)

        )

    except Exception as e:

        logger.exception(e)

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail="Unable to save lead."

        )
# ============================================================
# FEEDBACK API
# ============================================================

@router.post(
    "/feedback",
    tags=["Feedback"],
    dependencies=[Depends(verify_api_key)]
)
def submit_feedback(req: FeedbackRequest):

    try:

        session_id = ensure_session(req.session_id)

        save_feedback_to_db(

            session_id=session_id,

            question=req.question,

            answer=req.answer,

            rating=req.rating,

            comments=req.comments

        )

        logger.info(

            f"[Feedback] Rating={req.rating} Session={session_id}"

        )

        return {

            "success": True,

            "message": "Feedback submitted successfully.",

            "data": {

                "session_id": session_id,

                "rating": req.rating

            }

        }

    except ValueError as e:

        logger.error(e)

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=str(e)

        )

    except Exception as e:

        logger.exception(e)

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail="Unable to save feedback."

        )
# ============================================================
# SESSION INFO
# ============================================================

@router.get(
    "/session/{session_id}",
    tags=["Session"],
    dependencies=[Depends(verify_api_key)]
)
def get_session(session_id: str):

    session_id = ensure_session(session_id)

    return {

        "success": True,

        "session_id": session_id

    }


# ============================================================
# CHAT HISTORY
# ============================================================

@router.get(
    "/history/{session_id}",
    tags=["Chat"],
    dependencies=[Depends(verify_api_key)]
)
def chat_history(session_id: str):

    try:

        session_id = ensure_session(session_id)

        history = conversation_memory.get_history(session_id)

        return {

            "success": True,

            "session_id": session_id,

            "messages": history,

            "count": len(history)

        }

    except Exception as e:

        logger.exception(e)

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail="Unable to fetch conversation history."

        )


# ============================================================
# CLEAR SESSION
# ============================================================

@router.delete(
    "/session/{session_id}",
    tags=["Session"],
    dependencies=[Depends(verify_api_key)]
)
def clear_session(session_id: str):

    try:

        conversation_memory.clear_session(session_id)

        return {

            "success": True,

            "message": "Conversation cleared.",

            "session_id": session_id

        }

    except Exception as e:

        logger.exception(e)

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail="Unable to clear conversation."

        )


# ============================================================
# MODEL INFO
# ============================================================

@router.get(
    "/model/info",
    tags=["Model"],
    dependencies=[Depends(verify_api_key)]
)
def model_info():
    """
    Return model status, parameters, and device info.
    """
    import os
    from config import (
        MODEL_DIR, MODEL_FILE, VOCAB_FILE, CONFIG_FILE,
        DEVICE, USE_GPU, GPU_NAME, GPU_MEMORY, MODEL_READY,
        VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS,
        BLOCK_SIZE, MAX_OUTPUT_LENGTH,
    )
    model_size_mb = 0
    if MODEL_FILE.exists():
        model_size_mb = round(MODEL_FILE.stat().st_size / 1024**2, 2)

    return {
        "model_ready": MODEL_READY,
        "model_file": str(MODEL_FILE),
        "model_size_mb": model_size_mb,
        "device": DEVICE,
        "gpu": USE_GPU,
        "gpu_name": GPU_NAME if USE_GPU else None,
        "gpu_memory_gb": GPU_MEMORY if USE_GPU else None,
        "architecture": {
            "vocab_size": VOCAB_SIZE,
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "block_size": BLOCK_SIZE,
            "max_output_length": MAX_OUTPUT_LENGTH,
        },
        "pipeline": chatbot_health(),
    }


# ============================================================
# MODEL RELOAD
# ============================================================

@router.post(
    "/model/reload",
    tags=["Model"],
    dependencies=[Depends(verify_api_key)]
)
def model_reload():
    """
    Reload the model weights from disk without restarting server.
    Useful after retraining.
    """
    try:
        from ai.llm.inference import reload_model
        reload_model()
        logger.info("[Admin] Model reloaded.")
        return {"success": True, "message": "Model reloaded successfully."}
    except AttributeError:
        return {
            "success": False,
            "message": "reload_model() not implemented in inference.py. Restart the server instead."
        }
    except Exception as e:
        logger.exception("Model reload failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================
# PIPELINE TEST
# ============================================================

class PipelineTestRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)


@router.post(
    "/pipeline/test",
    tags=["Debug"],
    dependencies=[Depends(verify_api_key)]
)
def pipeline_test(req: PipelineTestRequest):
    """
    Test individual pipeline steps without full LLM generation.
    Useful for debugging domain guard, intent, and entities.
    """
    from ai.nlp.domain_guard import domain_guard
    from ai.nlp.intent_classifier import intent_classifier
    from ai.nlp.entity_extractor import entity_extractor
    from utils.helper import resolve_coreference, is_valid_query
    from ai.preprocessing.cleaner import cleaner
    from ai.preprocessing.spell import spell_checker

    text = req.text
    cleaned = cleaner.clean(text)
    coref = resolve_coreference(cleaned)
    corrected = spell_checker.correct(coref)

    return {
        "original": text,
        "cleaned": cleaned,
        "coreference_resolved": coref,
        "spell_corrected": corrected,
        "valid": is_valid_query(corrected),
        "domain": domain_guard.classify(corrected),
        "intent": intent_classifier.classify(corrected),
        "entities": entity_extractor.extract(corrected),
    }
