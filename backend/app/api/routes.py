"""
backend/app/api/routes.py
----------------------------------------------------
GENKIT AI v5.0 Enterprise FastAPI REST & SSE Streaming Routes
- Loads model/tokenizer via ModelLoader (checkpoint auto-detect)
- Per-IP rate limiting
- Intent detection from helper utilities
- Proper SSE streaming
"""

import time
import uuid
import sys
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field

# Ensure backend on path
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.core.logger import logger, new_trace_id
from app.core.security import security_service
from app.database.connection import db_pool
from app.ai.llm.model_loader import get_model_and_tokenizer
from app.ai.llm.inference import GenerationEngine
from app.ai.llm.prompt_builder import prompt_builder
from app.ai.rag.retriever import HybridRetriever
from app.api.streaming import create_sse_response

router = APIRouter(prefix="/api/v5")

# ---------------------------------------------------------------------------
# Lazy singletons — initialized once on first request
# ---------------------------------------------------------------------------
_gen_engine: Optional[GenerationEngine] = None
_retriever: Optional[HybridRetriever] = None


def _get_gen_engine() -> GenerationEngine:
    global _gen_engine
    if _gen_engine is None:
        logger.info("Initializing GenerationEngine (first request)...")
        model, tokenizer, _ = get_model_and_tokenizer()
        _gen_engine = GenerationEngine(model, tokenizer)
        logger.info("GenerationEngine ready.")
    return _gen_engine


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        logger.info("Initializing HybridRetriever (first request)...")
        _retriever = HybridRetriever()
        logger.info("HybridRetriever ready.")
    return _retriever


# ---------------------------------------------------------------------------
# Simple intent classifier using keyword rules
# ---------------------------------------------------------------------------
_INTENT_MAP = [
    (["price", "cost", "pricing", "budget", "plan", "rate", "fee", "how much"], "PricingInquiry"),
    (["service", "offer", "provide", "what do you do", "capabilities", "solutions"], "ServiceInquiry"),
    (["team", "who", "staff", "people", "about you", "founder"], "TeamInquiry"),
    (["contact", "email", "phone", "reach", "call", "address"], "ContactInquiry"),
    (["project", "portfolio", "case study", "work", "built", "client"], "PortfolioInquiry"),
    (["ai", "machine learning", "model", "llm", "gpt", "neural", "chatbot"], "AIInquiry"),
    (["web", "website", "react", "nextjs", "frontend", "backend"], "WebDevInquiry"),
    (["mobile", "flutter", "android", "ios", "app"], "MobileDevInquiry"),
    (["hello", "hi", "hey", "greet", "good morning", "good evening"], "Greeting"),
]


def _detect_intent(text: str) -> str:
    text_lower = text.lower()
    for keywords, intent in _INTENT_MAP:
        if any(kw in text_lower for kw in keywords):
            return intent
    return "GeneralInquiry"


def _compute_confidence(intent: str, query: str) -> float:
    if intent == "GeneralInquiry":
        return 0.65
    query_lower = query.lower()
    for keywords, mapped_intent in _INTENT_MAP:
        if mapped_intent == intent:
            hit_count = sum(1 for kw in keywords if kw in query_lower)
            return min(0.70 + hit_count * 0.05, 0.98)
    return 0.80


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))


class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: str
    confidence: float
    latency_ms: float


class LeadRequest(BaseModel):
    name: str = Field(..., min_length=2)
    email: EmailStr
    phone: Optional[str] = None
    service: Optional[str] = None
    budget: Optional[str] = None
    notes: Optional[str] = None
    session_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    session_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper: client IP extraction
# ---------------------------------------------------------------------------

def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/chat/query", response_model=ChatResponse)
async def chat_query(req: ChatRequest, request: Request):
    """Synchronous full JSON response chat endpoint."""
    client_ip = _get_client_ip(request)

    # Rate limiting
    if not security_service.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")

    start_time = time.time()

    # Input sanitization
    cleaned_input, is_safe = security_service.sanitize_input(req.message)
    if not is_safe:
        raise HTTPException(status_code=400, detail="Input rejected: Security violation detected.")

    if security_service.scan_prompt_injection(cleaned_input):
        raise HTTPException(status_code=400, detail="Input rejected: Prompt injection detected.")

    intent = _detect_intent(cleaned_input)
    confidence = _compute_confidence(intent, cleaned_input)

    try:
        retriever = _get_retriever()
        context_blocks, _ = retriever.retrieve(cleaned_input)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        context_blocks = []

    prompt = prompt_builder.build_prompt(cleaned_input, context_passages=context_blocks)

    try:
        engine = _get_gen_engine()
        generated_text = engine.generate_text(
            prompt,
            query=cleaned_input,
            context_passages=context_blocks,
            intent=intent,
            max_new_tokens=settings.MAX_GEN_TOKENS,
            temperature=settings.TEMPERATURE,
            top_k=settings.TOP_K,
            top_p=settings.TOP_P,
            repetition_penalty=settings.REPETITION_PENALTY,
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        generated_text = "I apologize, I encountered an error generating a response. Please try again."

    final_output = security_service.sanitize_output(generated_text)
    latency_ms = round((time.time() - start_time) * 1000, 2)

    db_pool.save_chat_message(req.session_id, "user", cleaned_input, intent=intent, confidence=confidence, latency_ms=latency_ms)
    db_pool.save_chat_message(req.session_id, "assistant", final_output, intent=intent, confidence=confidence, latency_ms=latency_ms)

    logger.info(f"Chat query [{intent}] conf={confidence:.2f} latency={latency_ms}ms")

    return ChatResponse(
        response=final_output,
        session_id=req.session_id,
        intent=intent,
        confidence=confidence,
        latency_ms=latency_ms,
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """Server-Sent Events (SSE) real-time token streaming endpoint."""
    client_ip = _get_client_ip(request)

    if not security_service.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")

    cleaned_input, is_safe = security_service.sanitize_input(req.message)
    if not is_safe or security_service.scan_prompt_injection(cleaned_input):
        raise HTTPException(status_code=400, detail="Security violation detected.")

    intent = _detect_intent(cleaned_input)
    confidence = _compute_confidence(intent, cleaned_input)

    try:
        retriever = _get_retriever()
        context_blocks, _ = retriever.retrieve(cleaned_input)
    except Exception as e:
        logger.error(f"Retrieval failed during stream: {e}")
        context_blocks = []

    prompt = prompt_builder.build_prompt(cleaned_input, context_passages=context_blocks)

    try:
        engine = _get_gen_engine()
        token_stream = engine.generate_stream(
            prompt,
            query=cleaned_input,
            context_passages=context_blocks,
            intent=intent,
            max_new_tokens=settings.MAX_GEN_TOKENS,
            temperature=settings.TEMPERATURE,
            top_k=settings.TOP_K,
            top_p=settings.TOP_P,
            repetition_penalty=settings.REPETITION_PENALTY,
        )
    except Exception as e:
        logger.error(f"Stream generation failed: {e}")
        def _error_stream():
            yield "I apologize, I encountered an error. Please try again."
        token_stream = _error_stream()

    db_pool.save_chat_message(req.session_id, "user", cleaned_input, intent=intent, confidence=confidence)

    return create_sse_response(
        token_stream,
        session_id=req.session_id,
        intent=intent,
        confidence=confidence,
    )


@router.get("/history")
async def get_history(session_id: str):
    """Retrieves session conversation history."""
    messages = db_pool.get_chat_history(session_id, limit=20)
    return {"status": "success", "session_id": session_id, "messages": messages}


@router.post("/lead")
async def create_lead(req: LeadRequest):
    """Captures client contact lead information."""
    success = db_pool.save_lead(
        session_id=req.session_id or "",
        name=req.name,
        email=req.email,
        phone=req.phone,
        service=req.service,
        budget=req.budget,
        notes=req.notes,
    )
    return {"status": "success" if success else "recorded_in_memory", "lead": req.dict()}


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Records user 1-5 star ratings."""
    logger.info(f"Feedback session={req.session_id} rating={req.rating} comment={req.comment}")
    return {"status": "success", "message": "Feedback recorded successfully."}


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    import torch
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "gpu_available": torch.cuda.is_available(),
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
        "model_checkpoint_exists": settings.model_checkpoint_exists(),
        "tokenizer_checkpoint_exists": settings.tokenizer_checkpoint_exists(),
        "dataset_dir": str(settings.DATASET_DIR),
        "database_connected": db_pool.is_connected,
    }


@router.get("/model-info")
async def model_info():
    """Returns model architecture information."""
    try:
        engine = _get_gen_engine()
        param_count = engine.model.count_parameters()
    except Exception:
        param_count = 0
    return {
        "vocab_size": settings.VOCAB_SIZE,
        "embed_dim": settings.EMBED_DIM,
        "num_heads": settings.NUM_HEADS,
        "num_kv_heads": settings.NUM_KV_HEADS,
        "num_layers": settings.NUM_LAYERS,
        "block_size": settings.BLOCK_SIZE,
        "parameters": param_count,
        "model_type": "EnterpriseGPT v5.0 (GQA + RoPE + SwiGLU + RMSNorm)",
        "external_apis": "None — 100% local inference",
    }
