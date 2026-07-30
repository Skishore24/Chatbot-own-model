"""
backend/app/api/routes.py
----------------------------------------------------
GENKIT AI v5.0 Enterprise FastAPI REST & SSE Streaming Routes
"""

import time
import uuid
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field

from app.core.config import settings
from app.core.logger import logger, new_trace_id
from app.core.security import security_service
from app.database.connection import db_pool
from app.ai.tokenizer.tokenizer import default_tokenizer
from app.ai.llm.ml_model import EnterpriseGPTModel, GPTConfig
from app.ai.llm.inference import GenerationEngine
from app.ai.llm.prompt_builder import prompt_builder
from app.ai.rag.retriever import HybridRetriever
from app.api.streaming import create_sse_response

router = APIRouter(prefix="/api/v5")

# Initialize AI Service Singletons
gpt_config = GPTConfig(vocab_size=16000, block_size=2048, n_embd=384, n_head=8, n_kv_head=2, n_layer=8)
gpt_model = EnterpriseGPTModel(gpt_config)
gen_engine = GenerationEngine(gpt_model, default_tokenizer)
retriever = HybridRetriever()


# Request / Response Schemas
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


@router.post("/chat/query", response_model=ChatResponse)
async def chat_query(req: ChatRequest):
    """Synchronous Full JSON Response Chat Endpoint."""
    start_time = time.time()
    trace_id = new_trace_id()

    # 1. Zero-Trust Security Sanitization
    cleaned_input, is_safe = security_service.sanitize_input(req.message)
    if not is_safe:
        raise HTTPException(status_code=400, detail="Security violation: Input contained illegal characters.")

    # 2. Prompt Injection Scan
    if security_service.scan_prompt_injection(cleaned_input):
        raise HTTPException(status_code=400, detail="Security violation: Prompt injection attack detected.")

    # 3. Hybrid RAG Retrieval
    context_blocks, _ = retriever.retrieve(cleaned_input)

    # 4. Prompt Construction
    prompt = prompt_builder.build_prompt(cleaned_input, context_passages=context_blocks)

    # 5. Token Generation
    generated_text = gen_engine.generate_text(
        prompt,
        max_new_tokens=settings.MAX_GEN_TOKENS,
        temperature=settings.TEMPERATURE,
        top_k=settings.TOP_K,
        top_p=settings.TOP_P,
        repetition_penalty=settings.REPETITION_PENALTY,
    )

    # 6. Output Sanitization
    final_output = security_service.sanitize_output(generated_text)
    latency_ms = round((time.time() - start_time) * 1000, 2)

    # Save to Database
    db_pool.save_chat_message(req.session_id, "user", cleaned_input, intent="DomainInquiry", confidence=0.95, latency_ms=latency_ms)
    db_pool.save_chat_message(req.session_id, "assistant", final_output, intent="DomainInquiry", confidence=0.95, latency_ms=latency_ms)

    return ChatResponse(
        response=final_output,
        session_id=req.session_id,
        intent="DomainInquiry",
        confidence=0.95,
        latency_ms=latency_ms,
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Server-Sent Events (SSE) Real-time Token Streaming Endpoint."""
    cleaned_input, is_safe = security_service.sanitize_input(req.message)
    if not is_safe or security_service.scan_prompt_injection(cleaned_input):
        raise HTTPException(status_code=400, detail="Security violation detected.")

    context_blocks, _ = retriever.retrieve(cleaned_input)
    prompt = prompt_builder.build_prompt(cleaned_input, context_passages=context_blocks)

    token_stream = gen_engine.generate_stream(
        prompt,
        max_new_tokens=settings.MAX_GEN_TOKENS,
        temperature=settings.TEMPERATURE,
        top_k=settings.TOP_K,
        top_p=settings.TOP_P,
        repetition_penalty=settings.REPETITION_PENALTY,
    )

    return create_sse_response(
        token_stream, session_id=req.session_id, intent="DomainInquiry", confidence=0.95
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
    logger.info(f"User Feedback received for session {req.session_id}: Rating={req.rating}, Comment={req.comment}")
    return {"status": "success", "message": "Feedback recorded successfully."}


@router.get("/health")
async def health_check():
    """Health check endpoint with GPU & DB status."""
    import torch
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "gpu_available": torch.cuda.is_available(),
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "database_connected": db_pool.is_connected,
    }
