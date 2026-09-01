"""
backend/app/api/chat.py
----------------------------------------------------
Standard synchronous chat endpoint for Genkit AI V6.1.
- Input security & rate limiting
- Hybrid RAG retrieval & Grounding
- Clean execution path separation:
    1. Out-of-Domain Refusal (response_mode="system")
    2. Trained Custom LLM Generation (response_mode="llm_rag")
    3. Deterministic RAG Fallback (response_mode="rag_direct")
- Structured logging & MySQL persistence
"""

import time
import uuid
from typing import Optional, Tuple
from fastapi import APIRouter, HTTPException, Request, status

from app.core.config import settings
from app.core.logger import logger
from app.core.security import security_service
from app.schemas.chat import ChatRequest, ChatResponse, ChatSource
from app.database.repository import ChatRepository
from app.rag.pipeline import get_rag_pipeline
from app.llm.inference import load_model_and_tokenizer, ModelStatus
from app.llm.generation import GenerationEngine

router = APIRouter(tags=["Chat"])

# Lazy singleton model runtime & status
_engine: Optional[GenerationEngine] = None
_model_status: str = ModelStatus.NOT_FOUND


def get_generation_engine_and_status() -> Tuple[GenerationEngine, str]:
    global _engine, _model_status
    if _engine is None:
        model, tokenizer, _, _model_status = load_model_and_tokenizer()
        _engine = GenerationEngine(model, tokenizer)
    return _engine, _model_status


def reset_generation_engine():
    """Forces reloading model runtime (used after re-training)."""
    global _engine, _model_status
    _engine = None
    _model_status = ModelStatus.NOT_FOUND


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, req: Request):
    """Processes user query through Input Guard -> Hybrid RAG -> Custom LLM / RAG Direct -> Output Persistence."""
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    client_ip = req.client.host if req.client else "127.0.0.1"

    # 1. Rate Limiting Check
    if not security_service.check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip} (Req: {request_id})")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait a moment before sending another query.",
        )

    # 2. Input Sanitization & Prompt Injection Guard
    cleaned_query, is_safe = security_service.sanitize_input(request.message)
    if not is_safe or security_service.scan_prompt_injection(cleaned_query):
        logger.warning(f"Security guard flagged prompt: '{request.message[:50]}' (Req: {request_id})")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Your query was flagged by our security guard. Please ask a valid question.",
        )

    session_id = request.session_id or str(uuid.uuid4())[:8]
    ChatRepository.save_message(session_id, "user", cleaned_query)

    # 3. Hybrid RAG Retrieval & Grounding
    rag_pipeline = get_rag_pipeline()
    chunks, confidence, is_grounded = rag_pipeline.retrieve(cleaned_query)
    engine, model_status = get_generation_engine_and_status()

    sources = [
        ChatSource(id=c.id, title=c.title, category=c.category, score=c.score)
        for c in chunks
    ]

    # 4. Determine Execution Path
    if not is_grounded:
        # Path A: Out-of-Domain Refusal
        answer = rag_pipeline.get_refusal_answer()
        intent = "OutOfDomain"
        response_mode = "system"
        sources = []
    elif model_status == ModelStatus.READY and engine.model is not None:
        # Path B: Custom LLM Generation conditioned on RAG context
        intent = chunks[0].category if chunks else "General"
        prompt = rag_pipeline.build_prompt(cleaned_query, chunks)
        try:
            llm_response = engine.generate(prompt, max_new_tokens=settings.MAX_NEW_TOKENS)
            if llm_response and len(llm_response.strip()) > 10:
                answer = llm_response.strip()
                response_mode = "llm_rag"
            else:
                answer = rag_pipeline.synthesize_answer(cleaned_query, chunks)
                response_mode = "rag_direct"
        except Exception as e:
            logger.error(f"LLM generation failed: {e}. Falling back to rag_direct.")
            answer = rag_pipeline.synthesize_answer(cleaned_query, chunks)
            response_mode = "rag_direct"
    else:
        # Path C: Explicit Deterministic RAG Direct Fallback
        intent = chunks[0].category if chunks else "General"
        answer = rag_pipeline.synthesize_answer(cleaned_query, chunks)
        response_mode = "rag_direct"

    # 5. Persist Response & Log
    latency_ms = (time.time() - start_time) * 1000
    ChatRepository.save_message(
        session_id=session_id,
        role_or_sender="assistant",
        content=answer,
        intent=intent,
        confidence_score=confidence,
        latency_ms=round(latency_ms, 2),
    )

    logger.info(
        f"Chat Execution [Req: {request_id}] | Mode: {response_mode} | Model: {model_status} | "
        f"Intent: {intent} | Conf: {confidence:.2f} | Chunks: {len(chunks)} | Latency: {latency_ms:.1f}ms"
    )

    return ChatResponse(
        success=True,
        session_id=session_id,
        query=cleaned_query,
        answer=answer,
        intent=intent,
        confidence=confidence,
        grounded=is_grounded,
        sources=sources,
        latency_ms=round(latency_ms, 2),
    )
