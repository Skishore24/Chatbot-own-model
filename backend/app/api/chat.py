"""
backend/app/api/chat.py
----------------------------------------------------
Standard synchronous chat endpoint for Genkit AI V6.1.
- Input security & rate limiting
- Hybrid RAG retrieval & Grounding
- Custom LLM inference (with ModelStatus readiness guard)
- Answer-level grounding validation
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
_model_status: str = ModelStatus.NOT_TRAINED


def get_generation_engine_and_status() -> Tuple[GenerationEngine, str]:
    global _engine, _model_status
    if _engine is None:
        model, tokenizer, _, _model_status = load_model_and_tokenizer()
        _engine = GenerationEngine(model, tokenizer)
    return _engine, _model_status


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, req: Request):
    """Processes user query through Input Guard -> Hybrid RAG -> Custom LLM -> Grounding Validator."""
    start_time = time.time()
    client_ip = req.client.host if req.client else "127.0.0.1"

    # 1. Rate Limiting Check
    if not security_service.check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait a moment before sending another query.",
        )

    # 2. Input Sanitization & Prompt Injection Guard
    cleaned_query, is_safe = security_service.sanitize_input(request.message)
    if not is_safe or security_service.scan_prompt_injection(cleaned_query):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Your query was flagged by our security guard. Please ask a valid question.",
        )

    session_id = request.session_id or str(uuid.uuid4())[:8]
    ChatRepository.save_message(session_id, "user", cleaned_query)

    # 3. Hybrid RAG Retrieval & Grounding
    rag_pipeline = get_rag_pipeline()
    chunks, confidence, is_grounded = rag_pipeline.retrieve(cleaned_query)

    # 4. Handle Out-of-Domain Refusal or Generation
    if not is_grounded:
        answer = rag_pipeline.get_refusal_answer()
        intent = "OutOfDomain"
        sources = []
    else:
        intent = chunks[0].category if chunks else "General"
        sources = [
            ChatSource(id=c.id, title=c.title, category=c.category, score=c.score)
            for c in chunks
        ]
        answer = rag_pipeline.synthesize_answer(cleaned_query, chunks)

    # 6. Persist Response & Return
    latency_ms = (time.time() - start_time) * 1000
    ChatRepository.save_message(
        session_id=session_id,
        role_or_sender="assistant",
        content=answer,
        intent=intent,
        confidence_score=confidence,
        latency_ms=round(latency_ms, 2),
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
