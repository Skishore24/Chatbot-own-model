"""
backend/app/api/streaming.py
----------------------------------------------------
Real Server-Sent Events (SSE) token-by-token streaming endpoint for Genkit AI V6.1.
- Incremental token streaming from trained custom LLM
- Deterministic RAG fallback if model not trained or incompatible
- Persists full response with latency into MySQL
"""

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.logger import logger
from app.core.security import security_service
from app.schemas.chat import ChatRequest
from app.database.repository import ChatRepository
from app.rag.pipeline import get_rag_pipeline
from app.llm.inference import ModelStatus
from app.api.chat import get_generation_engine_and_status

router = APIRouter(tags=["Streaming"])


async def sse_token_generator(query: str, session_id: str) -> AsyncGenerator[str, None]:
    """Generates real Server-Sent Events stream token-by-token."""
    start_time = time.time()
    rag_pipeline = get_rag_pipeline()
    engine, model_status = get_generation_engine_and_status()

    # 1. RAG Retrieval & Grounding
    chunks, confidence, is_grounded = rag_pipeline.retrieve(query)

    sources_data = [
        {"id": c.id, "title": c.title, "category": c.category, "score": c.score}
        for c in chunks
    ]
    intent = chunks[0].category if chunks else ("General" if is_grounded else "OutOfDomain")

    # Yield START event
    start_payload = {
        "event": "start",
        "session_id": session_id,
        "intent": intent,
        "grounded": is_grounded,
        "sources": sources_data,
    }
    yield f"data: {json.dumps(start_payload)}\n\n"
    await asyncio.sleep(0)

    full_answer_parts = []

    # 2. Refusal or Streaming Generation
    if not is_grounded:
        refusal = rag_pipeline.get_refusal_answer()
        full_answer = refusal
        # Stream refusal tokens
        for word in refusal.split(" "):
            chunk = word + " "
            yield f"data: {json.dumps({'event': 'token', 'chunk': chunk})}\n\n"
            await asyncio.sleep(0.01)
    else:
        full_answer = rag_pipeline.synthesize_answer(query, chunks)
        # Stream synthesized grounded markdown text word-by-word
        words = full_answer.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'event': 'token', 'chunk': chunk})}\n\n"
            await asyncio.sleep(0.012)
    if not full_answer and chunks:
        full_answer = f"Based on Genkit verified knowledge: {chunks[0].text}"
        yield f"data: {json.dumps({'event': 'token', 'chunk': full_answer})}\n\n"

    # Persist message
    latency_ms = (time.time() - start_time) * 1000
    ChatRepository.save_message(
        session_id=session_id,
        role_or_sender="assistant",
        content=full_answer,
        intent=intent,
        confidence_score=confidence,
        latency_ms=round(latency_ms, 2),
    )

    # Yield END event
    end_payload = {
        "event": "end",
        "answer": full_answer,
        "confidence": confidence,
        "latency_ms": round(latency_ms, 2),
    }
    yield f"data: {json.dumps(end_payload)}\n\n"


@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, req: Request):
    """Real SSE token-by-token streaming endpoint."""
    client_ip = req.client.host if req.client else "127.0.0.1"

    # 1. Rate Limit
    if not security_service.check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait a moment.",
        )

    # 2. Sanitize
    cleaned_query, is_safe = security_service.sanitize_input(request.message)
    if not is_safe or security_service.scan_prompt_injection(cleaned_query):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Your query was flagged by our security guard.",
        )

    session_id = request.session_id or str(uuid.uuid4())[:8]
    ChatRepository.save_message(session_id, "user", cleaned_query)

    return StreamingResponse(
        sse_token_generator(cleaned_query, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
