"""
backend/app/api/streaming.py
----------------------------------------------------
Real Server-Sent Events (SSE) token-by-token streaming endpoint for Genkit AI V6.1.
- Incremental token streaming from trained custom LLM (when MODEL_READY)
- Deterministic RAG fallback streaming if model is not loaded (response_mode="rag_direct")
- Out-of-domain refusal streaming (response_mode="system")
- Structured logging & MySQL message persistence
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


async def sse_token_generator(query: str, session_id: str, request_id: str) -> AsyncGenerator[str, None]:
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

    # Determine response mode
    if not is_grounded:
        response_mode = "system"
    elif model_status == ModelStatus.READY and engine.model is not None:
        response_mode = "llm_rag"
    else:
        response_mode = "rag_direct"

    # Yield START event
    start_payload = {
        "event": "start",
        "session_id": session_id,
        "intent": intent,
        "grounded": is_grounded,
        "response_mode": response_mode,
        "sources": sources_data,
    }
    yield f"data: {json.dumps(start_payload)}\n\n"
    await asyncio.sleep(0)

    full_answer_chunks = []

    # 2. Execution Paths
    if not is_grounded:
        refusal = rag_pipeline.get_refusal_answer()
        for word in refusal.split(" "):
            chunk = word + " "
            full_answer_chunks.append(chunk)
            yield f"data: {json.dumps({'event': 'token', 'chunk': chunk})}\n\n"
            await asyncio.sleep(0.01)

    elif response_mode == "llm_rag":
        prompt = rag_pipeline.build_prompt(query, chunks)
        try:
            tokens_generated = 0
            for token_chunk in engine.generate_stream(prompt, max_new_tokens=settings.MAX_NEW_TOKENS):
                full_answer_chunks.append(token_chunk)
                tokens_generated += 1
                yield f"data: {json.dumps({'event': 'token', 'chunk': token_chunk})}\n\n"
                await asyncio.sleep(0.005)

            if tokens_generated == 0:
                # LLM yielded no tokens, stream rag direct
                direct_ans = rag_pipeline.synthesize_answer(query, chunks)
                for word in direct_ans.split(" "):
                    chunk = word + " "
                    full_answer_chunks.append(chunk)
                    yield f"data: {json.dumps({'event': 'token', 'chunk': chunk})}\n\n"
                    await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"Error in LLM stream: {e}. Falling back to rag direct.")
            direct_ans = rag_pipeline.synthesize_answer(query, chunks)
            for word in direct_ans.split(" "):
                chunk = word + " "
                full_answer_chunks.append(chunk)
                yield f"data: {json.dumps({'event': 'token', 'chunk': chunk})}\n\n"
                await asyncio.sleep(0.01)

    else:
        # response_mode == "rag_direct"
        direct_ans = rag_pipeline.synthesize_answer(query, chunks)
        words = direct_ans.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            full_answer_chunks.append(chunk)
            yield f"data: {json.dumps({'event': 'token', 'chunk': chunk})}\n\n"
            await asyncio.sleep(0.01)

    full_answer = "".join(full_answer_chunks).strip()
    if not full_answer and chunks:
        full_answer = chunks[0].text.strip()
        yield f"data: {json.dumps({'event': 'token', 'chunk': full_answer})}\n\n"

    # Persist message & latency
    latency_ms = (time.time() - start_time) * 1000
    ChatRepository.save_message(
        session_id=session_id,
        role_or_sender="assistant",
        content=full_answer,
        intent=intent,
        confidence_score=confidence,
        latency_ms=round(latency_ms, 2),
    )

    logger.info(
        f"Stream Complete [Req: {request_id}] | Mode: {response_mode} | Model: {model_status} | "
        f"Tokens: {len(full_answer_chunks)} | Latency: {latency_ms:.1f}ms"
    )

    # Yield END event
    end_payload = {
        "event": "end",
        "answer": full_answer,
        "confidence": confidence,
        "response_mode": response_mode,
        "latency_ms": round(latency_ms, 2),
    }
    yield f"data: {json.dumps(end_payload)}\n\n"


@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, req: Request):
    """Real SSE token-by-token streaming endpoint."""
    request_id = str(uuid.uuid4())[:8]
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
        sse_token_generator(cleaned_query, session_id, request_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
