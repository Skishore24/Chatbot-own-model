"""
backend/app/api/langchain_api.py
----------------------------------------------------
FastAPI endpoints demonstrating LangChain LCEL RAG execution
and token streaming with Genkit AI's custom PyTorch LLM.
"""

import time
import json
import uuid
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import langchain
import langchain_core

from app.core.config import settings
from app.core.logger import logger
from app.core.security import security_service
from app.llm.langchain_wrapper import GenkitLLM
from app.rag.langchain_rag import GenkitRAGRetriever, create_genkit_rag_chain

router = APIRouter(prefix="/langchain", tags=["LangChain GenAI"])


class LangChainChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = Field(default=None)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=128, ge=1, le=512)


class LangChainChatResponse(BaseModel):
    success: bool
    query: str
    answer: str
    framework: str
    framework_version: str
    latency_ms: float
    sources: List[str] = []


@router.get("/info")
async def get_langchain_info():
    """Returns metadata about the LangChain GenAI integration."""
    return {
        "status": "operational",
        "framework": "LangChain",
        "langchain_version": langchain.__version__,
        "langchain_core_version": langchain_core.__version__,
        "llm_type": "genkit_custom_llm",
        "retriever_type": "genkit_hybrid_rag",
        "supported_features": [
            "LCEL Chains (Runnable)",
            "Token-by-token streaming (SSE)",
            "Hybrid RAG BaseRetriever",
            "Redis-backed session memory",
        ],
    }


@router.post("/chat", response_model=LangChainChatResponse)
async def langchain_chat(request: LangChainChatRequest, req: Request):
    """Executes user query through an end-to-end LangChain LCEL RAG pipeline."""
    start_time = time.time()
    client_ip = req.client.host if req.client else "127.0.0.1"

    # Rate limiting & security
    if not security_service.check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait a moment.",
        )

    cleaned_query, is_safe = security_service.sanitize_input(request.message)
    if not is_safe or security_service.scan_prompt_injection(cleaned_query):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query flagged by security guard.",
        )

    # Instantiate LangChain adapters
    llm = GenkitLLM(
        temperature=request.temperature or 0.7,
        max_new_tokens=request.max_tokens or 128,
    )
    retriever = GenkitRAGRetriever(top_k=3)

    # Retrieve docs for source attribution
    docs = retriever.invoke(cleaned_query)
    source_titles = [d.metadata.get("title", "Document") for d in docs]

    # Execute LCEL chain
    chain = create_genkit_rag_chain(llm=llm, retriever=retriever)
    answer = chain.invoke(cleaned_query)

    latency_ms = (time.time() - start_time) * 1000

    return LangChainChatResponse(
        success=True,
        query=cleaned_query,
        answer=answer.strip(),
        framework="LangChain LCEL",
        framework_version=langchain.__version__,
        latency_ms=round(latency_ms, 2),
        sources=source_titles,
    )


@router.post("/stream")
async def langchain_stream(request: LangChainChatRequest, req: Request):
    """Streams LCEL RAG chain output token-by-token via Server-Sent Events (SSE)."""
    client_ip = req.client.host if req.client else "127.0.0.1"

    if not security_service.check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded.",
        )

    cleaned_query, is_safe = security_service.sanitize_input(request.message)
    if not is_safe or security_service.scan_prompt_injection(cleaned_query):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query flagged by security guard.",
        )

    llm = GenkitLLM(
        temperature=request.temperature or 0.7,
        max_new_tokens=request.max_tokens or 128,
    )
    retriever = GenkitRAGRetriever(top_k=3)
    chain = create_genkit_rag_chain(llm=llm, retriever=retriever)

    def event_generator():
        yield f"data: {json.dumps({'event': 'start', 'framework': 'LangChain'})}\n\n"
        for chunk in chain.stream(cleaned_query):
            yield f"data: {json.dumps({'event': 'token', 'token': chunk})}\n\n"
        yield f"data: {json.dumps({'event': 'end'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

