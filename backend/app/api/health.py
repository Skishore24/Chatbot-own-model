"""
backend/app/api/health.py
----------------------------------------------------
Health, Model & RAG diagnostic endpoints for Genkit AI V6.
Accurately reports system health ('healthy' or 'degraded') based on:
- Model readiness state (MODEL_READY, MODEL_INVALID, MODEL_INCOMPATIBLE, MODEL_NOT_FOUND)
- Database connectivity state
- RAG document index status
"""

from fastapi import APIRouter
from typing import List, Dict, Any
import torch

from app.core.config import settings
from app.schemas.common import HealthResponse, ModelInfoResponse, ModelStatusInfo, RagStatusInfo, DatabaseStatusInfo
from app.database.connection import db_manager
from app.database.repository import ChatRepository
from app.rag.pipeline import get_rag_pipeline
from app.llm.inference import ModelStatus

router = APIRouter(tags=["Health & System"])


@router.get("/health", response_model=HealthResponse)
async def get_health():
    """Returns operational health diagnostics of the Genkit AI system."""
    from app.api.chat import get_generation_engine_and_status

    rag_pipe = get_rag_pipeline()
    cuda_avail = torch.cuda.is_available()
    device_name = "cuda" if cuda_avail else "cpu"

    checkpoint_exists = settings.model_checkpoint_exists()
    _, status = get_generation_engine_and_status()

    # Determine reason if not ready
    reason = None
    if status == ModelStatus.INVALID:
        reason = "Checkpoint could not be loaded or is corrupted"
    elif status == ModelStatus.NOT_FOUND or status == ModelStatus.NOT_TRAINED:
        reason = "Checkpoint file not found on disk"
    elif status == ModelStatus.INCOMPATIBLE:
        reason = "Checkpoint architecture does not match configuration"
    elif status == ModelStatus.ERROR:
        reason = "Unexpected error while initializing model"

    db_status_str = "ready" if db_manager.is_available else "unavailable"
    rag_status_str = "ready" if rag_pipe.total_documents > 0 else "empty"

    db_name = f"{settings.MYSQL_DATABASE} ({db_manager.engine_type})"

    # Overall system status: degraded if model is not READY or db is unavailable
    overall_status = "healthy" if (status == ModelStatus.READY and db_manager.is_available) else "degraded"

    return HealthResponse(
        status=overall_status,
        application=settings.APP_NAME,
        version=settings.APP_VERSION,
        model=ModelStatusInfo(
            status=status,
            device=device_name,
            checkpoint_exists=checkpoint_exists,
            checkpoint=settings.MODEL_CHECKPOINT_PATH.name if checkpoint_exists else None,
            reason=reason,
            vocab_size=settings.VOCAB_SIZE,
        ),
        rag=RagStatusInfo(
            status=rag_status_str,
            documents=rag_pipe.total_documents,
        ),
        database=DatabaseStatusInfo(
            status=db_status_str,
            database=db_name,
        ),
    )


@router.get("/model", response_model=ModelInfoResponse)
async def get_model_info():
    """Returns metadata and architecture configuration of the custom LLM."""
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    return ModelInfoResponse(
        model_name="Genkit Enterprise GPT v6.1",
        parameters=80_000_000,
        vocab_size=settings.VOCAB_SIZE,
        embed_dim=settings.EMBED_DIM,
        num_layers=settings.NUM_LAYERS,
        num_heads=settings.NUM_HEADS,
        num_kv_heads=settings.NUM_KV_HEADS,
        block_size=settings.BLOCK_SIZE,
        checkpoint_exists=settings.model_checkpoint_exists(),
        device=device_str,
    )


@router.get("/analytics")
async def get_analytics_endpoint():
    """Returns live analytics metrics for the dashboard."""
    return ChatRepository.get_analytics()


@router.get("/rag/knowledge")
async def get_knowledge_chunks():
    """Returns list of indexed RAG knowledge chunks for dashboard inspection."""
    rag_pipe = get_rag_pipeline()
    return [
        {
            "id": c.id,
            "source": c.source,
            "category": c.category,
            "title": c.title,
            "text": c.text,
            "keywords": c.keywords,
            "priority": c.priority,
        }
        for c in rag_pipe.chunks
    ]
