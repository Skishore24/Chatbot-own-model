"""
backend/app/api/health.py
----------------------------------------------------
Health & Model diagnostic endpoints for Genkit AI V6.
"""

from fastapi import APIRouter
import torch

from app.core.config import settings
from app.schemas.common import HealthResponse, ModelInfoResponse, ModelStatusInfo, RagStatusInfo, DatabaseStatusInfo
from app.database.connection import db_manager
from app.rag.pipeline import get_rag_pipeline
from app.llm.inference import load_model_and_tokenizer, ModelStatus

router = APIRouter(tags=["Health & System"])


@router.get("/health", response_model=HealthResponse)
async def get_health():
    """Returns operational health diagnostics of the Genkit AI system."""
    rag_pipe = get_rag_pipeline()
    cuda_avail = torch.cuda.is_available()
    device_name = "cuda" if cuda_avail else "cpu"

    # Check model status
    checkpoint_exists = settings.model_checkpoint_exists()
    if not checkpoint_exists:
        model_status_str = "not_trained"
    else:
        # Check loadability
        _, _, _, status = load_model_and_tokenizer()
        if status == ModelStatus.READY:
            model_status_str = "ready"
        elif status == ModelStatus.INCOMPATIBLE:
            model_status_str = "incompatible"
        else:
            model_status_str = "not_trained"

    db_status_str = "ready" if db_manager.is_available else "unavailable"
    rag_status_str = "ready" if rag_pipe.total_documents > 0 else "empty"

    return HealthResponse(
        status="healthy",
        application=settings.APP_NAME,
        version=settings.APP_VERSION,
        model=ModelStatusInfo(
            status=model_status_str,
            device=device_name,
            checkpoint_exists=checkpoint_exists,
            vocab_size=settings.VOCAB_SIZE,
        ),
        rag=RagStatusInfo(
            status=rag_status_str,
            documents=rag_pipe.total_documents,
        ),
        database=DatabaseStatusInfo(
            status=db_status_str,
            database=settings.MYSQL_DATABASE,
        ),
    )


@router.get("/model", response_model=ModelInfoResponse)
async def get_model_info():
    """Returns metadata and architecture configuration of the custom LLM."""
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    return ModelInfoResponse(
        model_name="Genkit Enterprise GPT v6.0",
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
