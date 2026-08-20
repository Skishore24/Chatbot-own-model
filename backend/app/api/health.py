"""
backend/app/api/health.py
----------------------------------------------------
Health & Model diagnostic endpoints for Genkit AI V6.
"""

from fastapi import APIRouter
import torch

from app.core.config import settings
from app.schemas.common import HealthResponse, ModelInfoResponse
from app.database.connection import db_manager
from app.rag.pipeline import get_rag_pipeline

router = APIRouter(tags=["Health & System"])


@router.get("/health", response_model=HealthResponse)
async def get_health():
    """Returns operational health diagnostics of the Genkit AI system."""
    rag_pipe = get_rag_pipeline()
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"

    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        device=device_name,
        cuda_available=cuda_avail,
        model_loaded=settings.model_checkpoint_exists(),
        tokenizer_loaded=settings.tokenizer_checkpoint_exists(),
        database_status=f"connected (MySQL: {settings.MYSQL_DATABASE})",
        rag_documents_indexed=rag_pipe.total_documents,
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
