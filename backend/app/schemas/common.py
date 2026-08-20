"""
backend/app/schemas/common.py
----------------------------------------------------
Standard API response schemas for Genkit AI V6.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class StandardResponse(BaseModel):
    success: bool = True
    message: str = "Operation successful"
    data: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str = "healthy"
    app_name: str
    version: str
    device: str
    cuda_available: bool
    model_loaded: bool
    tokenizer_loaded: bool
    database_status: str
    rag_documents_indexed: int


class ModelInfoResponse(BaseModel):
    model_name: str
    parameters: int
    vocab_size: int
    embed_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    block_size: int
    checkpoint_exists: bool
    device: str
