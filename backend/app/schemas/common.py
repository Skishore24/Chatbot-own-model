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


class ModelStatusInfo(BaseModel):
    status: str
    device: str
    checkpoint_exists: bool
    parameters: Optional[int] = None
    vocab_size: Optional[int] = None


class RagStatusInfo(BaseModel):
    status: str
    documents: int


class DatabaseStatusInfo(BaseModel):
    status: str
    database: str


class HealthResponse(BaseModel):
    status: str = "healthy"
    application: str
    version: str
    model: ModelStatusInfo
    rag: RagStatusInfo
    database: DatabaseStatusInfo


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
