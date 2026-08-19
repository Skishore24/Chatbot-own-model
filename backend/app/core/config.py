"""
backend/app/core/config.py
----------------------------------------------------
GENKIT AI v5.0 Enterprise Configuration Subsystem
Pydantic BaseSettings with strict validation & environment resolution.
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseSettings
    SettingsConfigDict = dict


class AppSettings(BaseSettings):
    # App Info
    APP_NAME: str = "Genkit AI v5.0 Enterprise"
    APP_VERSION: str = "5.0.0"
    AUTHOR: str = "Genkit AI"
    ENVIRONMENT: str = Field(default="development", description="development, staging, production")
    DEBUG: bool = Field(default=False)
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    SECRET_KEY: str = Field(default="genkit-enterprise-v5-secret-key-change-in-production-32bytes")
    API_KEY: str = Field(default="")
    ALLOWED_ORIGINS: List[str] = Field(default_factory=lambda: ["*"])

    # Path Resolution
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    ROOT_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent)

    @property
    def DATASET_DIR(self) -> Path:
        datasets_path = self.BASE_DIR / "datasets"
        if datasets_path.exists() and any(datasets_path.glob("*.json")):
            return datasets_path
        dataset_path = self.BASE_DIR / "dataset"
        return dataset_path if dataset_path.exists() else datasets_path

    @property
    def DATASET_DIRS(self) -> List[Path]:
        dirs = []
        for name in ("datasets", "dataset"):
            p = self.BASE_DIR / name
            if p.exists():
                dirs.append(p)
        return dirs or [self.BASE_DIR / "datasets"]

    @property
    def MODEL_DIR(self) -> Path:
        return self.BASE_DIR / "genkit-model"

    @property
    def MODEL_CHECKPOINT_PATH(self) -> Path:
        return self.MODEL_DIR / "model_v5.pt"

    @property
    def TOKENIZER_CHECKPOINT_PATH(self) -> Path:
        return self.MODEL_DIR / "bpe_tokenizer_v5.json"

    @property
    def UPLOAD_DIR(self) -> Path:
        return self.BASE_DIR / "uploads"

    @property
    def LOG_DIR(self) -> Path:
        return self.BASE_DIR / "logs"

    @property
    def CACHE_DIR(self) -> Path:
        return self.BASE_DIR / "cache"

    @property
    def TEMP_DIR(self) -> Path:
        return self.BASE_DIR / "temp"

    # MySQL Configuration
    MYSQL_HOST: str = Field(default="localhost")
    MYSQL_PORT: int = Field(default=3306)
    MYSQL_USER: str = Field(default="root")
    MYSQL_PASSWORD: str = Field(default="")
    MYSQL_DATABASE: str = Field(default="genkit_ai_v5")
    MYSQL_MIN_POOL_SIZE: int = Field(default=5)
    MYSQL_MAX_POOL_SIZE: int = Field(default=20)

    # Redis Configuration
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_PASSWORD: Optional[str] = Field(default=None)
    REDIS_ENABLED: bool = Field(default=False)

    # LLM v5.0 Neural Hyperparameters
    BLOCK_SIZE: int = Field(default=2048, description="Max sequence context window")
    VOCAB_SIZE: int = Field(default=16000, description="Byte-Fallback BPE Vocab Size")
    EMBED_DIM: int = Field(default=768, description="Hidden dimension d_model")
    NUM_HEADS: int = Field(default=12, description="Query Attention Heads (H_Q)")
    NUM_KV_HEADS: int = Field(default=4, description="Key-Value Attention Heads (H_KV for GQA)")
    NUM_LAYERS: int = Field(default=12, description="Transformer Decoder Layers")
    DROPOUT: float = Field(default=0.10)
    BIAS: bool = Field(default=False)
    KV_CACHE_PAGE_SIZE: int = Field(default=16, description="Paged KV-Cache Block Size")
    ROPE_FREQ_BASE: float = Field(default=10000.0)
    ROPE_SCALE_FACTOR: float = Field(default=1.0)

    # Training Parameters
    BATCH_SIZE: int = Field(default=32)
    EPOCHS: int = Field(default=60)
    LEARNING_RATE: float = Field(default=3e-4)
    MIN_LEARNING_RATE: float = Field(default=1e-5)
    WEIGHT_DECAY: float = Field(default=0.1)
    GRADIENT_CLIP: float = Field(default=1.0)
    WARMUP_STEPS: int = Field(default=500)
    GRADIENT_ACCUMULATION_STEPS: int = Field(default=4)
    USE_AMP: bool = Field(default=True, description="Automatic Mixed Precision (bfloat16/fp16)")

    # Inference Sampling
    TEMPERATURE: float = Field(default=0.7)
    TOP_K: int = Field(default=40)
    TOP_P: float = Field(default=0.90)
    REPETITION_PENALTY: float = Field(default=1.05)
    MAX_GEN_TOKENS: int = Field(default=512)

    # RAG & Graph Engine
    RAG_TOP_K: int = Field(default=5)
    RAG_RRF_K: int = Field(default=60)
    GRAPH_MAX_DEPTH: int = Field(default=2)
    DOMAIN_GUARD_SIM_THRESHOLD: float = Field(default=0.22)
    GROUNDEDNESS_THRESHOLD: float = Field(default=0.50)

    # Security & Rate Limiting
    JWT_ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60 * 24)  # 24 Hours
    RATE_LIMIT_PER_MINUTE: int = Field(default=30)

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def model_checkpoint_exists(self) -> bool:
        return self.MODEL_CHECKPOINT_PATH.exists()

    def tokenizer_checkpoint_exists(self) -> bool:
        return self.TOKENIZER_CHECKPOINT_PATH.exists()

    def init_directories(self) -> None:
        """Ensures all required operational directories exist."""
        for folder in [
            self.DATASET_DIR,
            self.MODEL_DIR,
            self.UPLOAD_DIR,
            self.LOG_DIR,
            self.CACHE_DIR,
            self.TEMP_DIR,
        ]:
            folder.mkdir(parents=True, exist_ok=True)


# Instantiated Singleton Settings
settings = AppSettings()
settings.init_directories()
