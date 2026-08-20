"""
backend/app/core/config.py
----------------------------------------------------
GENKIT AI v6.0 Configuration Subsystem
Single source of truth for paths, model hyperparameters, RAG parameters, and DB settings.
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
    APP_NAME: str = "Genkit AI Assistant"
    APP_VERSION: str = "6.0.0"
    AUTHOR: str = "Genkit.in"
    ENVIRONMENT: str = Field(default="development", description="development, staging, production")
    DEBUG: bool = Field(default=False)
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    SECRET_KEY: str = Field(default="genkit-ai-v6-secret-key-32bytes-long-signature")
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
    def MODEL_DIR(self) -> Path:
        return self.BASE_DIR / "genkit-model"

    @property
    def MODEL_CHECKPOINT_PATH(self) -> Path:
        for candidate in ["model_v6.pt", "model_v5.pt", "model.pt"]:
            p = self.MODEL_DIR / candidate
            if p.exists():
                return p
        return self.MODEL_DIR / "model_v6.pt"

    @property
    def TOKENIZER_CHECKPOINT_PATH(self) -> Path:
        for candidate in ["bpe_tokenizer_v6.json", "bpe_tokenizer_v5.json", "tokenizer.json"]:
            p = self.MODEL_DIR / candidate
            if p.exists():
                return p
        return self.MODEL_DIR / "bpe_tokenizer_v6.json"

    @property
    def CONFIG_CHECKPOINT_PATH(self) -> Path:
        for candidate in ["config_v6.json", "config_v5.json", "config.json"]:
            p = self.MODEL_DIR / candidate
            if p.exists():
                return p
        return self.MODEL_DIR / "config_v6.json"

    @property
    def LOG_DIR(self) -> Path:
        return self.BASE_DIR / "logs"

    # MySQL Database Configuration
    MYSQL_HOST: str = Field(default="localhost")
    MYSQL_PORT: int = Field(default=3306)
    MYSQL_USER: str = Field(default="root")
    MYSQL_PASSWORD: str = Field(default="")
    MYSQL_DATABASE: str = Field(default="genkit_ai")
    MYSQL_MIN_POOL_SIZE: int = Field(default=2)
    MYSQL_MAX_POOL_SIZE: int = Field(default=10)

    # Custom LLM v6.0 Architecture Hyperparameters (Optimized for RTX 3050 6GB GPU)
    # Model size: ~75M-85M parameters
    VOCAB_SIZE: int = Field(default=10000, description="Byte-Fallback BPE Vocab Size")
    BLOCK_SIZE: int = Field(default=512, description="Max sequence context window")
    EMBED_DIM: int = Field(default=384, description="Hidden dimension d_model")
    NUM_LAYERS: int = Field(default=6, description="Transformer Decoder Layers")
    NUM_HEADS: int = Field(default=6, description="Query Attention Heads (H_Q)")
    NUM_KV_HEADS: int = Field(default=2, description="Key-Value Attention Heads for GQA")
    DROPOUT: float = Field(default=0.10)
    BIAS: bool = Field(default=False)
    ROPE_FREQ_BASE: float = Field(default=10000.0)

    # Training Hyperparameters
    BATCH_SIZE: int = Field(default=4, description="Micro-batch size for training")
    GRADIENT_ACCUMULATION_STEPS: int = Field(default=8, description="Accumulation steps (effective batch = 32)")
    EPOCHS: int = Field(default=60)
    LEARNING_RATE: float = Field(default=3e-4)
    MIN_LEARNING_RATE: float = Field(default=1e-5)
    WEIGHT_DECAY: float = Field(default=0.1)
    GRADIENT_CLIP: float = Field(default=1.0)
    WARMUP_STEPS: int = Field(default=200)
    USE_AMP: bool = Field(default=True, description="Automatic Mixed Precision")

    # Generation & Sampling
    TEMPERATURE: float = Field(default=0.65)
    TOP_K: int = Field(default=40)
    TOP_P: float = Field(default=0.88)
    REPETITION_PENALTY: float = Field(default=1.12)
    MAX_NEW_TOKENS: int = Field(default=384)

    # RAG Engine Parameters
    RAG_TOP_K: int = Field(default=4)
    RAG_BM25_K1: float = Field(default=1.5)
    RAG_BM25_B: float = Field(default=0.75)
    RAG_FUSION_BM25_WEIGHT: float = Field(default=0.60)
    RAG_FUSION_TFIDF_WEIGHT: float = Field(default=0.40)
    RAG_CONFIDENCE_THRESHOLD: float = Field(default=0.25)
    GROUNDEDNESS_THRESHOLD: float = Field(default=0.40)

    # Security & Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=60)
    MAX_PROMPT_LENGTH: int = Field(default=2000)

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
        """Ensures operational directories exist."""
        for folder in [self.DATASET_DIR, self.MODEL_DIR, self.LOG_DIR]:
            folder.mkdir(parents=True, exist_ok=True)


# Instantiated Singleton Settings
settings = AppSettings()
settings.init_directories()
