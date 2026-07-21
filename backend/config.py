"""
config.py
----------------------------------------------------
Genkit AI - Production Configuration
Features
--------
✓ Environment Configuration
✓ GPU Detection
✓ MySQL Configuration
✓ AI Configuration
✓ Directory Management
✓ Logging
✓ Security
✓ Production Ready
Author : Genkit AI
"""
import os
import sys
import platform
from pathlib import Path
import torch
from dotenv import load_dotenv
# ============================================================
# PROJECT PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
BACKEND_DIR = BASE_DIR
FRONTEND_DIR = ROOT_DIR / "frontend"
DATASET_DIR = BACKEND_DIR / "dataset"
MODEL_DIR = BACKEND_DIR / "genkit-model"
UPLOAD_DIR = BACKEND_DIR / "uploads"
LOG_DIR = BACKEND_DIR / "logs"
CACHE_DIR = BACKEND_DIR / "cache"
TEMP_DIR = BACKEND_DIR / "temp"
DATABASE_DIR = BACKEND_DIR / "database"
# ============================================================
# CREATE DIRECTORIES
# ============================================================
for folder in [
    DATASET_DIR,
    MODEL_DIR,
    UPLOAD_DIR,
    LOG_DIR,
    CACHE_DIR,
    TEMP_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True
    )
# ============================================================
# LOAD ENVIRONMENT
# ============================================================
ENV_FILE = BACKEND_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
# ============================================================
# APPLICATION
# ============================================================
APP_NAME = "Genkit AI"
APP_VERSION = "4.0.0"
AUTHOR = "Genkit AI"
DEBUG = os.getenv(
    "DEBUG",
    "False"
).lower() == "true"
HOST = os.getenv(
    "HOST",
    "0.0.0.0"
)
PORT = int(
    os.getenv(
        "PORT",
        8000
    )
)
SECRET_KEY = os.getenv(
    "SECRET_KEY",
    "genkit-secret-key"
)
API_KEY = os.getenv(
    "API_KEY",
    ""
)

# ✅ Security: In production, replace "*" with specific origins e.g.:
# ALLOWED_ORIGINS=https://yoursite.com,https://www.yoursite.com
# Wildcard "*" allows any origin (CORS bypass risk in production).
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in
    os.getenv(
        "ALLOWED_ORIGINS",
        "*"
    ).split(",")
]
# ============================================================
# PLATFORM
# ============================================================
OPERATING_SYSTEM = platform.system()
PYTHON_VERSION = platform.python_version()
MACHINE = platform.machine()
PROCESSOR = platform.processor()
# ============================================================
# MYSQL CONFIGURATION
# ============================================================
MYSQL_HOST = os.getenv(
    "MYSQL_HOST",
    "localhost"
)
MYSQL_PORT = int(
    os.getenv(
        "MYSQL_PORT",
        "3306"
    )
)
MYSQL_DATABASE = os.getenv(
    "MYSQL_DATABASE",
    "genkit_ai"
)
MYSQL_USER = os.getenv(
    "MYSQL_USER",
    "root"
)
MYSQL_PASSWORD = os.getenv(
    "MYSQL_PASSWORD",
    ""
)
MYSQL_POOL_SIZE = int(
    os.getenv(
        "MYSQL_POOL_SIZE",
        "10"
    )
)
MYSQL_AUTOCOMMIT = False
# ============================================================
# DATASET
# ============================================================
DATASET_PATH = DATASET_DIR / "dataset.json"
LEGACY_DATASET = BACKEND_DIR / "dataset.json"
if (not DATASET_PATH.exists()) and LEGACY_DATASET.exists():
    DATASET_PATH = LEGACY_DATASET
# ============================================================
# MODEL FILES
# ============================================================
MODEL_FILE = MODEL_DIR / "model.pt"
CONFIG_FILE = MODEL_DIR / "config.json"
VOCAB_FILE = MODEL_DIR / "vocab.json"
TOKENIZER_FILE = MODEL_DIR / "bpe_tokenizer.json"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(
    exist_ok=True
)
# ============================================================
# DEVICE CONFIGURATION
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_GPU = DEVICE == "cuda"
GPU_NAME = ""
GPU_MEMORY = 0
if USE_GPU:
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY = round(
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3,
        2
    )
CPU_THREADS = os.cpu_count()
# ============================================================
# LLM CONFIGURATION
# ============================================================
BLOCK_SIZE = 1024
VOCAB_SIZE = 5000
EMBED_DIM = 384
NUM_HEADS = 8
NUM_LAYERS = 10
DROPOUT = 0.10
MAX_INPUT_LENGTH = 1024
MAX_OUTPUT_LENGTH = 1024
TEMPERATURE = float(
    os.getenv(
        "TEMPERATURE",
        "0.7"
    )
)
TOP_K = int(
    os.getenv(
        "TOP_K",
        "40"
    )
)
TOP_P = float(
    os.getenv(
        "TOP_P",
        "0.90"
    )
)
REPETITION_PENALTY = float(
    os.getenv(
        "REPETITION_PENALTY",
        "1.05"
    )
)
# ============================================================
# TRAINING
# ============================================================
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
WARMUP_STEPS = 500
SAVE_EVERY = 5
# ============================================================
# NLP CONFIGURATION
# ============================================================
DOMAIN_CONFIDENCE_THRESHOLD = 0.25   # min score for in-domain
INTENT_CONFIDENCE_THRESHOLD = 0.30   # min score for intent
COREFERENCE_ENABLED = True           # resolve pronouns
LEMMATIZATION_ENABLED = True         # apply lemmatization
# ============================================================
# RAG CONFIGURATION
# ============================================================
TOP_DOCUMENTS = 5
MIN_SIMILARITY = 0.20
MAX_CONTEXT_DOCUMENTS = 5
MAX_HISTORY_MESSAGES = 5
# ============================================================
# MEMORY CONFIGURATION
# ============================================================
MAX_MEMORY_ITEMS = 100
SESSION_TIMEOUT = 3600
CACHE_SIZE = 1000
# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# ============================================================
# LOGGING
# ============================================================
sys.path.insert(0, str(BACKEND_DIR))
from utils.logger import (
    setup_logging,
    log_startup
)
logger = setup_logging(
    str(LOG_DIR),
    debug=DEBUG
)
# ============================================================
# STARTUP LOG
# ============================================================
log_startup()
logger.info("=" * 70)
logger.info(f"Application : {APP_NAME}")
logger.info(f"Version     : {APP_VERSION}")
logger.info(f"Author      : {AUTHOR}")
logger.info(f"Debug Mode  : {DEBUG}")
logger.info(f"Host        : {HOST}")
logger.info(f"Port        : {PORT}")
logger.info(f"OS          : {OPERATING_SYSTEM}")
logger.info(f"Python      : {PYTHON_VERSION}")
logger.info(f"Device      : {DEVICE}")
if USE_GPU:
    logger.info(f"GPU         : {GPU_NAME}")
    logger.info(f"GPU Memory  : {GPU_MEMORY} GB")
else:
    logger.info("GPU         : Not Available")
logger.info(f"MySQL DB    : {MYSQL_DATABASE}")
logger.info(f"Dataset     : {DATASET_PATH}")
logger.info(f"Model Dir   : {MODEL_DIR}")
logger.info("=" * 70)
# ============================================================
# VALIDATION
# ============================================================
def validate_environment():
    required = [
        DATASET_DIR,
        MODEL_DIR,
        LOG_DIR,
        UPLOAD_DIR,
    ]
    for folder in required:
        if not folder.exists():
            raise RuntimeError(
                f"Missing folder: {folder}"
            )
validate_environment()
# ============================================================
# MODEL VALIDATION
# ============================================================
MODEL_READY = True
for file in [
    CONFIG_FILE,
    MODEL_FILE,
    VOCAB_FILE
]:
    if not file.exists():
        MODEL_READY = False
        logger.warning(
            f"Missing model file : {file.name}"
        )
if MODEL_READY:
    logger.info(
        "Custom GPT model detected."
    )
else:
    logger.warning(
        "Model is not trained yet."
    )
# ============================================================
# PROJECT INFORMATION
# ============================================================
PROJECT_INFO = {
    "name": APP_NAME,
    "version": APP_VERSION,
    "author": AUTHOR,
    "device": DEVICE,
    "gpu": USE_GPU,
    "database": MYSQL_DATABASE,
    "dataset": str(DATASET_PATH),
    "model": str(MODEL_DIR),
}
# ============================================================
# EXPORTS
# ============================================================
__all__ = [
    "APP_NAME",
    "APP_VERSION",
    "AUTHOR",
    "DEBUG",
    "HOST",
    "PORT",
    "DEVICE",
    "USE_GPU",
    "GPU_NAME",
    "GPU_MEMORY",
    "MYSQL_HOST",
    "MYSQL_PORT",
    "MYSQL_DATABASE",
    "MYSQL_USER",
    "MYSQL_PASSWORD",
    "MODEL_DIR",
    "MODEL_FILE",
    "CONFIG_FILE",
    "VOCAB_FILE",
    "TOKENIZER_FILE",
    "DATASET_DIR",
    "DATASET_PATH",
    "UPLOAD_DIR",
    "LOG_DIR",
    "CACHE_DIR",
    "TEMP_DIR",
    "CHECKPOINT_DIR",
    "TEMPERATURE",
    "TOP_K",
    "TOP_P",
    "BLOCK_SIZE",
    "VOCAB_SIZE",
    "EMBED_DIM",
    "NUM_HEADS",
    "NUM_LAYERS",
    "DROPOUT",
    "MAX_INPUT_LENGTH",
    "MAX_OUTPUT_LENGTH",
    "TOP_DOCUMENTS",
    "MAX_CONTEXT_DOCUMENTS",
    "MAX_HISTORY_MESSAGES",
    "MODEL_READY",
    "PROJECT_INFO",
    "logger",
    # NLP config
    "DOMAIN_CONFIDENCE_THRESHOLD",
    "INTENT_CONFIDENCE_THRESHOLD",
    "COREFERENCE_ENABLED",
    "LEMMATIZATION_ENABLED",
]
