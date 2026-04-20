import os
import logging
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BACKEND_DIR)

# Load .env safely
ENV_PATH = os.path.join(BACKEND_DIR, ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
APP_NAME = "Genkit AI"
APP_VERSION = "2.1.0"

DEBUG = os.getenv("DEBUG", "False").lower() == "true"
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

# ─────────────────────────────────────────────
# PATH CONFIG (ABSOLUTE)
# ─────────────────────────────────────────────
ML_DIR = os.path.join(BACKEND_DIR, "ml")
MODEL_DIR = os.path.join(ML_DIR, "genkit-model")
DATASET_PATH = os.path.join(ML_DIR, "dataset.json")

DB_PATH = os.path.join(BACKEND_DIR, "genkit.db")
CHROMA_PATH = os.path.join(BACKEND_DIR, "chroma_db")

FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

# ─────────────────────────────────────────────
# AUTO CREATE DIRECTORIES (IMPORTANT FIX)
# ─────────────────────────────────────────────
os.makedirs(ML_DIR, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

# ─────────────────────────────────────────────
# LOGGING (PRODUCTION READY)
# ─────────────────────────────────────────────
LOG_LEVEL = logging.DEBUG if DEBUG else logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(APP_NAME)

# ─────────────────────────────────────────────
# ENV SAFETY CHECKS
# ─────────────────────────────────────────────
if DEBUG:
    logger.debug("Running in DEBUG mode")
else:
    logger.info("Running in PRODUCTION mode")

# Warn if model not found (prevents silent GPT fallback bugs)
if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    logger.warning(
        f"⚠️ Model not found at {MODEL_DIR}\n"
        "👉 Train model using: python ml/train.py"
    )

# ─────────────────────────────────────────────
# TRANSFORMER SETTINGS (OPTIMIZED)
# ─────────────────────────────────────────────
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Optional: reduce torch warnings
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"