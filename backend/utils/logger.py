"""
utils/logger.py
----------------------------------------------------
Genkit AI - Production Logging System
Features
--------
✓ Console Logging
✓ Rotating File Logging
✓ Error Logging
✓ Database Logging
✓ Training Logging
✓ API Logging
✓ UTF-8 Support
✓ Thread Safe
Author : Genkit AI
"""
import os
import logging
from logging.handlers import RotatingFileHandler
# ============================================================
# LOG FORMAT
# ============================================================
LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)-8s | "
    "%(name)s | "
    "%(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
# ============================================================
# THIRD PARTY LOGGERS
# ============================================================
SUPPRESS_LOGGERS = [
    "mysql",
    "mysql.connector",
    "watchfiles",
    "watchfiles.main",
    "asyncio",
    "uvicorn.access",
    "urllib3",
    "httpx",
    "PIL",
]
# ============================================================
# LOGGER FACTORY
# ============================================================
def get_logger(name="Genkit AI"):
    return logging.getLogger(name)
# ============================================================
# MAIN LOGGER
# ============================================================
def setup_logging(
    log_dir,
    debug=False
):
    os.makedirs(
        log_dir,
        exist_ok=True
    )
    level = (
        logging.DEBUG
        if debug
        else logging.INFO
    )
    logger = logging.getLogger()
    logger.setLevel(level)
    if logger.handlers:
        logger.handlers.clear()
    formatter = logging.Formatter(
        LOG_FORMAT,
        DATE_FORMAT
    )
    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    # --------------------------------------------------------
    # Main Log
    # --------------------------------------------------------
    file_handler = RotatingFileHandler(
        os.path.join(
            log_dir,
            "genkit.log"
        ),
        maxBytes=10 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # --------------------------------------------------------
    # Error Log
    # --------------------------------------------------------
    error_handler = RotatingFileHandler(
        os.path.join(
            log_dir,
            "error.log"
        ),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    error_handler.setLevel(
        logging.ERROR
    )
    error_handler.setFormatter(formatter)
    logger.addHandler(
        error_handler
    )
    # --------------------------------------------------------
    # Silence Libraries
    # --------------------------------------------------------
    for lib in SUPPRESS_LOGGERS:
        logging.getLogger(lib).setLevel(
            logging.WARNING
        )
    app_logger = logging.getLogger(
        "Genkit AI"
    )
    app_logger.info(
        "Logging initialized."
    )
    return app_logger
# ============================================================
# SPECIALIZED LOGGERS
# ============================================================
database_logger = logging.getLogger(
    "Genkit.Database"
)
api_logger = logging.getLogger(
    "Genkit.API"
)
training_logger = logging.getLogger(
    "Genkit.Training"
)
rag_logger = logging.getLogger(
    "Genkit.RAG"
)
llm_logger = logging.getLogger(
    "Genkit.LLM"
)
security_logger = logging.getLogger(
    "Genkit.Security"
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def log_database(message, level="info"):
    getattr(
        database_logger,
        level.lower(),
        database_logger.info
    )(message)

def log_api(message, level="info"):
    getattr(
        api_logger,
        level.lower(),
        api_logger.info
    )(message)

def log_training(message, level="info"):
    getattr(
        training_logger,
        level.lower(),
        training_logger.info
    )(message)

def log_rag(message, level="info"):
    getattr(
        rag_logger,
        level.lower(),
        rag_logger.info
    )(message)

def log_llm(message, level="info"):
    getattr(
        llm_logger,
        level.lower(),
        llm_logger.info
    )(message)

def log_security(message, level="warning"):
    getattr(
        security_logger,
        level.lower(),
        security_logger.warning
    )(message)

# ============================================================
# EXCEPTION LOGGER
# ============================================================
def log_exception(
    logger_obj,
    exception,
    message=""
):
    if message:
        logger_obj.error(message)
    logger_obj.exception(exception)

# ============================================================
# STARTUP INFORMATION
# ============================================================
def log_startup():
    logger = get_logger()
    logger.info("=" * 60)
    logger.info("Genkit AI Starting...")
    logger.info("Production Logging Enabled")
    logger.info("=" * 60)

# ============================================================
# SHUTDOWN INFORMATION
# ============================================================
def log_shutdown():
    logger = get_logger()
    logger.info("=" * 60)
    logger.info("Genkit AI Shutdown Complete")
    logger.info("=" * 60)

# ============================================================
# EXPORTS
# ============================================================
__all__ = [
    "setup_logging",
    "get_logger",
    "database_logger",
    "api_logger",
    "training_logger",
    "rag_logger",
    "llm_logger",
    "security_logger",
    "log_database",
    "log_api",
    "log_training",
    "log_rag",
    "log_llm",
    "log_security",
    "log_exception",
    "log_startup",
    "log_shutdown",
]
