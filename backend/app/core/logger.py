"""
backend/app/core/logger.py
----------------------------------------------------
GENKIT AI v5.0 Enterprise Logging & Telemetry Subsystem
Structured JSON & Colored Console Logger with Trace UUID propagation.
"""

import sys
import os
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from colorama import Fore, Style, init

from .config import settings

# Initialize colorama for Windows/Unix CLI support
init(autoreset=True)


class JSONFormatter(logging.Formatter):
    """Formats log records into structured JSON lines for enterprise observability."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "trace_id": getattr(record, "trace_id", "N/A"),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_fields"):
            log_data.update(getattr(record, "extra_fields"))

        return json.dumps(log_data)


class ColoredConsoleFormatter(logging.Formatter):
    """Formats log records with colorama colors for CLI debugging."""

    COLOR_MAP = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLOR_MAP.get(record.levelno, Fore.WHITE)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trace_id = getattr(record, "trace_id", "main")
        prefix = f"[{timestamp}] [{color}{record.levelname:<8}{Style.RESET_ALL}] [{record.name}] [Trace: {trace_id}]"
        message = record.getMessage()

        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            return f"{prefix} {message}\n{exc_text}"

        return f"{prefix} {message}"


def get_logger(name: str = "GenkitAI") -> logging.Logger:
    """Configures and returns a structured logger instance."""
    logger_inst = logging.getLogger(name)

    if logger_inst.handlers:
        return logger_inst

    logger_inst.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    logger_inst.propagate = False

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    console_handler.setFormatter(ColoredConsoleFormatter())
    logger_inst.addHandler(console_handler)

    # JSON File Handler
    log_file = settings.LOG_DIR / "app_v5.json.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(JSONFormatter())
    logger_inst.addHandler(file_handler)

    return logger_inst


# Global default logger
logger = get_logger("GenkitCore")


def new_trace_id() -> str:
    """Generates a unique request trace ID."""
    return str(uuid.uuid4())[:8]
