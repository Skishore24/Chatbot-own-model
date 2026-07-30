"""
backend/main.py
----------------------------------------------------
GENKIT AI v5.0 Enterprise Server Launcher
Delegates execution to the production FastAPI application in app.main.
"""

import sys
from pathlib import Path

# Ensure backend directory is in sys.path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app
from app.core.config import settings
from app.core.logger import logger

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Launching {settings.APP_NAME} (v{settings.APP_VERSION}) on http://{settings.HOST}:{settings.PORT}...")
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
