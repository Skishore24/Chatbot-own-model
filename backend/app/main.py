"""
backend/app/main.py
----------------------------------------------------
GENKIT AI v5.0 Enterprise FastAPI Application Server Entrypoint
"""

import sys
import time
from pathlib import Path

# Ensure backend directory is in sys.path for clean package imports
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.core.config import settings
from app.core.logger import logger, new_trace_id
from app.api.routes import router as api_router

# Instantiate Master FastAPI Application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise-grade AI Assistant written natively in PyTorch and Python.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Trace ID & Timing Middleware
@app.middleware("http")
async def add_telemetry_headers(request: Request, call_next):
    trace_id = new_trace_id()
    start_time = time.time()

    request.state.trace_id = trace_id
    response = await call_next(request)

    latency_ms = (time.time() - start_time) * 1000
    response.headers["X-Trace-ID"] = trace_id
    response.headers["X-Response-Time-MS"] = f"{latency_ms:.2f}"

    return response


# Include Router
app.include_router(api_router)


@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "docs": "/docs",
    }


if __name__ == "__main__":
    logger.info(f"Starting {settings.APP_NAME} server on {settings.HOST}:{settings.PORT}...")
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
