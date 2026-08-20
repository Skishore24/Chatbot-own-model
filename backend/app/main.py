"""
backend/app/main.py
----------------------------------------------------
GENKIT AI v6.0 FastAPI Server Entrypoint
"""

import sys
import time
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch

# Ensure backend directory in sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.core.logger import logger, new_trace_id
from app.api import api_v1_router

# Master FastAPI Application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Genkit AI — 100% Self-Hosted Custom LLM + Hybrid RAG Assistant.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# Mount API Routers
app.include_router(api_v1_router)
# Also alias under /api/v5 for backward compatibility
app.include_router(api_v1_router, prefix="/api/v5")


@app.get("/")
async def root():
    cuda_status = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_status else "CPU"
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "device": device_name,
        "cuda_available": cuda_status,
        "model_trained": settings.model_checkpoint_exists(),
        "api_v1": "/api/v1",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    from app.api.health import get_health
    return await get_health()


def start_server():
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} on {settings.HOST}:{settings.PORT}")
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)


if __name__ == "__main__":
    start_server()
