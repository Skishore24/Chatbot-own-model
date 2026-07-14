"""
main.py
----------------------------------------------------
Genkit AI - Production FastAPI Server
Architecture
1. Environment Validation
2. Database Initialization
3. Vector Store Initialization
4. LLM Initialization
5. API Startup
6. Static Frontend
7. Health Monitoring
Author : Genkit AI
"""
import os
import sys
import time
from contextlib import asynccontextmanager
sys.path.insert(
    0,
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from config import (
    APP_NAME,
    APP_VERSION,
    DEBUG,
    HOST,
    PORT,
    FRONTEND_DIR,
    ALLOWED_ORIGINS,
    logger,
)
from utils.logger import (
    log_startup,
    log_shutdown,
)
from api.routes import (
    router as api_router
)
from api.admin import (
    router as admin_router
)
# ============================================================
# APPLICATION START TIME
# ============================================================
START_TIME = time.time()
# ============================================================
# LIFESPAN
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 70)
    logger.info(
        "Starting Genkit AI..."
    )
    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------
    try:
        from database import init_db
        init_db()
        logger.info(
            "MySQL initialized."
        )
    except Exception:
        logger.exception(
            "Database initialization failed."
        )
    # --------------------------------------------------------
    # Vector Store
    # --------------------------------------------------------
    try:
        from ai.embeddings.embedding import (
            init_vector_store
        )
        init_vector_store()
        logger.info(
            "Vector Store initialized."
        )
    except Exception:
        logger.exception(
            "Vector Store initialization failed."
        )
    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------
    try:
        from ai.llm.inference import (
            load_model
        )
        ready = load_model()
        if ready:
            logger.info(
                "Custom GPT loaded."
            )
        else:
            logger.warning(
                "Model not trained."
            )
    except Exception:
        logger.exception(
            "Model initialization failed."
        )
    log_startup()
    logger.info("=" * 70)
    yield
    logger.info("=" * 70)
    logger.info(
        "Stopping Genkit AI..."
    )
    log_shutdown()
    logger.info("=" * 70)
# ============================================================
# FASTAPI
# ============================================================
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    lifespan=lifespan,
    description="""
Genkit AI
Own GPT Model
Own Tokenizer
Own RAG
Own Vector Database
MySQL Backend
No External APIs
""",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)
# ============================================================
# MIDDLEWARE
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    GZipMiddleware,
    minimum_size=1024
)
# ============================================================
# REQUEST LOGGER
# ============================================================
@app.middleware("http")
async def log_requests(
    request: Request,
    call_next
):
    start = time.time()
    logger.info(
        f"{request.method} {request.url.path}"
    )
    response = await call_next(request)
    duration = round(
        time.time() - start,
        3
    )
    logger.info(
        f"{request.method} "
        f"{request.url.path} "
        f"{response.status_code} "
        f"{duration}s"
    )
    response.headers["X-Process-Time"] = str(duration)
    return response

# ============================================================
# GLOBAL EXCEPTION HANDLER
# ============================================================
@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception
):
    logger.exception(exc)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal Server Error"
        }
    )

# ============================================================
# ROOT
# ============================================================
@app.get("/")
async def root():
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "running",
        "message": "Welcome to Genkit AI"
    }

# ============================================================
# HEALTH
# ============================================================
@app.get("/health")
async def health():
    uptime = round(
        time.time() - START_TIME,
        2
    )
    return {
        "status": "healthy",
        "uptime": uptime,
        "device": os.getenv(
            "DEVICE",
            "cpu"
        )
    }

# ============================================================
# VERSION
# ============================================================
@app.get("/version")
async def version():
    return {
        "application": APP_NAME,
        "version": APP_VERSION,
        "debug": DEBUG
    }

# ============================================================
# ROUTERS
# ============================================================
app.include_router(
    api_router
)
app.include_router(
    admin_router
)
# ============================================================
# STATIC FRONTEND
# ============================================================
if os.path.exists(FRONTEND_DIR):
    logger.info(
        f"Frontend found: {FRONTEND_DIR}"
    )
    app.mount(
        "/",
        StaticFiles(
            directory=FRONTEND_DIR,
            html=True
        ),
        name="frontend"
    )
else:
    logger.warning(
        f"Frontend directory not found: {FRONTEND_DIR}"
    )

# ============================================================
# STARTUP VALIDATION
# ============================================================
def validate_startup():
    logger.info("=" * 70)
    logger.info("Genkit AI Startup Validation")
    logger.info("=" * 70)
    folders = [
        "frontend",
        "dataset",
        "genkit-model",
        "logs",
        "uploads"
    ]
    backend = os.path.dirname(os.path.abspath(__file__))
    for folder in folders:
        if folder == "frontend":
            path = os.path.join(os.path.dirname(backend), folder)
        else:
            path = os.path.join(backend, folder)
        if os.path.exists(path):
            logger.info(f"✓ {folder}")
        else:
            logger.warning(f"✗ {folder}")
    logger.info("=" * 70)

validate_startup()

# ============================================================
# SHUTDOWN EVENT
# ============================================================
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("=" * 70)
    logger.info("Stopping Genkit AI")
    logger.info("=" * 70)
    try:
        from database import close_pool
        close_pool()
        logger.info("Database pool closed.")
    except Exception:
        pass

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 70)
    logger.info("Launching Uvicorn Server")
    logger.info("=" * 70)
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        workers=1,
        log_level="info",
        access_log=True,
        reload_dirs=[
            os.path.dirname(
                os.path.abspath(__file__)
            )
        ],
        reload_excludes=[
            "*.log",
            "*.pyc",
            "__pycache__",
            "logs",
            "uploads",
            "cache",
            "temp",
            "genkit-model",
            "dataset"
        ]
    )
