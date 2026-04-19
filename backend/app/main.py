import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes import router
from db.database import init_db
from services.vector_store import load_and_split, add_documents
from app.config import APP_NAME, APP_VERSION, FRONTEND_DIR, logger


# ─────────────────────────────────────────────
# LIFECYCLE
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 Starting {APP_NAME} v{APP_VERSION}...")
    
    # 1. Initialize Database
    try:
        init_db()
    except Exception as e:
        logger.error(f"FATAL: Database initialization failed: {e}")
    
    # 2. Initialize Vector Store
    try:
        docs = load_and_split()
        add_documents(docs)
    except Exception as e:
        logger.error(f"CRITICAL: Vector Store initialization failed: {e}")

    yield
    logger.info("🛑 Shutting down server...")


# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)


# ─────────────────────────────────────────────
# FRONTEND SERVING
# ─────────────────────────────────────────────
if os.path.exists(FRONTEND_DIR):
    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    # Mount static files
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
else:
    logger.warning(f"Frontend directory not found at {FRONTEND_DIR}. API only mode.")

@app.get("/favicon.ico")
async def favicon():
    favicon_path = os.path.join(FRONTEND_DIR, "images", "logo1.png")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return {"status": "no favicon"}


if __name__ == "__main__":
    import uvicorn
    logger.info("Initializing Uvicorn...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)