"""
AI-Powered Resume Analyzer -- FastAPI Application

Main entry point for the backend API server.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from core.database import init_db, close_db
from nlp.models import preload_models
from api.routes.resume import router as resume_router
from api.routes.analysis import router as analysis_router
from api.routes.health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle -- startup and shutdown events."""
    # Startup
    print("[START] Starting Resume Analyzer API...")

    # Connect to database
    try:
        await init_db()
    except Exception as e:
        print(f"[WARN] Database init failed: {e}")

    # Pre-load NLP models
    try:
        preload_models()
    except Exception as e:
        print(f"[WARN] NLP model loading failed: {e}")
        print("   Models will be loaded on first request.")

    print("[READY] Resume Analyzer API is ready!")

    yield

    # Shutdown
    print("[STOP] Shutting down...")
    await close_db()


# -- Create FastAPI app ---

app = FastAPI(
    title="AI Resume Analyzer API",
    description=(
        "AI-powered resume analysis and job matching system. "
        "Upload resumes and job descriptions to get match scores, "
        "skill analysis, and improvement recommendations."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# -- CORS Middleware ---

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Register Routers ---

app.include_router(resume_router)
app.include_router(analysis_router)
app.include_router(health_router)


# -- Root Endpoint ---

@app.get("/")
async def root():
    return {
        "name": "AI Resume Analyzer API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)

