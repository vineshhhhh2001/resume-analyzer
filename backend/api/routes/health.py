"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
async def health_check():
    """Check service health and readiness."""
    try:
        from nlp.models import _nlp_model, _sentence_model

        return {
            "status": "healthy",
            "storage": "file-based",
            "nlp_model": "loaded" if _nlp_model else "not loaded",
            "sentence_model": "loaded" if _sentence_model else "not loaded",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }
