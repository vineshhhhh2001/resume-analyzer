"""Analysis CRUD endpoints — list, get, delete past analyses using SQL storage."""

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from core.database import get_db, Analysis

router = APIRouter(prefix="/api/v1", tags=["analyses"])


@router.get("/analyses")
async def list_analyses(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """List all past analyses with pagination."""
    # Get total count
    total_query = select(func.count(Analysis.id))
    total_res = await db.execute(total_query)
    total = total_res.scalar() or 0

    # Fetch analyses (most recent first)
    # Project specific columns to optimize listing performance
    stmt = select(
        Analysis.id,
        Analysis.created_at,
        Analysis.status,
        Analysis.overall_score,
        Analysis.grade,
        Analysis.resume_filename,
        Analysis.job_data,
        Analysis.skills_detail,
        Analysis.category_scores,
    ).order_by(desc(Analysis.created_at)).offset(skip).limit(limit)

    res = await db.execute(stmt)
    analyses = []
    for row in res.all():
        created_at = row.created_at
        # Handle datetime formatting
        if hasattr(created_at, "isoformat"):
            created_at = created_at.isoformat()

        job_data = row.job_data or {}
        skills_detail = row.skills_detail or {}

        analyses.append({
            "id": row.id,
            "created_at": created_at,
            "status": row.status,
            "overall_score": row.overall_score,
            "grade": row.grade,
            "resume_filename": row.resume_filename,
            "job_title": job_data.get("title"),
            "matched_skills_count": len(skills_detail.get("matched_skills", [])),
            "missing_skills_count": len(skills_detail.get("missing_required", [])),
            "category_scores": row.category_scores,
        })

    return {
        "analyses": analyses,
        "total": total,
    }


@router.get("/analyses/{analysis_id}")
async def get_analysis(analysis_id: str, db: AsyncSession = Depends(get_db)):
    """Get a specific analysis by ID."""
    stmt = select(Analysis).where(Analysis.id == analysis_id)
    res = await db.execute(stmt)
    analysis = res.scalar_one_or_none()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    created_at = analysis.created_at
    if hasattr(created_at, "isoformat"):
        created_at = created_at.isoformat()

    return {
        "id": analysis.id,
        "created_at": created_at,
        "status": analysis.status,
        "overall_score": analysis.overall_score,
        "grade": analysis.grade,
        "category_scores": analysis.category_scores,
        "skills_detail": analysis.skills_detail,
        "strengths": analysis.strengths,
        "recommendations": analysis.recommendations,
        "negatives": analysis.negatives,
        "ats_suggestions": analysis.ats_suggestions,
        "hiring_assessment": analysis.hiring_assessment,
        "resume_data": analysis.resume_data,
        "job_data": analysis.job_data,
        "resume_filename": analysis.resume_filename,
        "resume_file_type": analysis.resume_file_type,
    }


@router.delete("/analyses/{analysis_id}")
async def delete_analysis(analysis_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a specific analysis."""
    stmt = select(Analysis).where(Analysis.id == analysis_id)
    res = await db.execute(stmt)
    analysis = res.scalar_one_or_none()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    await db.delete(analysis)
    await db.commit()

    return {"message": "Analysis deleted successfully.", "id": analysis_id}
