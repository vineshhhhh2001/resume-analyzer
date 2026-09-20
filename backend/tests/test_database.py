"""Tests for the SQL database operations."""

from datetime import datetime, timezone
import pytest
from sqlalchemy import select

from core.database import Base, Analysis, engine, async_session_maker

# Set pytest-asyncio to auto or default
pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
async def setup_db():
    """Create a clean database schema for each test."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def test_db_crud_operations():
    """Test inserting, querying, and deleting an Analysis row in SQLite."""
    async with async_session_maker() as session:
        # Create a mock analysis record
        new_analysis = Analysis(
            id="testanalysis123456789abc",
            created_at=datetime.now(timezone.utc),
            status="completed",
            resume_filename="test_resume.pdf",
            resume_file_type="pdf",
            resume_text="John Doe Python developer",
            job_description_text="Python developer needed",
            resume_data={"skills": ["Python"]},
            job_data={"required_skills": ["Python"]},
            overall_score=85.0,
            grade={"grade": "A", "label": "Excellent", "color": "green"},
            category_scores={"skills_match": 90.0, "experience_match": 80.0, "education_match": 100.0, "semantic_similarity": 70.0},
            skills_detail={"matched_skills": ["Python"], "missing_required": [], "missing_preferred": [], "extra_skills": []},
            strengths=["Python programming"],
            recommendations=[],
            negatives=[],
            ats_suggestions=[],
            hiring_assessment={"verdict": "Hire", "verdict_color": "green", "leveling": "Mid", "summary": "Good", "checklist": [], "screening_actions": []},
        )
        
        session.add(new_analysis)
        await session.commit()
        
    async with async_session_maker() as session:
        # Query the inserted record
        stmt = select(Analysis).where(Analysis.id == "testanalysis123456789abc")
        result = await session.execute(stmt)
        analysis = result.scalar_one_or_none()
        
        assert analysis is not None
        assert analysis.resume_filename == "test_resume.pdf"
        assert analysis.overall_score == 85.0
        assert analysis.resume_data["skills"] == ["Python"]
        assert analysis.hiring_assessment["verdict"] == "Hire"
        
        # Delete the record
        await session.delete(analysis)
        await session.commit()
        
    async with async_session_maker() as session:
        # Verify it was deleted
        stmt = select(Analysis).where(Analysis.id == "testanalysis123456789abc")
        result = await session.execute(stmt)
        analysis = result.scalar_one_or_none()
        
        assert analysis is None
