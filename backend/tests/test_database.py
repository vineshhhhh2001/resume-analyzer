
"""Tests for the SQL database operations."""

from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select

from core.database import Base, Analysis, engine, async_session_maker

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    """Create a clean database schema for each test."""
    print("DATABASE URL:", engine.url)
    async with engine.begin() as conn:
        # Create all registered database tables
        await conn.run_sync(Base.metadata.create_all)

        # Diagnostic: display tables created in this database
        tables = await conn.run_sync(
            lambda sync_conn: sync_conn.dialect.get_table_names(sync_conn)
        )
        print("DATABASE TABLES:", tables)

    yield

    # Clean up tables after the test
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def test_db_crud_operations():
    """Test inserting, querying, and deleting an Analysis row."""

    # INSERT
    async with async_session_maker() as session:
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
            grade={
                "grade": "A",
                "label": "Excellent",
                "color": "green",
            },
            category_scores={
                "skills_match": 90.0,
                "experience_match": 80.0,
                "education_match": 100.0,
                "semantic_similarity": 70.0,
            },
            skills_detail={
                "matched_skills": ["Python"],
                "missing_required": [],
                "missing_preferred": [],
                "extra_skills": [],
            },
            strengths=["Python programming"],
            recommendations=[],
            negatives=[],
            ats_suggestions=[],
            hiring_assessment={
                "verdict": "Hire",
                "verdict_color": "green",
                "leveling": "Mid",
                "summary": "Good",
                "checklist": [],
                "screening_actions": [],
            },
        )

        session.add(new_analysis)
        await session.commit()

    # READ
    async with async_session_maker() as session:
        stmt = select(Analysis).where(
            Analysis.id == "testanalysis123456789abc"
        )
        result = await session.execute(stmt)
        analysis = result.scalar_one_or_none()

        assert analysis is not None
        assert analysis.resume_filename == "test_resume.pdf"
        assert analysis.overall_score == 85.0
        assert analysis.resume_data["skills"] == ["Python"]
        assert analysis.hiring_assessment["verdict"] == "Hire"

        # DELETE
        await session.delete(analysis)
        await session.commit()

    # VERIFY DELETION
    async with async_session_maker() as session:
        stmt = select(Analysis).where(
            Analysis.id == "testanalysis123456789abc"
        )
        result = await session.execute(stmt)
        analysis = result.scalar_one_or_none()

        assert analysis is None
