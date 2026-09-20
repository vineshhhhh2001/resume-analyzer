"""
SQL database persistence layer using SQLAlchemy and aiosqlite.
Replaces the old file-based/MongoDB storage with fully relational SQL storage.
Includes automatic zero-loss migration for existing JSON analyses.
"""

from collections.abc import AsyncGenerator
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import uuid

from sqlalchemy import String, Float, DateTime, Text, JSON
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config import get_settings

# Get database URL from settings
settings = get_settings()
DATABASE_URL = settings.database_url

# Create Async Engine and SessionMaker
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)

async_session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


class Analysis(Base):
    """Analysis model representing stored resume analysis results."""

    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(24), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    status: Mapped[str] = mapped_column(String(50), default="completed")
    resume_filename: Mapped[str] = mapped_column(String(255), nullable=True)
    resume_file_type: Mapped[str] = mapped_column(String(10), nullable=True)
    resume_text: Mapped[str] = mapped_column(Text, nullable=True)
    job_description_text: Mapped[str] = mapped_column(Text, nullable=True)
    
    # Store nested structured data as JSON columns
    resume_data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True)
    job_data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True)
    overall_score: Mapped[float] = mapped_column(Float, nullable=True)
    grade: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True)
    category_scores: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True)
    skills_detail: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True)
    strengths: Mapped[list[str]] = mapped_column(JSON, nullable=True)
    recommendations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    negatives: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    ats_suggestions: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    hiring_assessment: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True)


async def init_db() -> None:
    """Initialize database schema (creates tables if they do not exist)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("[OK] SQL Database schema initialized")
    
    # Trigger auto-migration
    await migrate_existing_data()


async def close_db() -> None:
    """Close the database connection pool."""
    await engine.dispose()
    print("[OK] SQL Database connections closed")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency generator for database sessions in FastAPI routes."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def migrate_existing_data() -> None:
    """
    Migrates existing JSON analyses from storage/analyses directory to the SQL database.
    Runs only if the SQL database is currently empty.
    """
    storage_dir = Path(__file__).parent.parent / "storage"
    analyses_dir = storage_dir / "analyses"
    if not analyses_dir.exists():
        return

    async with async_session_maker() as session:
        from sqlalchemy import select, func
        # Check if database already has any records
        count_stmt = select(func.count(Analysis.id))
        result = await session.execute(count_stmt)
        count = result.scalar() or 0
        if count > 0:
            # Table already has analyses, skip migration
            return

        json_files = list(analyses_dir.glob("*.json"))
        if not json_files:
            return

        print(f"[MIGRATION] Scanning storage/analyses/ for JSON records to migrate (found {len(json_files)})...")
        migrated_count = 0

        for filepath in json_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    doc = json.load(f)

                doc_id = doc.get("_id") or doc.get("id") or filepath.stem
                
                # Parse created_at datetime
                created_at_val = doc.get("created_at")
                if isinstance(created_at_val, str):
                    try:
                        created_at = datetime.fromisoformat(created_at_val.replace("Z", "+00:00"))
                    except ValueError:
                        created_at = datetime.now(timezone.utc)
                else:
                    created_at = datetime.now(timezone.utc)

                analysis = Analysis(
                    id=doc_id,
                    created_at=created_at,
                    status=doc.get("status", "completed"),
                    resume_filename=doc.get("resume_filename", ""),
                    resume_file_type=doc.get("resume_file_type", ""),
                    resume_text=doc.get("resume_text", ""),
                    job_description_text=doc.get("job_description_text", ""),
                    resume_data=doc.get("resume_data", {}),
                    job_data=doc.get("job_data", {}),
                    overall_score=float(doc.get("overall_score", 0.0)),
                    grade=doc.get("grade", {}),
                    category_scores=doc.get("category_scores", {}),
                    skills_detail=doc.get("skills_detail", {}),
                    strengths=doc.get("strengths", []),
                    recommendations=doc.get("recommendations", []),
                    negatives=doc.get("negatives", []),
                    ats_suggestions=doc.get("ats_suggestions", []),
                    hiring_assessment=doc.get("hiring_assessment", {}),
                )
                session.add(analysis)
                migrated_count += 1
            except Exception as e:
                print(f"[WARN] Failed to parse and prepare migration for {filepath.name}: {e}")

        if migrated_count > 0:
            try:
                await session.commit()
                print(f"[MIGRATION] Successfully migrated {migrated_count} analyses from JSON to the SQL database!")
            except Exception as e:
                await session.rollback()
                print(f"[ERROR] Migration commit failed: {e}")
