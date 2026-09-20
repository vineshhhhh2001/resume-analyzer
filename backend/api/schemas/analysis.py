"""Pydantic schemas for analysis responses."""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ContactInfo(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None


class EducationEntry(BaseModel):
    degree: str
    level: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None


class ExperienceEntry(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    duration: Optional[str] = None
    description: Optional[str] = None


class ProjectEntry(BaseModel):
    title: str
    description: Optional[str] = None
    technologies: list[str] = []


class ResumeData(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    skills: list[str] = []
    skills_categorized: dict[str, list[str]] = {}
    education: list[EducationEntry] = []
    experience: list[ExperienceEntry] = []
    projects: list[ProjectEntry] = []
    certifications: list[str] = []
    total_experience_years: float = 0.0
    is_student: bool = False


class JobDescriptionData(BaseModel):
    title: Optional[str] = None
    all_skills: list[str] = []
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    min_experience_years: Optional[float] = None
    education_requirements: list[str] = []


class CategoryScores(BaseModel):
    skills_match: float = 0.0
    experience_match: float = 0.0
    education_match: float = 0.0
    semantic_similarity: float = 0.0


class GradeInfo(BaseModel):
    grade: str
    label: str
    color: str


class Recommendation(BaseModel):
    priority: str  # high, medium, low
    category: str
    text: str


class ATSSuggestion(BaseModel):
    category: str
    text: str


class ResumeGap(BaseModel):
    category: str
    missing_item: str
    impact: str  # high, medium, low
    suggestion: str


class SkillsDetail(BaseModel):
    matched_skills: list[str] = []
    missing_required: list[str] = []
    missing_preferred: list[str] = []
    extra_skills: list[str] = []


class ChecklistItem(BaseModel):
    criteria: str
    status: str
    details: str
    is_met: bool


class HiringAssessment(BaseModel):
    verdict: str
    verdict_color: str
    leveling: str
    summary: str
    checklist: list[ChecklistItem] = []
    screening_actions: list[str] = []


class AnalysisResult(BaseModel):
    id: str
    created_at: datetime
    status: str  # completed, processing, failed

    # Scores
    overall_score: float
    grade: GradeInfo
    category_scores: CategoryScores

    # Skills breakdown
    skills_detail: SkillsDetail

    # Strengths & recommendations
    strengths: list[str] = []
    recommendations: list[Recommendation] = []
    negatives: list[ResumeGap] = []
    ats_suggestions: list[ATSSuggestion] = []
    hiring_assessment: Optional[HiringAssessment] = None

    # Extracted data
    resume_data: ResumeData
    job_data: JobDescriptionData

    # Metadata
    resume_filename: str
    resume_file_type: str


class AnalysisListItem(BaseModel):
    id: str
    created_at: datetime
    status: str
    overall_score: float
    grade: GradeInfo
    resume_filename: str
    job_title: Optional[str] = None
    matched_skills_count: int = 0
    missing_skills_count: int = 0


class AnalysisListResponse(BaseModel):
    analyses: list[AnalysisListItem]
    total: int
