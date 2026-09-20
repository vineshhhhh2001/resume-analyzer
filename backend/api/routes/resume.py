"""Resume upload and analysis endpoint — the main orchestrator."""

from datetime import datetime, timezone
import uuid

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from core.database import get_db, Analysis
from core.file_processor import extract_text
from nlp.extractor import extract_resume_data, extract_job_description_data
from nlp.matcher import (
    compute_skills_match,
    compute_experience_match,
    compute_education_match,
    compute_semantic_similarity,
)
from nlp.scorer import calculate_overall_score, get_grade, JUNIOR_WEIGHTS
from nlp.gemini_analyzer import analyze_with_gemini
from nlp.openai_analyzer import analyze_with_openai
from nlp.groq_analyzer import analyze_with_groq
from nlp.recommender import (
    generate_recommendations,
    identify_strengths,
    generate_ats_suggestions,
    generate_hiring_assessment,
    generate_negatives,
)

router = APIRouter(prefix="/api/v1", tags=["resume"])


@router.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(..., description="Resume file (PDF or DOCX)"),
    job_description: str = Form(..., description="Job description text"),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a resume and job description, run the full AI analysis pipeline.

    Returns a comprehensive analysis with scores, matched/missing skills,
    recommendations, and ATS suggestions.
    """
    settings = get_settings()

    # ── Validate file ────────────────────────────────────────────────────
    if not resume.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    file_ext = resume.filename.lower().rsplit(".", 1)[-1] if "." in resume.filename else ""
    if file_ext not in ("pdf", "docx", "doc", "txt"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{file_ext}. Please upload PDF, DOCX, or TXT.",
        )

    # Read file bytes
    file_bytes = await resume.read()
    if len(file_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {settings.upload_max_size_mb}MB.",
        )

    if not job_description.strip():
        raise HTTPException(
            status_code=400,
            detail="Job description cannot be empty.",
        )

    # ── Extract text ─────────────────────────────────────────────────────
    try:
        resume_text = extract_text(file_bytes, resume.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # ── Run AI analysis matching configured provider ───────────────────
    ai_result = None

    is_groq_configured = settings.groq_api_key and settings.groq_api_key != "your_groq_api_key_here"

    if settings.ai_provider == "groq" or is_groq_configured:
        try:
            ai_result = analyze_with_groq(resume_text, job_description)
        except Exception as e:
            import logging
            logging.getLogger("uvicorn.error").error(f"[WARN] analyze_with_groq raised exception: {e}")
            ai_result = None

    if not ai_result and (settings.ai_provider == "openai" or settings.openai_api_key):
        try:
            ai_result = analyze_with_openai(resume_text, job_description)
        except Exception as e:
            import logging
            logging.getLogger("uvicorn.error").error(f"[WARN] analyze_with_openai raised exception: {e}")
            ai_result = None

    if not ai_result and settings.gemini_api_key:
        try:
            ai_result = analyze_with_gemini(resume_text, job_description)
        except Exception as e:
            import logging
            logging.getLogger("uvicorn.error").error(f"[WARN] analyze_with_gemini raised exception: {e}")
            ai_result = None

    if ai_result:
        resume_data_clean = ai_result["resume_data"]
        jd_data_clean = ai_result["job_data"]
        overall_score = ai_result["overall_score"]
        grade = ai_result["grade"]
        category_scores = ai_result["category_scores"]
        skills_detail = ai_result["skills_detail"]
        strengths = ai_result["strengths"]
        recommendations = ai_result["recommendations"]
        negatives = ai_result["negatives"]
        ats_suggestions = ai_result["ats_suggestions"]
        hiring_assessment = ai_result["hiring_assessment"]
    else:
        # ── Fallback to local rule-based / transformer matching pipeline ──
        # ── Extract structured data ──────────────────────────────────────────
        resume_data = extract_resume_data(resume_text)
        jd_data = extract_job_description_data(job_description)

        # Store raw text for recommendations engine
        resume_data["_raw_text"] = resume_text
        jd_data["_raw_text"] = job_description

        # ── Run matching algorithms ──────────────────────────────────────────
        # Detect if Job Description is entry level or internship
        jd_title = jd_data.get("title") or ""
        jd_text = job_description.lower()
        is_entry_level = any(kw in jd_title.lower() for kw in ["intern", "junior", "entry level", "fresh", "co-op", "trainee", "associate"]) or \
                         any(kw in jd_text for kw in ["internship", "entry-level", "fresh graduate", "university graduate", "no experience required"])
        min_exp = jd_data.get("min_experience_years")
        if min_exp is not None and min_exp <= 1.0:
            is_entry_level = True

        skills_result = compute_skills_match(
            resume_skills=resume_data["skills"],
            required_skills=jd_data["required_skills"],
            preferred_skills=jd_data["preferred_skills"],
        )

        experience_result = compute_experience_match(
            resume_years=resume_data["total_experience_years"],
            required_years=jd_data["min_experience_years"],
            is_entry_level=is_entry_level,
            projects=resume_data["projects"],
            is_student=resume_data.get("is_student", False),
        )

        education_result = compute_education_match(
            resume_education=resume_data["education"],
            jd_education_requirements=jd_data["education_requirements"],
            is_student=resume_data.get("is_student", False),
            is_entry_level=is_entry_level,
        )

        semantic_result = compute_semantic_similarity(
            resume_text=resume_text,
            jd_text=job_description,
        )

        # ── Calculate scores ─────────────────────────────────────────────────
        category_scores = {
            "skills_match": skills_result["score"],
            "experience_match": experience_result["score"],
            "education_match": education_result["score"],
            "semantic_similarity": semantic_result["score"],
        }

        # Project & Certification relevance boost
        project_boost = 0.0
        if resume_data.get("projects") and category_scores["skills_match"] > 0.0:
            all_jd_skills = set(jd_data.get("required_skills", [])) | set(jd_data.get("preferred_skills", []))
            all_jd_skills_lower = {s.lower() for s in all_jd_skills}
            matching_projects = 0
            for proj in resume_data["projects"]:
                proj_techs = {t.lower() for t in proj.get("technologies", [])}
                if proj_techs & all_jd_skills_lower:
                    matching_projects += 1
            project_boost = min(matching_projects * 2.5, 7.5)  # Max +7.5 points

        cert_boost = 0.0
        if resume_data.get("certifications") and category_scores["skills_match"] > 0.0:
            cert_boost = min(len(resume_data["certifications"]) * 2.5, 5.0)  # Max +5.0 points

        # If a candidate matches absolutely zero skills, they are a complete domain mismatch.
        # Zero out their experience and education matches.
        if category_scores["skills_match"] == 0.0:
            category_scores["experience_match"] = 0.0
            category_scores["education_match"] = 0.0

        # Dynamic weights for student/entry level positions
        active_weights = JUNIOR_WEIGHTS if (is_entry_level or resume_data.get("is_student", False)) else None

        exp_req = jd_data.get("min_experience_years") is not None and jd_data.get("min_experience_years", 0) > 0
        edu_req = bool(jd_data.get("education_requirements"))
        
        if category_scores["skills_match"] == 0.0:
            overall_score = 0.0
        else:
            overall_score = calculate_overall_score(category_scores, weights=active_weights, experience_required=exp_req, education_required=edu_req)
            overall_score = min(overall_score + project_boost + cert_boost, 100.0)
            
        overall_score = round(overall_score, 1)
        grade = get_grade(overall_score)

        # ── Generate recommendations ─────────────────────────────────────────
        match_results = {
            "skills": skills_result,
            "experience": experience_result,
            "education": education_result,
            "semantic": semantic_result,
        }

        recommendations = generate_recommendations(resume_data, jd_data, match_results)
        strengths = identify_strengths(resume_data, jd_data, match_results)
        ats_suggestions = generate_ats_suggestions(resume_text, job_description)
        hiring_assessment = generate_hiring_assessment(resume_data, jd_data, match_results, weights=active_weights)
        negatives = generate_negatives(resume_data, jd_data, match_results)

        # ── Clean data for storage (remove raw text) ─────────────────────────
        resume_data_clean = {k: v for k, v in resume_data.items() if k != "_raw_text"}
        jd_data_clean = {k: v for k, v in jd_data.items() if k != "_raw_text"}
        skills_detail = {
            "matched_skills": skills_result["matched_skills"],
            "missing_required": skills_result["missing_required"],
            "missing_preferred": skills_result["missing_preferred"],
            "extra_skills": skills_result["extra_skills"],
        }

    # ── Store in SQL Database ────────────────────────────────────────────
    analysis_id = uuid.uuid4().hex[:24]
    analysis_doc = Analysis(
        id=analysis_id,
        created_at=datetime.now(timezone.utc),
        status="completed",
        resume_filename=resume.filename,
        resume_file_type=file_ext,
        resume_text=resume_text,
        job_description_text=job_description,
        resume_data=resume_data_clean,
        job_data=jd_data_clean,
        overall_score=overall_score,
        grade=grade,
        category_scores=category_scores,
        skills_detail=skills_detail,
        strengths=strengths,
        recommendations=recommendations,
        negatives=negatives,
        ats_suggestions=ats_suggestions,
        hiring_assessment=hiring_assessment,
    )

    db.add(analysis_doc)
    await db.commit()

    # ── Build response ───────────────────────────────────────────────────
    return {
        "id": analysis_id,
        "created_at": analysis_doc.created_at.isoformat(),
        "status": "completed",
        "overall_score": overall_score,
        "grade": grade,
        "category_scores": category_scores,
        "skills_detail": skills_detail,
        "strengths": strengths,
        "recommendations": recommendations,
        "negatives": negatives,
        "ats_suggestions": ats_suggestions,
        "hiring_assessment": hiring_assessment,
        "resume_data": resume_data_clean,
        "job_data": jd_data_clean,
        "resume_filename": resume.filename,
        "resume_file_type": file_ext,
    }
