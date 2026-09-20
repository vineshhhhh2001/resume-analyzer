"""
Structured resume analysis and job matching engine using Google Gemini API.
Provides high-precision extraction, context-aware scoring, and customized recommendations.
Falls back to the local spaCy/transformer pipeline if key is missing or calls fail.
"""

import os
import json
import logging
from typing import Optional
from pydantic import BaseModel, Field

import google.generativeai as genai
from config import get_settings
from nlp.scorer import calculate_overall_score, get_grade, JUNIOR_WEIGHTS

logger = logging.getLogger("uvicorn.error")

# ─── Pydantic Schemas for Gemini Structured Output ─────────────────────────

class EducationEntry(BaseModel):
    degree: str = Field(description="Degree title, e.g., 'Bachelor of Science in Computer Science'")
    level: str = Field(description="Degree level: 'PhD', 'Masters', 'Bachelors', 'Associates', 'Diploma', or '' if none")
    institution: str = Field(description="Name of university, college, or school, or '' if none")
    year: str = Field(description="Graduation year, e.g., '2024', or '' if none")

class ExperienceEntry(BaseModel):
    title: str = Field(description="Job title, e.g., 'Software Engineer', or '' if none")
    company: str = Field(description="Company or employer name, or '' if none")
    duration: str = Field(description="Date range, e.g., 'Jun 2022 - Present', or '' if none")
    description: str = Field(description="Summary of key responsibilities and achievements, or '' if none")
    months: int = Field(description="Estimated duration of this role in months")

class ProjectEntry(BaseModel):
    title: str = Field(description="Project name or title")
    description: str = Field(description="Overview of project goals and achievements, or '' if none")
    technologies: list[str] = Field(description="List of technologies, languages, or tools used in the project")

class ResumeDataSchema(BaseModel):
    name: str = Field(description="Full name of the candidate, or '' if none")
    email: str = Field(description="Email address, or '' if none")
    phone: str = Field(description="Phone number, or '' if none")
    linkedin: str = Field(description="LinkedIn profile URL, or '' if none")
    skills: list[str] = Field(description="Flat list of all technical and soft skills")
    skills_categorized: dict[str, list[str]] = Field(
        description="Skills grouped by category (e.g. 'programming_languages', 'web_frameworks', 'databases', 'soft_skills')"
    )
    education: list[EducationEntry] = Field(description="Education history entries")
    experience: list[ExperienceEntry] = Field(description="Work experience entries")
    projects: list[ProjectEntry] = Field(description="Personal or academic projects")
    certifications: list[str] = Field(description="Professional certifications, e.g., 'AWS Certified Cloud Practitioner'")
    total_experience_years: float = Field(description="Estimated total years of professional work experience, strictly excluding internships, co-ops, trainees, and apprentice roles.")
    is_student: bool = Field(description="Whether the candidate is currently a student or a very recent graduate (within last year)")

class JobDescriptionDataSchema(BaseModel):
    title: str = Field(description="Extracted job title, or '' if none")
    all_skills: list[str] = Field(description="All skills extracted from the job description")
    required_skills: list[str] = Field(description="Core required/essential skills for the role")
    preferred_skills: list[str] = Field(description="Preferred/nice-to-have skills for the role")
    min_experience_years: float = Field(description="Minimum years of experience required (0.0 if not specified)")
    education_requirements: list[str] = Field(description="Required education levels, e.g., ['Bachelors', 'Masters']")

class CategoryScoresSchema(BaseModel):
    skills_match: float = Field(description="Skills compatibility score (0 to 100)")
    experience_match: float = Field(description="Experience compatibility score (0 to 100)")
    education_match: float = Field(description="Education compatibility score (0 to 100)")
    semantic_similarity: float = Field(description="Semantic alignment score (0 to 100)")

class GradeSchema(BaseModel):
    grade: str = Field(description="Letter grade: 'A', 'B', 'C', 'D', 'F'")
    label: str = Field(description="Match label: 'Excellent Match', 'Good Match', 'Fair Match', 'Weak Match', 'Poor Match'")
    color: str = Field(description="Color hex code matching the grade: '#10b981' for A, '#6366f1' for B, '#f59e0b' for C, '#f97316' for D, '#ef4444' for F")

class SkillsDetailSchema(BaseModel):
    matched_skills: list[str] = Field(description="Skills present in both resume and JD")
    missing_required: list[str] = Field(description="Required JD skills missing from the resume")
    missing_preferred: list[str] = Field(description="Preferred JD skills missing from the resume")
    extra_skills: list[str] = Field(description="Candidate skills not specified in the JD")

class RecommendationSchema(BaseModel):
    priority: str = Field(description="Priority: 'high', 'medium', or 'low'")
    category: str = Field(description="Category of suggestion, e.g., 'Skill Deficit', 'Experience Gap', 'Resume Detail'")
    text: str = Field(description="Recruiter-facing screening flag or recommendation comment")

class NegativeSchema(BaseModel):
    category: str = Field(description="Category of the gap, e.g., 'Missing Required Skills', 'Experience Gap'")
    missing_item: str = Field(description="Summary of the missing component or gap, e.g., 'Role requires AWS; resume does not mention it'")
    impact: str = Field(description="Impact: 'high', 'medium', or 'low'")
    suggestion: str = Field(description="Evaluation of the candidate's capability and readiness regarding this gap, specifically how it affects their fit for the role from the hiring team's perspective. Focus strictly on whether they are capable or not, comparing what's in the resume vs what's in the JD.")

class ATSSuggestionSchema(BaseModel):
    category: str = Field(description="Topic or category of screening question")
    text: str = Field(description="Screening/interview question to ask the candidate to probe this specific gap or highlight")

class ChecklistItemSchema(BaseModel):
    criteria: str = Field(description="Criteria evaluated, e.g., 'Skills Criteria Check'")
    status: str = Field(description="Status: 'PASS' or 'FAIL'")
    details: str = Field(description="Detailed verification text")
    is_met: bool = Field(description="Whether the criteria is met")

class HiringAssessmentSchema(BaseModel):
    verdict: str = Field(description="Recruitment verdict: 'Auto-Shortlist (Strong Match)', 'Manual Review / Phone Screen', or 'Hold / Mismatch'")
    verdict_color: str = Field(description="Hex color code for the verdict: '#10b981', '#f59e0b', or '#ef4444'")
    leveling: str = Field(description="Level estimation: 'Junior Level (0-2 years)', 'Mid-Level (3-5 years)', 'Senior Level (6-9 years)', or 'Principal / Lead (10+ years)'")
    summary: str = Field(
        description=(
            "Detailed executive summary narrative covering each and every part of the resume (projects, education, certifications, and experience), "
            "clearly bringing out the pros (strengths, alignment) and cons (gaps, risks). Format as a professional, multi-paragraph report with "
            "clear sections: PROFILE SNAPSHOT, PROS (STRENGTHS & ALIGNMENT), CONS (CRITICAL GAPS & RISKS), and VERDICT JUSTIFICATION."
        )
    )
    checklist: list[ChecklistItemSchema] = Field(description="HR compliance checklist")
    screening_actions: list[str] = Field(description="Recruiter/screener recommended next steps")

class GeminiAnalysisResult(BaseModel):
    resume_data: ResumeDataSchema
    job_data: JobDescriptionDataSchema
    overall_score: float = Field(description="Weighted overall match score (0 to 100)")
    grade: GradeSchema
    category_scores: CategoryScoresSchema
    skills_detail: SkillsDetailSchema
    strengths: list[str] = Field(description="List of major candidate strength highlights relative to the JD")
    recommendations: list[RecommendationSchema] = Field(description="Recruiter-facing notes and screening details")
    negatives: list[NegativeSchema] = Field(description="Resume gaps and actionable improvement advice")
    ats_suggestions: list[ATSSuggestionSchema] = Field(description="Tailored interview/screening questions")
    hiring_assessment: HiringAssessmentSchema


# ─── Gemini Analyzer Execution ─────────────────────────────────────────────

def analyze_with_gemini(resume_text: str, jd_text: str) -> Optional[dict]:
    """
    Perform high-precision resume parsing and job description matching using Gemini API.
    Returns the parsed dict matching the API response/DB schema or None on failure/missing key.
    """
    settings = get_settings()
    api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY")

    if not api_key:
        logger.info("[NLP] GEMINI_API_KEY not configured. Falling back to local NLP pipeline.")
        return None

    try:
        genai.configure(api_key=api_key)
        system_instruction = (
            "You are an elite corporate technical recruiter and ATS parsing specialist. "
            "Your task is to analyze the candidate's resume and compare it against the provided Job Description (JD). "
            "Extract structured data from both texts and calculate precision match scores based on contextual relevance, "
            "experience depth, and actual project history (do not just do simple keyword count). "
            "Formulate recruitment verdicts, strengths, recruiter-facing gap assessments (negatives) with capability evaluations, "
            "and candidate screening questions.\n\n"
            "Experience Calculation Guidelines:\n"
            "- For the 'total_experience_years' field, you must strictly calculate only the duration of completed/full-time professional jobs. "
            "Do NOT include internships, co-ops, traineeships, or apprentice roles in this years sum. "
            "However, you must still extract and list all such roles in the 'experience' field so they are visible in the recruiter history.\n\n"
            "Executive Summary formatting under 'hiring_assessment':\n"
            "The 'summary' field must be a detailed, recruiter-oriented analysis covering each and every part of the resume (projects, education, certifications, and experience). "
            "Always structure it with the following clear labels (using two newlines to separate sections) to clearly bring out the pros and cons of the candidate:\n"
            "PROFILE SNAPSHOT: Discuss the candidate's experience, key past roles, certifications, projects, and education level.\n\n"
            "PROS (STRENGTHS & ALIGNMENT): Highlight the candidate's pros, matching competencies, certifications, and project achievements where they align closest with the JD.\n\n"
            "CONS (CRITICAL GAPS & RISKS): Identify the candidate's cons, missing required/preferred skills, experience gaps, lack of certifications, or educational mismatches.\n\n"
            "VERDICT JUSTIFICATION: Explain why the matching score and verdict was reached and the recommendation for the hiring team.\n\n"
            "Scoring Guidelines:\n"
            "1. Skills Match (0-100): Score based on required and preferred skills. Give partial credit (e.g. 50-80%) "
            "if the candidate has highly relevant/equivalent technologies (e.g. PyTorch instead of TensorFlow) and categorizes them correctly.\n"
            "2. Experience Match (0-100): Assess if the candidate has the years of RELEVANT experience required. "
            "Exclude internships, co-ops, traineeships, and apprentice roles from this calculation. "
            "If the role is entry-level/junior or the candidate is a student, apply a generous project-based boost "
            "(minimum score of 80-85 if they have strong personal/academic projects in the domain).\n"
            "3. Education Match (0-100): Compare degree levels. PhD=5, Masters=4, Bachelors=3, Associates=2, Diploma=1. "
            "If the JD has no education requirement, return 100.0.\n"
            "4. Semantic Similarity (0-100): Evaluate the general profile alignment. How well does the candidate's career "
            "path, project scope, and terminology align with the JD?\n"
            "5. Overall Score (0-100): Calculate the weighted score. Use standard weights (40% skills, 25% experience, "
            "15% education, 20% semantic similarity). Adjust dynamically (renormalize) if experience or education is not required. "
            "If the candidate is a student or the role is junior/entry-level, apply junior weights: (45% skills, 10% experience, "
            "20% education, 25% semantic similarity).\n"
            "6. Grade: Map the overall score to a letter grade: A (>=85, color #10b981), B (>=70, color #6366f1), C (>=55, color #f59e0b), "
            "D (>=40, color #f97316), else F (color #ef4444).\n\n"
            "Personalized Feedback Guidelines:\n"
            "- Group all missing required skills into a single high-impact Negative entry.\n"
            "- If the candidate's highest completed degree is in a field unrelated to computer science/IT/software engineering (e.g. EEE, Mechanical Engineering, Commerce, Arts, etc.), you must explicitly state under CRITICAL GAPS & RISKS in the executive summary that their educational background/degree field is unrelated to the technical role.\n"
            "- For the education check in the compliance checklist, if the degree level matches but the field is unrelated (e.g. EEE for a CS/IT role), set the checklist item status to 'FAIL' and output the detail text exactly as: 'Completed level aligns, but candidate degree '<exact degree name>' is unrelated to the role.' (substituting the candidate's actual parsed degree title, e.g. B.Tech in EEE).\n"
            "- For the capability evaluation (suggestion field) of each negative, analyze whether the candidate is capable or not for this role, comparing what's in the resume vs what's in the JD. Focus strictly on candidate capability and readiness from the hiring team's perspective, NOT on how they should update their resume.\n"
            "- If the candidate is a student, write capability evaluations tailored for entry-level readiness."
        )

        # We will use gemini-2.5-flash which supports structured outputs and is fast & accurate.
        # Fallback to gemini-1.5-flash if needed, but 2.5-flash is current.
        model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=system_instruction)

        prompt = (
            f"=== CANDIDATE RESUME ===\n{resume_text}\n\n"
            f"=== JOB DESCRIPTION ===\n{jd_text}\n\n"
            "Execute the high-precision matching and extract the analysis results."
        )

        import time
        max_retries = 3
        backoff_sec = 2.0
        response = None

        for attempt in range(max_retries):
            try:
                logger.info(f"[NLP] Executing structured resume matching via Gemini API (attempt {attempt + 1}/{max_retries})...")
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=GeminiAnalysisResult,
                        temperature=0.1,
                    ),
                    request_options={"timeout": 60.0}
                )
                break
            except Exception as exc:
                if attempt < max_retries - 1:
                    logger.warning(f"[NLP] Gemini API attempt {attempt + 1} failed: {exc}. Retrying in {backoff_sec}s...")
                    time.sleep(backoff_sec)
                    backoff_sec *= 2
                else:
                    raise exc
        
        result_dict = json.loads(response.text)

        # If the candidate has 0 matched skills, they are a complete domain mismatch.
        # Zero out other scores and force overall score to 0.0.
        matched_skills = result_dict.get("skills_detail", {}).get("matched_skills", [])
        if not matched_skills:
            result_dict["category_scores"]["skills_match"] = 0.0
            result_dict["category_scores"]["experience_match"] = 0.0
            result_dict["category_scores"]["education_match"] = 0.0
            result_dict["overall_score"] = 0.0
            result_dict["grade"] = get_grade(0.0)
            logger.info("[NLP] Complete domain mismatch detected (0 matching skills). Overall score forced to 0.0.")
            return result_dict

        # ── Post-process education match score for students / in-progress degrees ──
        try:
            level_scores = {"PhD": 5, "Masters": 4, "Bachelors": 3, "Associates": 2, "Diploma": 1}
            resume_data = result_dict.get("resume_data", {})
            job_data = result_dict.get("job_data", {})
            
            is_student = resume_data.get("is_student", False)
            edu_entries = resume_data.get("education", [])
            jd_requirements = job_data.get("education_requirements", [])
            
            if jd_requirements:
                required_level = max((level_scores.get(req, 0) for req in jd_requirements), default=0)
                
                # Determine candidate highest level and if it is in progress
                highest_level = 0
                highest_entry = None
                for edu in edu_entries:
                    lvl = edu.get("level")
                    if lvl in level_scores:
                        score = level_scores[lvl]
                        if score > highest_level:
                            highest_level = score
                            highest_entry = edu
                            
                degree_in_progress = False
                if highest_entry:
                    degree_text = highest_entry.get("degree", "").lower()
                    inst_text = highest_entry.get("institution", "")
                    inst_text = inst_text.lower() if inst_text else ""
                    yr_str = highest_entry.get("year")
                    yr_str = yr_str.lower() if yr_str else ""
                    
                    import datetime
                    current_year = datetime.datetime.now().year
                    
                    in_progress_terms = ["current", "pursuing", "ongoing", "expected", "in progress", "in-progress", "studying", "present"]
                    
                    if any(term in degree_text for term in in_progress_terms) or \
                       any(term in inst_text for term in in_progress_terms) or \
                       any(term in yr_str for term in in_progress_terms):
                        degree_in_progress = True
                    elif yr_str and yr_str.isdigit() and int(yr_str) > current_year:
                        degree_in_progress = True
                    elif is_student:
                        if yr_str and yr_str.isdigit() and int(yr_str) <= current_year:
                            degree_in_progress = False
                        else:
                            degree_in_progress = True
                
                # Enforce completed vs in-progress levels
                completed_level = highest_level
                if degree_in_progress:
                    completed_level = max(0, highest_level - 1)
                    
                # Evaluate score with major relevance check
                is_major_relevant = True
                if highest_entry:
                    tech_degree_keywords = [
                        "computer", "software", "engineering", "information technology",
                        "science", "math", "systems", "application", "it", "cs", "cse",
                        "mca", "bca", "technology", "programming", "developer",
                        "tech", "btech", "mtech", "b.tech", "m.tech", "b.e", "b.e."
                    ]
                    degree_title = highest_entry.get("degree", "").lower()
                    unrelated_major_keywords = [
                        "electrical", "electronics", "eee", "ece", "mechanical",
                        "civil", "chemical", "biotech", "biotechnology",
                        "telecommunication", "instrumentation", "aerospace",
                        "industrial", "production", "power"
                    ]
                    if any(ukw in degree_title for ukw in unrelated_major_keywords):
                        is_major_relevant = False
                    elif not any(kw in degree_title for kw in tech_degree_keywords):
                        is_major_relevant = False

                jd_title = job_data.get("title") or ""
                min_exp = job_data.get("min_experience_years")
                is_entry_level = any(kw in jd_title.lower() for kw in ["intern", "junior", "entry level", "fresh", "co-op", "trainee", "associate"]) or \
                                 (min_exp is not None and min_exp <= 1.0)

                # Cross-check with LLM's own assessment of the degree relevance
                edu_fail = False
                for item in result_dict.get("hiring_assessment", {}).get("checklist", []):
                    if item.get("criteria") == "Education Criteria Check" and item.get("status") == "FAIL":
                        edu_fail = True
                        break

                if completed_level >= required_level:
                    edu_score = 100.0 if (is_major_relevant and not edu_fail) else 0.0
                else:
                    edu_score = 0.0
                    
                result_dict["category_scores"]["education_match"] = round(edu_score, 1)
                
                exp_req = job_data.get("min_experience_years") is not None and job_data.get("min_experience_years", 0) > 0
                edu_req = True
                active_weights = JUNIOR_WEIGHTS if (is_entry_level or is_student) else None
                
                overall_score = calculate_overall_score(
                    result_dict["category_scores"],
                    weights=active_weights,
                    experience_required=exp_req,
                    education_required=edu_req
                )
            else:
                jd_title = job_data.get("title") or ""
                min_exp = job_data.get("min_experience_years")
                is_entry_level = any(kw in jd_title.lower() for kw in ["intern", "junior", "entry level", "fresh", "co-op", "trainee", "associate"]) or \
                                 (min_exp is not None and min_exp <= 1.0)
                if is_student and not is_entry_level:
                    edu_score = 0.0
                else:
                    edu_score = 100.0
                result_dict["category_scores"]["education_match"] = round(edu_score, 1)
                
                exp_req = job_data.get("min_experience_years") is not None and job_data.get("min_experience_years", 0) > 0
                edu_req = False
                active_weights = JUNIOR_WEIGHTS if (is_entry_level or is_student) else None
                
                overall_score = calculate_overall_score(
                    result_dict["category_scores"],
                    weights=active_weights,
                    experience_required=exp_req,
                    education_required=edu_req
                )

            # Apply Project & Certification relevance boosts to overall score
            project_boost = 0.0
            if resume_data.get("projects") and result_dict["category_scores"]["skills_match"] > 0.0:
                all_jd_skills = set(job_data.get("required_skills", [])) | set(job_data.get("preferred_skills", []))
                all_jd_skills_lower = {s.lower() for s in all_jd_skills}
                matching_projects = 0
                for proj in resume_data["projects"]:
                    proj_techs = {t.lower() for t in proj.get("technologies", [])}
                    if proj_techs & all_jd_skills_lower:
                        matching_projects += 1
                project_boost = min(matching_projects * 2.5, 7.5)

            cert_boost = 0.0
            if resume_data.get("certifications") and result_dict["category_scores"]["skills_match"] > 0.0:
                cert_boost = min(len(resume_data["certifications"]) * 2.5, 5.0)

            overall_score = min(overall_score + project_boost + cert_boost, 100.0)
            result_dict["overall_score"] = round(overall_score, 1)
            result_dict["grade"] = get_grade(overall_score)
        except Exception as pe:
            logger.error(f"[WARN] Failed to post-process education match: {pe}")

        logger.info("[NLP] Gemini matching completed successfully.")
        return result_dict

    except Exception as e:
        logger.error(f"[WARN] Gemini matching failed: {e}. Falling back to local NLP pipeline.")
        return None
