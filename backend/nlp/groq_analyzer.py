"""
Structured resume analysis and job matching engine using Groq API (Llama-3-70b-8192).
Provides high-precision extraction, context-aware scoring, and recruiter-facing evaluations.
"""

import os
import json
import logging
from typing import Optional
from pydantic import BaseModel, Field

from groq import Groq
from config import get_settings
from nlp.scorer import calculate_overall_score, get_grade, JUNIOR_WEIGHTS

logger = logging.getLogger("uvicorn.error")

# ─── Pydantic Schemas for Groq Structured Output ─────────────────────────
# Groq natively supports response_format={"type": "json_object"} when we ask for JSON in prompt.

class GroqAnalysisResult(BaseModel):
    # This schema is used purely for documentation of the expected output.
    # Groq JSON output will be parsed into a dict that matches this structure.
    pass

# ─── Groq Analyzer Execution ─────────────────────────────────────────────

def analyze_with_groq(resume_text: str, jd_text: str) -> Optional[dict]:
    """
    Perform structured resume matching using Groq API (llama3-70b-8192).
    """
    settings = get_settings()
    api_key = settings.groq_api_key or os.environ.get("GROQ_API_KEY")

    if not api_key or api_key == "your_groq_api_key_here":
        logger.info("[NLP] GROQ_API_KEY not configured or is placeholder. Falling back to next available provider.")
        return None

    try:
        # Initialize Groq Client
        client = Groq(api_key=api_key)

        # We must explicitly ask for JSON output in the system prompt for Groq JSON mode to work
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
            "- If the candidate is a student, write capability evaluations tailored for entry-level readiness.\n\n"
            "CRITICAL INSTRUCTION:\n"
            "You MUST return ONLY a valid JSON object. Do not include any markdown formatting like ```json or ```. Just raw JSON.\n"
            "The JSON must have this exact structure:\n"
            "{\n"
            '  "resume_data": {\n'
            '    "name": "", "email": "", "phone": "", "linkedin": "", "skills": [], "skills_categorized": {}, "education": [{"degree": "", "level": "", "institution": "", "year": ""}], "experience": [{"title": "", "company": "", "duration": "", "description": "", "months": 0}], "projects": [{"title": "", "description": "", "technologies": []}], "certifications": [], "total_experience_years": 0.0, "is_student": false\n'
            "  },\n"
            '  "job_data": {\n'
            '    "title": "", "all_skills": [], "required_skills": [], "preferred_skills": [], "min_experience_years": 0.0, "education_requirements": []\n'
            "  },\n"
            '  "overall_score": 0.0,\n'
            '  "grade": {"grade": "", "label": "", "color": ""},\n'
            '  "category_scores": {"skills_match": 0.0, "experience_match": 0.0, "education_match": 0.0, "semantic_similarity": 0.0},\n'
            '  "skills_detail": {"matched_skills": [], "missing_required": [], "missing_preferred": [], "extra_skills": []},\n'
            '  "strengths": [],\n'
            '  "recommendations": [{"priority": "", "category": "", "text": ""}],\n'
            '  "negatives": [{"category": "", "missing_item": "", "impact": "", "suggestion": ""}],\n'
            '  "ats_suggestions": [{"category": "", "text": ""}],\n'
            '  "hiring_assessment": {\n'
            '    "verdict": "", "verdict_color": "", "leveling": "", "summary": "",\n'
            '    "checklist": [{"criteria": "", "status": "", "details": "", "is_met": true}],\n'
            '    "screening_actions": []\n'
            "  }\n"
            "}\n"
        )

        prompt = (
            f"=== CANDIDATE RESUME ===\n{resume_text}\n\n"
            f"=== JOB DESCRIPTION ===\n{jd_text}\n\n"
            "Execute the high-precision matching and extract the analysis results. Return ONLY JSON."
        )

        logger.info("[NLP] Executing structured resume matching via Groq API (llama3-70b-8192)...")
        
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        result_dict = json.loads(response.choices[0].message.content)

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
                    yr_str = str(highest_entry.get("year", ""))
                    yr_str = yr_str.lower()
                    
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

        logger.info("[NLP] Groq matching completed successfully.")
        return result_dict

    except Exception as e:
        logger.error(f"[WARN] Groq matching failed: {e}.")
        return None
