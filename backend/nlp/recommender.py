"""Resume improvement recommendation engine."""

import json
import re
from pathlib import Path
from typing import Any

from nlp.extractor import _get_skills_db


def _load_ats_data() -> dict:
    """Load ATS keywords data."""
    data_path = Path(__file__).parent.parent / "data" / "ats_keywords.json"
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_recommendations(
    resume_data: dict[str, Any],
    jd_data: dict[str, Any],
    match_results: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Generate recruiter-facing screening red flags and assessment notes.

    Returns a list of insight dicts with priority, category, and text.
    """
    insights = []

    # 1. Critical Skill Deficiencies
    missing_required = match_results.get("skills", {}).get("missing_required", [])
    if missing_required:
        if len(missing_required) <= 3:
            skill_list = ", ".join(missing_required)
            insights.append({
                "priority": "high",
                "category": "Skill Deficit",
                "text": f"Candidate lacks the following critical required skills: {skill_list}. Verification of equivalent technical knowledge is highly recommended.",
            })
        else:
            top_skills = ", ".join(missing_required[:4])
            insights.append({
                "priority": "high",
                "category": "Skill Deficit",
                "text": f"Candidate is missing {len(missing_required)} core required skills. Major gaps detected in: {top_skills}.",
            })

    # 2. Experience Gap
    exp_result = match_results.get("experience", {})
    if exp_result.get("required_years") and exp_result.get("resume_years", 0) < exp_result["required_years"]:
        gap = exp_result["required_years"] - exp_result.get("resume_years", 0)
        insights.append({
            "priority": "high",
            "category": "Experience Gap",
            "text": f"Candidate's experience ({exp_result.get('resume_years', 0):.1f} years) falls short of the required {exp_result['required_years']:.1f} years by {gap:.1f} years. Assess depth of projects during phone screening.",
        })

    # 3. Education Gap
    edu_result = match_results.get("education", {})
    if edu_result.get("score", 100) < 70:
        req_levels = ", ".join(jd_data.get("education_requirements", []))
        insights.append({
            "priority": "medium",
            "category": "Education Gap",
            "text": f"Candidate degree credentials do not fully meet the specified requirements (Requires: {req_levels or 'Bachelors'}). Validate certifications and work history equivalents.",
        })

    # 4. Missing Preferred Skill Sets
    missing_preferred = match_results.get("skills", {}).get("missing_preferred", [])
    if missing_preferred:
        skill_list = ", ".join(missing_preferred[:4])
        insights.append({
            "priority": "low",
            "category": "Preferred Gaps",
            "text": f"Candidate lacks the following preferred/nice-to-have skill qualifiers: {skill_list}.",
        })

    # 5. Content Metrics & Low Terminology Matching
    semantic_score = match_results.get("semantic", {}).get("score", 100)
    if semantic_score < 50:
        insights.append({
            "priority": "medium",
            "category": "Relevance Alert",
            "text": "Low semantic keyword alignment with the job description. The candidate's background might be focused on a different domain or tech stack.",
        })

    # 6. Absence of Quantifiable Achievements
    resume_text = resume_data.get("_raw_text", "")
    has_numbers = bool(re.search(r"\d+%|\$\d+|\d+\s*(?:users|customers|projects|clients)", resume_text, re.IGNORECASE))
    if not has_numbers:
        insights.append({
            "priority": "low",
            "category": "Resume Detail",
            "text": "Work descriptions do not highlight quantifiable metrics or numeric outcomes. Recruiter should ask for concrete metrics during screening.",
        })

    return insights


def identify_strengths(
    resume_data: dict[str, Any],
    jd_data: dict[str, Any],
    match_results: dict[str, Any],
) -> list[str]:
    """Identify candidate's strengths relative to the job description."""
    strengths = []

    # Skills strengths
    matched = match_results.get("skills", {}).get("matched_skills", [])
    required = jd_data.get("required_skills", [])
    if required and len(matched) >= len(required) * 0.7:
        strengths.append(
            f"Strong skill match — candidate has {len(matched)} of {len(required)} required skills."
        )

    # Experience strength
    exp = match_results.get("experience", {})
    if exp.get("required_years") and exp.get("resume_years", 0) >= exp["required_years"]:
        strengths.append(
            f"Experience meets/exceeds requirement ({exp['resume_years']:.0f} vs {exp['required_years']:.0f} years needed)."
        )

    # Education strength
    edu = match_results.get("education", {})
    if edu.get("score", 0) >= 90:
        strengths.append("Education level meets or exceeds the job requirement.")

    # Extra skills
    extra = match_results.get("skills", {}).get("extra_skills", [])
    if extra and len(extra) >= 3:
        top_extra = ", ".join(list(extra)[:5])
        strengths.append(
            f"Additional valuable skills: {top_extra}."
        )

    # Certifications
    if resume_data.get("certifications"):
        strengths.append(
            f"Relevant certifications: {', '.join(resume_data['certifications'][:3])}."
        )

    # Semantic match
    semantic = match_results.get("semantic", {}).get("score", 0)
    if semantic >= 65:
        strengths.append("Resume terminology aligns well with the job description.")

    if not strengths:
        strengths.append("Candidate demonstrates a diverse skill profile.")

    return strengths


def generate_ats_suggestions(
    resume_text: str,
    jd_text: str,
) -> list[dict[str, str]]:
    """
    Generate tailored interview screening questions for the recruiter based on profile gaps.

    Returns a list of suggestion dicts with category (topic) and text (question).
    """
    questions = []

    # Extract skills and matcher results locally
    from nlp.extractor import extract_resume_data, extract_job_description_data
    from nlp.matcher import compute_skills_match, compute_experience_match

    resume_data = extract_resume_data(resume_text)
    jd_data = extract_job_description_data(jd_text)

    skills_result = compute_skills_match(
        resume_skills=resume_data["skills"],
        required_skills=jd_data["required_skills"],
        preferred_skills=jd_data["preferred_skills"],
    )

    experience_result = compute_experience_match(
        resume_years=resume_data["total_experience_years"],
        required_years=jd_data["min_experience_years"],
        is_student=resume_data.get("is_student", False),
    )

    # 1. Questions about missing required skills
    missing_req = skills_result.get("missing_required", [])
    if missing_req:
        top_missing = missing_req[0]
        questions.append({
            "category": f"Missing Core: {top_missing}",
            "text": f"The position requires experience with '{top_missing}', which is not mentioned on your resume. Could you describe your level of exposure to this or a related technology?"
        })
        if len(missing_req) > 1:
            second_missing = missing_req[1]
            questions.append({
                "category": f"Missing Core: {second_missing}",
                "text": f"How have you solved problems using '{second_missing}'? If you haven't used it directly, what similar tools/systems have you worked with?"
            })

    # 2. Questions about experience gaps
    req_years = experience_result.get("required_years")
    res_years = experience_result.get("resume_years", 0)
    if req_years and res_years < req_years:
        questions.append({
            "category": "Experience Depth",
            "text": f"This role specifies {req_years:.0f} years of experience, while your resume shows approximately {res_years:.1f} years. Can you elaborate on the complex projects you ran that demonstrate your readiness for this scope?"
        })

    # 3. Questions about preferred qualifications
    missing_pref = skills_result.get("missing_preferred", [])
    if missing_pref:
        top_pref = missing_pref[0]
        questions.append({
            "category": f"Preferred Skill: {top_pref}",
            "text": f"We prefer candidates to have exposure to '{top_pref}'. Have you worked with this tool or concept, and how would you approach utilizing it in this role?"
        })

    # 4. Challenge and metric verification question based on history
    if resume_data.get("experience"):
        last_job = resume_data["experience"][0]
        title = last_job.get("title") or "your recent role"
        company = last_job.get("company") or "your last employer"
        questions.append({
            "category": "Technical Impact",
            "text": f"In your role as '{title}' at '{company}', what is an example of a project where you drove measurable business or technical results?"
        })
    else:
        questions.append({
            "category": "Technical Overview",
            "text": "Could you walk us through the system architecture of a major project you contributed to, explaining the decisions and trade-offs you made?"
        })

    # 5. Certification question (if mentioned in job description)
    db = _get_skills_db()
    cert_keywords = db.get("certifications_keywords", [])
    jd_certs = [c for c in cert_keywords if c.lower() in jd_text.lower()]
    if jd_certs and not resume_data.get("certifications"):
        questions.append({
            "category": f"Verification: {jd_certs[0]}",
            "text": f"The job details list '{jd_certs[0]}' as a relevant certification. Do you currently hold this, or do you have plan to obtain this certification?"
        })

    return questions


def generate_hiring_assessment(
    resume_data: dict[str, Any],
    jd_data: dict[str, Any],
    match_results: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Generate an executive hiring recommendation and verification checklist for HR.
    """
    from nlp.scorer import calculate_overall_score

    exp_req = jd_data.get("min_experience_years") is not None and jd_data.get("min_experience_years", 0) > 0
    edu_req = bool(jd_data.get("education_requirements"))
    overall_score = calculate_overall_score({
        "skills_match": match_results["skills"]["score"],
        "experience_match": match_results["experience"]["score"],
        "education_match": match_results["education"]["score"],
        "semantic_similarity": match_results["semantic"]["score"],
    }, weights=weights, experience_required=exp_req, education_required=edu_req)
    
    # 1. Determine Recruitment Verdict
    if overall_score >= 75:
        verdict = "Auto-Shortlist (Strong Match)"
        verdict_color = "#10b981" # green
        action_verdict = "Advance to Technical Screen"
    elif overall_score >= 50:
        verdict = "Manual Review / Phone Screen"
        verdict_color = "#f59e0b" # amber
        action_verdict = "Schedule Recruiter Phone Screen"
    else:
        verdict = "Hold / Mismatch"
        verdict_color = "#ef4444" # red
        action_verdict = "Review for Alternative Roles"

    # 2. Candidate Leveling based on parsed years of experience
    res_years = resume_data.get("total_experience_years", 0)
    if res_years >= 10:
        leveling = "Principal / Lead (10+ years)"
    elif res_years >= 6:
        leveling = "Senior Level (6-9 years)"
    elif res_years >= 3:
        leveling = "Mid-Level (3-5 years)"
    else:
        leveling = "Junior Level (0-2 years)"

    # 3. Formulate Checklist Items
    checklist = []
    
    # Skills Check
    matched_skills = match_results["skills"].get("matched_skills", [])
    required_skills = jd_data.get("required_skills", [])
    if required_skills:
        skills_met = len(matched_skills) >= len(required_skills) * 0.6
        skills_text = f"Matched {len(matched_skills)} of {len(required_skills)} required core skills."
    else:
        skills_met = True
        skills_text = "No required skills specified in Job Description."
    checklist.append({
        "criteria": "Skills Criteria Check",
        "status": "PASS" if skills_met else "FAIL",
        "details": skills_text,
        "is_met": skills_met
    })
    
    # Experience Check
    req_years = jd_data.get("min_experience_years")
    if req_years:
        exp_met = res_years >= req_years
        exp_text = f"Candidate has {res_years:.1f} years, role specifies min {req_years:.1f} years."
    else:
        exp_met = True
        exp_text = f"Candidate has {res_years:.1f} years (no minimum specified)."
    checklist.append({
        "criteria": "Experience Criteria Check",
        "status": "PASS" if exp_met else "FAIL",
        "details": exp_text,
        "is_met": exp_met
    })
    
    # Education Check
    edu_score = match_results["education"].get("score", 100)
    edu_met = edu_score >= 70
    req_edu = ", ".join(jd_data.get("education_requirements", []))
    
    # Find candidate highest degree title
    candidate_degree = ""
    if resume_data.get("education"):
        level_scores = {"PhD": 5, "Masters": 4, "Bachelors": 3, "Associates": 2, "Diploma": 1}
        highest_score = -1
        for edu in resume_data["education"]:
            score = level_scores.get(edu.get("level", ""), 0)
            if score > highest_score:
                highest_score = score
                candidate_degree = edu.get("degree", "")

    if edu_met:
        edu_text = f"Credentials align with requirements ({req_edu or 'Bachelors'})."
    elif "unrelated" in match_results.get("education", {}).get("detail", "").lower():
        degree_name = candidate_degree.replace("\n", " ").strip() if candidate_degree else "unrelated degree"
        edu_text = f"Completed level aligns, but candidate degree '{degree_name}' is unrelated to the role."
    else:
        edu_text = f"Credentials do not meet requirements ({req_edu or 'Bachelors'})."
        
    checklist.append({
        "criteria": "Education Criteria Check",
        "status": "PASS" if edu_met else "FAIL",
        "details": edu_text,
        "is_met": edu_met
    })

    # 4. Formulate Recruiter Action Plan
    actions = []
    missing_required = match_results["skills"].get("missing_required", [])
    if missing_required:
        actions.append(f"Technical Screen Focus: Assess depth in missing core skills ({', '.join(missing_required[:3])}).")
    
    if req_years and res_years < req_years:
        gap = req_years - res_years
        actions.append(f"Experience Check: Verify if candidate's projects bridge the {gap:.1f}-year experience gap.")
        
    if not edu_met:
        actions.append("Credential Check: Verify professional equivalence of education background or certifications.")
        
    if overall_score < 60:
        actions.append("Portfolio Review: Check candidate github or previous work portfolio to double-check core competence.")
    
    if not actions:
        actions.append("Standard Screening: Proceed with standard hiring process and verify culture fit.")

    # 5. Build Narrative Summary
    is_edu_unrelated = "unrelated" in match_results.get("education", {}).get("detail", "").lower()
    matched_certs = resume_data.get("certifications", [])
    matched_projects = resume_data.get("projects", [])
    
    # Formulate PROS prose
    pros_sentences = []
    if matched_skills:
        skills_str = ", ".join(matched_skills[:4])
        pros_sentences.append(f"The candidate demonstrates alignment in key skills, including {skills_str}.")
    if matched_projects:
        project_titles = [p.get('title', '').strip().rstrip('.').lstrip('1234567890. ') for p in matched_projects[:2]]
        project_titles = [f"'{t}'" for t in project_titles if t]
        if project_titles:
            pros_sentences.append(f"Their profile highlights relevant projects such as {', '.join(project_titles)}.")
    if matched_certs:
        certs_str = ", ".join(f"'{c}'" for c in matched_certs[:2])
        pros_sentences.append(f"They also hold professional certifications including {certs_str}.")
    
    pros_text = " ".join(pros_sentences) if pros_sentences else "No significant technical alignment was parsed."

    # Formulate CONS prose
    cons_sentences = []
    if missing_required:
        missing_str = ", ".join(missing_required[:3])
        cons_sentences.append(f"The candidate lacks direct exposure to critical required skills such as {missing_str}.")
    if is_edu_unrelated:
        cons_sentences.append("Their educational background is in a field unrelated to computer science or software engineering.")
    if req_years and res_years < req_years:
        cons_sentences.append(f"There is a professional experience gap of {req_years - res_years:.1f} years relative to the role's requirements.")
    
    cons_text = " ".join(cons_sentences) if cons_sentences else "No critical gaps or mismatch risks were identified."

    summary_lines = [
        f"PROFILE SNAPSHOT: Candidate is a {leveling.split(' (')[0]} professional with {res_years:.1f} years of estimated experience. Highest education level: {resume_data.get('education')[0].get('degree') if resume_data.get('education') else 'Not specified'}.",
        f"PROS (STRENGTHS & ALIGNMENT): {pros_text}",
        f"CONS (CRITICAL GAPS & RISKS): {cons_text}",
        f"VERDICT JUSTIFICATION: Recommended action for the hiring team is {action_verdict} due to the overall compatibility check."
    ]

    return {
        "verdict": verdict,
        "verdict_color": verdict_color,
        "leveling": leveling,
        "summary": "\n\n".join(summary_lines),
        "checklist": checklist,
        "screening_actions": actions
    }


def generate_negatives(
    resume_data: dict[str, Any],
    jd_data: dict[str, Any],
    match_results: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Generate a compact list of gap assessments comparing resume vs job description.
    Focuses strictly on candidate capability and readiness from the hiring team's perspective.
    """
    negatives = []
    is_student = resume_data.get("is_student", False)

    # 1. Missing Required Skills
    missing_required = match_results.get("skills", {}).get("missing_required", [])
    if missing_required:
        skill_list = ", ".join(missing_required[:6])
        more = len(missing_required) - 6
        display = skill_list + (f" (+{more} more)" if more > 0 else "")
        suggestion = (
            f"The candidate has not shown capability in {len(missing_required)} required core skill(s) on their resume. "
            "This lack of exposure to essential technical requirements will require extensive ramp-up time or initial guidance."
        )
        impact = "high"
        negatives.append({
            "category": "Missing Required Skills",
            "missing_item": display,
            "impact": impact,
            "suggestion": suggestion,
        })

    # 2. Experience shortfall
    exp_result = match_results.get("experience", {})
    req_years = exp_result.get("required_years")
    res_years = exp_result.get("resume_years", 0.0)

    if req_years and res_years < req_years:
        shortfall = round(req_years - res_years, 1)
        suggestion = (
            f"The candidate is short by {shortfall} year(s) of the required {req_years:.0f} years of experience. "
            "This experience deficit indicates the candidate may lack the maturity and depth needed to solve complex production issues independently."
        )
        impact = "high" if shortfall >= 2 else "medium"
        negatives.append({
            "category": "Experience Gap",
            "missing_item": f"Short by {shortfall} yr(s) — required {req_years:.0f}, found {res_years:.1f}",
            "impact": impact,
            "suggestion": suggestion,
        })

    # 3. Education shortfall
    edu_result = match_results.get("education", {})
    if edu_result.get("score", 100.0) < 70.0:
        req_levels = jd_data.get("education_requirements", [])
        missing_deg = ", ".join(req_levels) if req_levels else "Bachelors"
        if is_student:
            suggestion = (
                f"The candidate is a current student, and their target degree level ({missing_deg}) is still in progress. "
                "They are not yet fully qualified or available for a standard full-time commitment without an internship/co-op structure."
            )
            impact = "low"
        else:
            suggestion = (
                f"The candidate does not hold the required {missing_deg} degree. "
                "Their academic background is not aligned with the formal qualifications expected for this technical role."
            )
            impact = "medium"
        negatives.append({
            "category": "Education Match",
            "missing_item": f"Credential gap: role requires {missing_deg}",
            "impact": impact,
            "suggestion": suggestion,
        })

    # 4. Missing Preferred Skills
    missing_preferred = match_results.get("skills", {}).get("missing_preferred", [])
    if missing_preferred:
        pref_list = ", ".join(missing_preferred[:5])
        more = len(missing_preferred) - 5
        display = pref_list + (f" (+{more} more)" if more > 0 else "")
        negatives.append({
            "category": "Missing Preferred Skills",
            "missing_item": display,
            "impact": "low",
            "suggestion": (
                "The candidate is missing several preferred nice-to-have skills. While not immediate blockers, "
                "it shows they may not bring the desired advanced/supplementary expertise to the team."
            ),
        })

    # 5. Low Semantic Alignment
    semantic_score = match_results.get("semantic", {}).get("score", 100.0)
    if semantic_score < 50.0:
        negatives.append({
            "category": "Content Alignment",
            "missing_item": "Resume language doesn't closely mirror the job description",
            "impact": "medium",
            "suggestion": (
                "The candidate's resume terminology and overall focus show low semantic similarity to the job description. "
                "This indicates their past work domain, language, or systems do not align well with the team's operational stack."
            ),
        })

    # 6. No Quantifiable Achievements
    resume_text = resume_data.get("_raw_text", "")
    has_numbers = bool(re.search(r"\d+%|\$\d+|\d+\s*(?:users|customers|projects|clients)", resume_text, re.IGNORECASE))
    if not has_numbers:
        negatives.append({
            "category": "Impact Metrics",
            "missing_item": "No measurable achievements or numbers found",
            "impact": "low",
            "suggestion": (
                "The candidate's resume does not contain quantifiable data, metrics, or measurable achievements. "
                "This makes it difficult to assess the scale, scope, and technical impact of their past contributions."
            ),
        })

    return negatives
