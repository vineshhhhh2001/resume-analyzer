"""Resume-to-Job Description matching algorithm using NLP."""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from nlp.models import get_sentence_model
from nlp.extractor import get_education_level_score


def compute_skills_match(
    resume_skills: list[str],
    required_skills: list[str],
    preferred_skills: list[str] | None = None,
) -> dict:
    """
    Compute skills match score between resume and job description.

    Uses Jaccard-like similarity weighted by required vs preferred skills.
    Supports alias and synonym normalization to avoid mismatches on minor casing/naming formats.

    Returns:
        Dictionary with score (0-100), matched, missing, and extra skills.
    """
    def normalize(name: str) -> str:
        s = name.strip().lower()
        aliases = {
            "js": "javascript",
            "javascript": "javascript",
            "react.js": "react",
            "reactjs": "react",
            "vue.js": "vue",
            "vuejs": "vue",
            "angular.js": "angular",
            "angularjs": "angular",
            "node.js": "node",
            "nodejs": "node",
            "express.js": "express",
            "expressjs": "express",
            "next.js": "next",
            "nextjs": "next",
            "nuxt.js": "nuxt",
            "nuxtjs": "nuxt",
            "three.js": "three",
            "threejs": "three",
            "amazon web services": "aws",
            "google cloud platform": "gcp",
            "google cloud": "gcp",
            "microsoft azure": "azure",
            "postgres": "postgresql",
            "postgres sql": "postgresql",
            "spring boot": "spring",
            "git/github": "git",
            "github": "git",
            "gitlab": "git",
            "scikit-learn": "scikit-learn",
            "scikitlearn": "scikit-learn",
            "tensorflow": "tensorflow",
            "tf": "tensorflow",
            "pytorch": "pytorch",
            "nlp": "natural language processing",
            "llm": "large language models",
            "generative ai": "genai",
            "gen ai": "genai",
        }
        if s in aliases:
            return aliases[s]
        
        # Strip .js or js suffix safely
        if s.endswith(".js"):
            s = s[:-3]
        elif s.endswith("js") and len(s) > 2:
            s = s[:-2]
            
        s_clean = s.replace(" ", "").replace("-", "").replace("_", "")
        return s_clean

    # Map normalized names to their original casings (favor shortest original string)
    skill_case_map = {}
    for s in resume_skills + required_skills + (preferred_skills or []):
        norm = normalize(s)
        if norm not in skill_case_map or len(s) < len(skill_case_map[norm]):
            skill_case_map[norm] = s

    resume_norm = {normalize(s) for s in resume_skills}
    required_norm = {normalize(s) for s in required_skills}
    preferred_norm = {normalize(s) for s in (preferred_skills or [])}
    all_jd_norm = required_norm | preferred_norm

    # Matched skills
    matched_required_norm = resume_norm & required_norm
    matched_preferred_norm = resume_norm & preferred_norm
    matched_all_norm = resume_norm & all_jd_norm

    # Missing skills
    missing_required_norm = required_norm - resume_norm
    missing_preferred_norm = preferred_norm - resume_norm

    # Extra skills
    extra_norm = resume_norm - all_jd_norm

    # Score: 80% weight on required skills, 20% on preferred
    # Semantic Matching for missing skills
    semantic_matches = {}
    if (missing_required_norm or missing_preferred_norm) and resume_skills:
        try:
            from nlp.extractor import _get_skill_categories
            category_map = _get_skill_categories()
            
            model = get_sentence_model()
            cand_skills_list = list(resume_skills)
            cand_embeddings = model.encode(cand_skills_list, show_progress_bar=False, normalize_embeddings=True)
            
            missing_to_orig = {}
            for s in required_skills + (preferred_skills or []):
                norm = normalize(s)
                if norm in missing_required_norm or norm in missing_preferred_norm:
                    missing_to_orig[norm] = s
            
            missing_norms = list(missing_to_orig.keys())
            missing_origs = [missing_to_orig[n] for n in missing_norms]
            
            if missing_origs:
                missing_embeddings = model.encode(missing_origs, show_progress_bar=False, normalize_embeddings=True)
                similarities = np.dot(missing_embeddings, cand_embeddings.T)
                
                for idx, norm in enumerate(missing_norms):
                    max_sim_idx = int(np.argmax(similarities[idx]))
                    max_sim = float(similarities[idx][max_sim_idx])
                    
                    matched_cand_skill = cand_skills_list[max_sim_idx]
                    matched_cand_norm = normalize(matched_cand_skill)
                    
                    cat1 = category_map.get(norm, "other")
                    cat2 = category_map.get(matched_cand_norm, "other2")
                    same_category = (cat1 != "other" and cat1 == cat2)
                    
                    required_sim = 0.40 if same_category else 0.55
                    if max_sim >= required_sim:
                        semantic_matches[norm] = (matched_cand_norm, max_sim)
        except Exception as e:
            print(f"[WARN] Semantic skill matching failed: {e}")

    # Load skill categories
    try:
        from nlp.extractor import _get_skill_categories
        category_map = _get_skill_categories()
    except Exception:
        category_map = {}

    # Separate required and preferred norms into technical vs soft
    tech_required = {s for s in required_norm if category_map.get(s, "other") != "soft_skills"}
    soft_required = {s for s in required_norm if category_map.get(s, "other") == "soft_skills"}

    if required_norm:
        # Tech required
        if tech_required:
            tech_points = 0.0
            for r_norm in tech_required:
                if r_norm in resume_norm:
                    tech_points += 1.0
                elif r_norm in semantic_matches:
                    sim = semantic_matches[r_norm][1]
                    partial_credit = 0.5 + max(0.0, (sim - 0.45) * 0.6)
                    tech_points += min(0.8, partial_credit)
            tech_req_score = (tech_points / len(tech_required)) * 100
        else:
            tech_req_score = 100.0

        # Soft required
        if soft_required:
            soft_points = 0.0
            for r_norm in soft_required:
                if r_norm in resume_norm:
                    soft_points += 1.0
                elif r_norm in semantic_matches:
                    sim = semantic_matches[r_norm][1]
                    partial_credit = 0.5 + max(0.0, (sim - 0.45) * 0.6)
                    soft_points += min(0.8, partial_credit)
            soft_req_score = (soft_points / len(soft_required)) * 100
        else:
            soft_req_score = 100.0

        # Weighted required score: 85% technical, 15% soft (renormalized if one is empty)
        if tech_required and soft_required:
            required_score = tech_req_score * 0.85 + soft_req_score * 0.15
        elif tech_required:
            required_score = tech_req_score
        else:
            required_score = soft_req_score
    else:
        required_score = 100.0

    # Preferred norms
    tech_preferred = {s for s in preferred_norm if category_map.get(s, "other") != "soft_skills"}
    soft_preferred = {s for s in preferred_norm if category_map.get(s, "other") == "soft_skills"}

    if preferred_norm:
        # Tech preferred
        if tech_preferred:
            tech_points = 0.0
            for p_norm in tech_preferred:
                if p_norm in resume_norm:
                    tech_points += 1.0
                elif p_norm in semantic_matches:
                    sim = semantic_matches[p_norm][1]
                    partial_credit = 0.5 + max(0.0, (sim - 0.45) * 0.6)
                    tech_points += min(0.8, partial_credit)
            tech_pref_score = (tech_points / len(tech_preferred)) * 100
        else:
            tech_pref_score = 100.0

        # Soft preferred
        if soft_preferred:
            soft_points = 0.0
            for p_norm in soft_preferred:
                if p_norm in resume_norm:
                    soft_points += 1.0
                elif p_norm in semantic_matches:
                    sim = semantic_matches[p_norm][1]
                    partial_credit = 0.5 + max(0.0, (sim - 0.45) * 0.6)
                    soft_points += min(0.8, partial_credit)
            soft_pref_score = (soft_points / len(soft_preferred)) * 100
        else:
            soft_pref_score = 100.0

        # Weighted preferred score: 85% technical, 15% soft (renormalized if one is empty)
        if tech_preferred and soft_preferred:
            preferred_score = tech_pref_score * 0.85 + soft_pref_score * 0.15
        elif tech_preferred:
            preferred_score = tech_pref_score
        else:
            preferred_score = soft_pref_score
    else:
        preferred_score = 100.0

    # Combine required and preferred scores, or just use required if preferred is empty
    if preferred_norm:
        score = required_score * 0.8 + preferred_score * 0.2
    else:
        score = required_score

    # If the candidate matches absolutely zero skills (both exact and semantic),
    # and the job description actually has required skills,
    # their skills match score should be 0.0.
    if required_norm and not matched_all_norm and not semantic_matches:
        score = 0.0

    return {
        "score": round(min(score, 100.0), 1),
        "matched_skills": sorted([skill_case_map.get(s, s) for s in matched_all_norm]),
        "missing_required": sorted([skill_case_map.get(s, s) for s in missing_required_norm]),
        "missing_preferred": sorted([skill_case_map.get(s, s) for s in missing_preferred_norm]),
        "extra_skills": sorted([skill_case_map.get(s, s) for s in extra_norm]),
        "semantic_matches": [
            {
                "missing": skill_case_map.get(m_norm, m_norm),
                "matched_with": skill_case_map.get(c_norm, c_norm),
                "similarity": round(sim, 2)
            }
            for m_norm, (c_norm, sim) in semantic_matches.items()
        ]
    }


def compute_experience_match(
    resume_years: float,
    required_years: float | None,
    is_entry_level: bool = False,
    projects: list[dict] = None,
    is_student: bool = False,
) -> dict:
    """
    Compute experience match score.

    Returns:
        Dictionary with score (0-100) and details.
    """
    if required_years is None or required_years == 0:
        return {
            "score": 100.0,
            "resume_years": resume_years,
            "required_years": None,
            "detail": "No specific experience requirement found in job description.",
        }

    # Project boosting only for entry level or internship roles
    adjusted_resume_years = resume_years
    boost_applied = 0.0
    project_count = len(projects) if projects else 0

    if is_entry_level:
        if project_count > 0:
            boost_applied = min(project_count * 0.5, 1.5)
            adjusted_resume_years += boost_applied

    if adjusted_resume_years >= required_years:
        score = 100.0
        if boost_applied > 0:
            detail = f"Meets requirement with project boost: {resume_years:.1f} yrs experience + {boost_applied:.1f} yrs from {project_count} projects (required: {required_years:.1f})"
        else:
            detail = f"Exceeds requirement: {resume_years:.0f} years (required: {required_years:.0f})"
    elif adjusted_resume_years >= required_years * 0.7:
        ratio = adjusted_resume_years / required_years
        score = ratio * 100
        if boost_applied > 0:
            detail = f"Close to requirement with project boost: {resume_years:.1f} yrs experience + {boost_applied:.1f} yrs from projects (required: {required_years:.1f})"
        else:
            detail = f"Close to requirement: {resume_years:.0f} years (required: {required_years:.0f})"
    else:
        ratio = adjusted_resume_years / required_years
        score = max(ratio * 100, 10.0)
        if is_entry_level and project_count > 0:
            score = max(score, 75.0)
            detail = f"Under years requirement but considered close due to academic/personal projects ({project_count} projects found)."
        else:
            detail = f"Below requirement: {resume_years:.0f} years (required: {required_years:.0f})"

    # If the job is entry-level/internship and candidate is student or early-career, set minimum score
    if is_entry_level and (is_student or resume_years > 0 or project_count > 0):
        if required_years <= 2.0 and score < 85.0:
            score = 85.0
            detail = "Early-career candidate or student meets baseline experience (up to 2 years required) through projects/work history."
        elif required_years <= 3.0 and score < 80.0:
            score = 80.0
            detail = "Early-career candidate or student meets baseline experience (up to 3 years required) through projects/work history."

    return {
        "score": round(min(score, 100.0), 1),
        "resume_years": resume_years,
        "required_years": required_years,
        "detail": detail,
    }


def compute_education_match(
    resume_education: list[dict],
    jd_education_requirements: list[str],
    is_student: bool = False,
    is_entry_level: bool = False,
) -> dict:
    """
    Compute education match score.

    Returns:
        Dictionary with score (0-100) and details.
    """
    level_scores = {
        "PhD": 5,
        "Masters": 4,
        "Bachelors": 3,
        "Associates": 2,
        "Diploma": 1,
    }

    # Determine candidate's resume education level score
    resume_level = get_education_level_score(resume_education)

    # Determine candidate's completed level by evaluating each education entry.
    # If a degree is in progress:
    # - Integrated courses (e.g. B.Tech + M.Tech) mean they have completed NO degree yet.
    # - Standard courses mean they completed the next lower degree level (e.g. Bachelors if pursuing Masters).
    completed_level = 0
    if resume_education:
        highest_level_val = 0
        for edu in resume_education:
            lvl = edu.get("level", "")
            highest_level_val = max(highest_level_val, level_scores.get(lvl, 0))

        for edu in resume_education:
            level_str = edu.get("level", "")
            level_val = level_scores.get(level_str, 0)
            if level_val == 0:
                continue

            edu_in_progress = False
            degree_text = edu.get("degree", "").lower()
            inst_text = edu.get("institution", "")
            inst_text = inst_text.lower() if inst_text else ""
            yr_str = edu.get("year")
            yr_str = yr_str.lower() if yr_str else ""

            in_progress_terms = ["current", "pursuing", "ongoing", "expected", "in progress", "in-progress", "studying", "present"]
            
            from datetime import datetime
            current_year = datetime.now().year

            if any(term in degree_text for term in in_progress_terms) or \
               any(term in inst_text for term in in_progress_terms) or \
               any(term in yr_str for term in in_progress_terms):
                edu_in_progress = True
            elif yr_str and yr_str.isdigit() and int(yr_str) > current_year:
                edu_in_progress = True
            elif is_student and level_val == highest_level_val:
                if yr_str and yr_str.isdigit() and int(yr_str) <= current_year:
                    edu_in_progress = False
                else:
                    edu_in_progress = True

            if edu_in_progress:
                integrated_keywords = [
                    "integrated", "dual degree", "dual-degree", "btech+mtech", 
                    "b.tech+m.tech", "btech + mtech", "b.tech + m.tech",
                    "b.e. + m.tech", "b.e+m.tech", "b.e. + m.s.", "btech/mtech",
                    "b.tech/m.tech", "b.e./m.tech"
                ]
                is_integrated = any(kw in degree_text for kw in integrated_keywords)
                
                if is_integrated:
                    entry_completed_level = 0
                else:
                    entry_completed_level = max(0, level_val - 1)
            else:
                entry_completed_level = level_val

            completed_level = max(completed_level, entry_completed_level)
    else:
        completed_level = resume_level

    # Determine highest required education level
    if not jd_education_requirements:
        # If the job doesn't specify education requirements
        if is_student and not is_entry_level:
            # A current student applying to a full-time mid-level role has an education/availability gap
            score = 0.0
            detail = "No specific education requirement in JD, but candidate is a current student applying to a full-time role."
        else:
            score = 100.0
            detail = "No specific education requirement found in job description."
            
        return {
            "score": score,
            "resume_level": resume_level,
            "required_level": None,
            "detail": detail,
        }

    required_level = max(
        (level_scores.get(req, 0) for req in jd_education_requirements),
        default=0,
    )

    # Validate major/field of study alignment
    is_major_relevant = True
    if resume_education:
        tech_degree_keywords = [
            "computer", "software", "engineering", "information technology",
            "science", "math", "systems", "application", "it", "cs", "cse",
            "mca", "bca", "technology", "programming", "developer",
            "tech", "btech", "mtech", "b.tech", "m.tech", "b.e", "b.e."
        ]
        highest_entry = None
        highest_score = -1
        for edu in resume_education:
            score = level_scores.get(edu.get("level", ""), 0)
            if score > highest_score:
                highest_score = score
                highest_entry = edu
        if highest_entry:
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

    if completed_level >= required_level:
        if is_major_relevant:
            score = 100.0
            detail = "Completed education level meets or exceeds requirement and is in a relevant field."
        else:
            score = 0.0
            detail = "Completed education level meets level requirement, but degree field appears unrelated to the role."
    else:
        score = 0.0
        detail = "Education requirement not met (required degree not completed/still studying)."

    return {
        "score": round(min(score, 100.0), 1),
        "resume_level": resume_level,
        "required_level": required_level,
        "detail": detail,
    }


def compute_semantic_similarity(
    resume_text: str,
    jd_text: str,
) -> dict:
    """
    Compute semantic similarity between resume and job description
    using Sentence Transformer embeddings + cosine similarity.

    Returns:
        Dictionary with score (0-100) and raw similarity.
    """
    model = get_sentence_model()

    # Truncate texts to avoid memory issues (model max is 256 tokens)
    resume_truncated = resume_text[:3000]
    jd_truncated = jd_text[:2000]

    # Encode both texts
    embeddings = model.encode(
        [resume_truncated, jd_truncated],
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    # Compute cosine similarity
    similarity = cosine_similarity(
        embeddings[0].reshape(1, -1),
        embeddings[1].reshape(1, -1),
    )[0][0]

    # Scale to 0-100 (cosine similarity ranges from -1 to 1, usually 0 to 1 for text)
    # Typical text similarity ranges from 0.2 to 0.8, so we scale accordingly
    score = max(0, min(100, (similarity - 0.1) / 0.7 * 100))

    return {
        "score": round(score, 1),
        "raw_similarity": round(float(similarity), 4),
    }
