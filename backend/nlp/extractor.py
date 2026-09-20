"""Resume and Job Description information extraction using NLP."""

from datetime import datetime
import json
import os
import re
from pathlib import Path
from typing import Any

from nlp.models import get_nlp_model

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
}

def parse_single_date(date_str: str) -> tuple[int, int] | None:
    """Parse a date string into (year, month)."""
    date_str = date_str.strip().lower()
    if not date_str:
        return None
        
    if "present" in date_str or "current" in date_str:
        now = datetime.now()
        return now.year, now.month
        
    # Check for formats like "06/2022", "6/2022", "06-2022", "06.2022" or "6.22"
    slash_match = re.search(r"(\d{1,2})\s*[-./\s]\s*(\d{2,4})", date_str)
    if slash_match:
        yr_val = int(slash_match.group(2))
        if yr_val < 100:
            yr_val = 2000 + yr_val if yr_val < 50 else 1900 + yr_val
        return yr_val, int(slash_match.group(1))
        
    # Check for formats like "2022" (only year)
    year_match = re.search(r"\b(\d{4})\b", date_str)
    if year_match:
        year = int(year_match.group(1))
        # Look for month name in the string
        month = 1
        for m_name, m_val in MONTH_MAP.items():
            if m_name in date_str:
                month = m_val
                break
        return year, month
        
    # Check for formats like "jun 22" or "22 jun" where year is 2 digits
    year_match_2 = re.search(r"\b(\d{2})\b", date_str)
    if year_match_2:
        month = None
        for m_name, m_val in MONTH_MAP.items():
            if m_name in date_str:
                month = m_val
                break
        if month is not None:
            yr_val = int(year_match_2.group(1))
            yr_val = 2000 + yr_val if yr_val < 50 else 1900 + yr_val
            return yr_val, month
            
    return None

def calculate_months_between(start_date: str, end_date: str) -> int:
    """Calculate the number of months between two date strings."""
    start = parse_single_date(start_date)
    end = parse_single_date(end_date)
    if not start or not end:
        return 0
    start_yr, start_mo = start
    end_yr, end_mo = end
    
    months = (end_yr - start_yr) * 12 + (end_mo - start_mo)
    return max(0, months)

def get_nearest_section(text_before: str) -> str | None:
    """Identify the nearest preceding section header."""
    lines = [line.strip().lower() for line in text_before.split("\n") if line.strip()]
    if not lines:
        return None
    # Look at the last few lines before the match
    for line in reversed(lines):
        if len(line) < 30:
            if any(kw in line for kw in ["education", "academic", "qualification", "study"]):
                return "education"
            if any(kw in line for kw in ["experience", "work", "employment", "history", "career"]):
                return "experience"
            if any(kw in line for kw in ["project", "coursework", "capstone", "academic work"]):
                return "projects"
            if any(kw in line for kw in ["activity", "leadership", "extracurricular", "volunteer", "club", "organization", "involvement"]):
                return "activities"
            if any(kw in line for kw in ["certification", "certificate", "award"]):
                return "certifications"
            if "skill" in line:
                return "skills"
    return None


# Load skills database
_skills_db = None


def _get_skills_db() -> dict:
    """Load and cache the skills database."""
    global _skills_db
    if _skills_db is None:
        data_path = Path(__file__).parent.parent / "data" / "skills_database.json"
        with open(data_path, "r", encoding="utf-8") as f:
            _skills_db = json.load(f)
    return _skills_db


def _get_all_skills() -> list[str]:
    """Get a flat list of all skills from the database."""
    db = _get_skills_db()
    all_skills = []
    for category_skills in db.values():
        all_skills.extend(category_skills)
    return all_skills


def _get_skill_categories() -> dict[str, str]:
    """Create a mapping from skill (lowercase) to category."""
    db = _get_skills_db()
    mapping = {}
    for category, skills in db.items():
        for skill in skills:
            mapping[skill.lower()] = category
    return mapping


# ─── Contact Information Extraction ─────────────────────────────────────────

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(
    r"\+?\d{1,4}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{0,4}"
)
LINKEDIN_PATTERN = re.compile(
    r"(?:linkedin\.com/in/|linkedin\.com/profile/view\?id=)[\w\-]+"
)


def extract_contact_info(text: str) -> dict[str, str | None]:
    """Extract email, phone, and LinkedIn from resume text."""
    email_match = EMAIL_PATTERN.search(text)
    phone_match = PHONE_PATTERN.search(text)
    linkedin_match = LINKEDIN_PATTERN.search(text)

    return {
        "email": email_match.group() if email_match else None,
        "phone": phone_match.group() if phone_match else None,
        "linkedin": linkedin_match.group() if linkedin_match else None,
    }


# ─── Name Extraction ────────────────────────────────────────────────────────

def extract_name(text: str) -> str | None:
    """Extract candidate name from the beginning of resume text."""
    nlp = get_nlp_model()

    # Clean the first few lines to remove headers/junk
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines[:8]:
        line_strip = line.strip()
        if not line_strip:
            continue
        # Skip if contains email, phone, or link
        if EMAIL_PATTERN.search(line_strip) or PHONE_PATTERN.search(line_strip) or LINKEDIN_PATTERN.search(line_strip):
            continue
        # Skip if contains common resume section headers/titles
        line_lower = line_strip.lower()
        if any(h in line_lower for h in ["resume", "cv", "curriculum vitae", "portfolio", "page", "contact", "about me", "summary"]):
            continue
        # Skip if too long (names are usually 2-4 words)
        words = line_strip.split()
        if len(words) < 1 or len(words) > 4:
            continue
        # Skip if contains digits (like address or phone fallback)
        if any(c.isdigit() for c in line_strip):
            continue
        # Skip if contains academic or job title keywords that aren't names
        academic_kws = ["student", "candidate", "university", "college", "school", "pursuing", "major", "graduate", "gpa", "bachelor", "master", "phd", "education", "experience"]
        if any(akw in line_lower for akw in academic_kws):
            continue
        cleaned_lines.append(line_strip)

    # First, run NER on the first few cleaned lines
    if cleaned_lines:
        context = "\n".join(cleaned_lines)
        doc = nlp(context)
        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text.split()) <= 4:
                return ent.text.strip()
        
        # If NER fails, use the first cleaned line that starts with uppercase
        for line in cleaned_lines:
            words = line.split()
            if all(w[0].isupper() for w in words if w.isalpha()):
                return line

    # Absolute fallback (original logic)
    for line in text.split("\n")[:3]:
        line = line.strip()
        if line and len(line.split()) <= 4 and not any(c.isdigit() for c in line):
            if not EMAIL_PATTERN.search(line) and not PHONE_PATTERN.search(line):
                # Filter out generic header words
                if not any(h in line.lower() for h in ["resume", "cv", "curriculum vitae"]):
                    return line

    return None


# ─── Skills Extraction ──────────────────────────────────────────────────────

def extract_skills(text: str) -> list[str]:
    """
    Extract skills from text by matching against the skills database.
    Uses case-insensitive matching with word boundary awareness and overlap checks.
    """
    all_skills = _get_all_skills()
    matches_found = []
    text_lower = text.lower()

    for skill in all_skills:
        skill_lower = skill.lower()
        
        # Pattern generation with boundary awareness for alphanumeric and special characters
        if skill_lower[0].isalnum():
            start = r"\b"
        else:
            start = ""
            
        if skill_lower[-1].isalnum():
            end = r"\b"
        elif skill_lower[-1] in ['+', '#']:
            escaped_char = re.escape(skill_lower[-1])
            end = f"(?!{escaped_char}|\\w)"
        else:
            end = ""
            
        pattern = start + re.escape(skill_lower) + end
        
        for m in re.finditer(pattern, text_lower):
            matches_found.append((m.start(), m.end(), skill))

    # Sort matches by length descending, then by start index
    matches_found.sort(key=lambda x: (x[1] - x[0]), reverse=True)

    accepted_spans = []
    accepted_skills = set()

    for start, end, skill in matches_found:
        # Check if this span overlaps with any already accepted (longer) span
        overlap = False
        for s_start, s_end in accepted_spans:
            if s_start <= start < s_end or s_start < end <= s_end:
                overlap = True
                break
        if not overlap:
            accepted_spans.append((start, end))
            accepted_skills.add(skill)

    return sorted(list(accepted_skills))


def categorize_skills(skills: list[str]) -> dict[str, list[str]]:
    """Organize skills by their category."""
    category_map = _get_skill_categories()
    categorized = {}

    for skill in skills:
        category = category_map.get(skill.lower(), "other")
        if category not in categorized:
            categorized[category] = []
        categorized[category].append(skill)

    return categorized


# ─── Education Extraction ───────────────────────────────────────────────────

DEGREE_PATTERNS = [
    (r"\b(?:Ph\.?D\.?|Doctor(?:ate)?)(?:\b|\.)(?:\s+(?:of|in|-|:)?\s+|\s+)\w[\w\s,]+", "PhD"),
    (r"\b(?:Master(?:s|'s)?|M\.?S\.?|M\.?A\.?|M\.?Tech\.?|M\.?Eng\.?|MBA|M\.?C\.?A\.?)(?:\b|\.)(?:\s+(?:of|in|-|:)?\s+|\s+)\w[\w\s,]+", "Masters"),
    (r"\b(?:Master(?:s|'s)?|M\.?S\.?|M\.?A\.?|M\.?Tech\.?|M\.?Eng\.?|MBA|M\.?C\.?A\.?)(?:\b|\.)(?:\s|$|,)", "Masters"),
    (r"\b(?:Bachelor(?:s|'s)?|B\.?S\.?|B\.?A\.?|B\.?Tech\.?|B\.?Eng\.?|B\.?Sc\.?|B\.?E\.?|B\.?C\.?A\.?|B\.?B\.?A\.?)(?:\b|\.)(?:\s+(?:of|in|-|:)?\s+|\s+)\w[\w\s,]+", "Bachelors"),
    (r"\b(?:Bachelor(?:s|'s)?|B\.?S\.?|B\.?A\.?|B\.?Tech\.?|B\.?Eng\.?|B\.?Sc\.?|B\.?E\.?|B\.?C\.?A\.?|B\.?B\.?A\.?)(?:\b|\.)(?:\s|$|,)", "Bachelors"),
    (r"\b(?:Associate(?:s|'s)?|A\.?S\.?|A\.?A\.?)(?:\b|\.)\s+(?:of|in)\s+\w[\w\s,]+", "Associates"),
    (r"\b(?:Diploma|Certificate)\b\s+(?:of|in)\s+\w[\w\s,]+", "Diploma"),
]

YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")


def extract_education(text: str) -> list[dict[str, str | None]]:
    """Extract education entries from resume text."""
    nlp = get_nlp_model()
    education_entries = []
    matched_spans = []

    for pattern, degree_level in DEGREE_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            span_start = match.start()
            span_end = match.end()

            # Skip matches starting with lowercase/titlecase "be" (like "be responsible", "be admitted")
            # to prevent the common English verb "be" from being parsed as a Bachelor of Engineering (B.E.).
            match_str = match.group().strip()
            if match_str.lower().startswith("be") and not match_str.startswith("BE"):
                continue

            # Check overlap with already matched degrees to prevent duplicates/sub-parts
            if any(s_start <= span_start < s_end or s_start < span_end <= s_end for s_start, s_end in matched_spans):
                continue

            matched_spans.append((span_start, span_end))

            # Get surrounding context for institution and year
            start = max(0, span_start - 120)
            end = min(len(text), span_end + 120)
            context = text[start:end]

            # Extract institution using NER
            doc = nlp(context)
            org_entities = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]

            # Filter out degree/junk names
            filtered_orgs = []
            for org in org_entities:
                if not any(x in org.lower() for x in ["bachelor", "master", "science", "arts", "engineering", "technology", "phd", "candidate"]):
                    filtered_orgs.append(org)

            institution = None
            school_kws = ["university", "college", "school", "institute", "academy", "vit", "iit", "nit", "bits", "iiit", "vellore"]

            # 1. Try to find an ORG entity that contains school keywords
            for org in filtered_orgs:
                if any(kw in org.lower() for kw in school_kws):
                    institution = org
                    break

            # 2. Fallback to regex matching for school keywords in context lines
            if not institution:
                lines = context.split("\n")
                for line in lines:
                    line_strip = line.strip()
                    line_lower = line_strip.lower()
                    if any(kw in line_lower for kw in ["university", "college", "school of", "institute of", "vit ", "vit university", "vit vellore"]):
                        if len(line_strip) < 80:
                            institution = line_strip
                            break

            # 3. Fallback to the first filtered ORG entity
            if not institution and filtered_orgs:
                institution = filtered_orgs[0]

            # Extract year
            years = YEAR_PATTERN.findall(context)
            year = max(years) if years else None

            entry = {
                "degree": match.group().strip(),
                "level": degree_level,
                "institution": institution,
                "year": year,
            }

            # Avoid duplicates
            if not any(
                e["degree"].lower() == entry["degree"].lower()
                for e in education_entries
            ):
                education_entries.append(entry)

    return education_entries


def get_education_level_score(education: list[dict]) -> int:
    """
    Assign a numeric score to education level for comparison.
    PhD=5, Masters=4, Bachelors=3, Associates=2, Diploma=1, None=0
    """
    level_scores = {
        "PhD": 5,
        "Masters": 4,
        "Bachelors": 3,
        "Associates": 2,
        "Diploma": 1,
    }
    if not education:
        return 0
    max_score = 0
    for edu in education:
        score = level_scores.get(edu.get("level", ""), 0)
        max_score = max(max_score, score)
    return max_score


# ─── Experience Extraction ──────────────────────────────────────────────────

EXPERIENCE_YEAR_PATTERN = re.compile(
    r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)?",
    re.IGNORECASE,
)

DATE_RANGE_PATTERN = re.compile(
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?,?\s+(?:\d{4}|\d{2})\b|"
    r"\b\d{1,2}[-./\s]+(?:\d{4}|\d{2})\b|"
    r"\b\d{4}\b)\s*[-–—to]+\s*"
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?,?\s+(?:\d{4}|\d{2})\b|"
    r"\b\d{1,2}[-./\s]+(?:\d{4}|\d{2})\b|"
    r"\b\d{4}\b|"
    r"\b[Pp]resent\b|\b[Cc]urrent\b)",
    re.IGNORECASE,
)

JOB_TITLE_KEYWORDS = [
    "engineer", "developer", "architect", "designer", "manager",
    "analyst", "consultant", "specialist", "coordinator", "director",
    "lead", "senior", "junior", "intern", "associate", "principal",
    "chief", "head", "vp", "president", "officer", "administrator",
    "scientist", "researcher", "technician", "strategist", "planner",
]


def extract_experience(text: str) -> list[dict[str, str | None]]:
    """Extract work experience entries from resume text."""
    nlp = get_nlp_model()
    experiences = []

    # Find date ranges to anchor experience entries
    date_matches = list(DATE_RANGE_PATTERN.finditer(text))

    for i, match in enumerate(date_matches):
        # Determine the nearest section header before this date range
        start_pos = max(0, match.start() - 400)
        text_before = text[start_pos:match.start()]
        section = get_nearest_section(text_before)

        # Skip if nearest section is not experience-related (e.g. projects, activities, education)
        if section in ["education", "projects", "activities", "skills", "certifications"]:
            continue

        # Check context for education indicators
        check_start = max(0, match.start() - 150)
        check_context = text[check_start:match.end()].lower()
        
        is_edu = False
        if section == "education":
            is_edu = True
        else:
            edu_kws = ["bachelor", "master", "ph.d", "degree", "gpa", "b.s.", "m.s.", "b.s", "m.s", "btech", "mtech", "b.e.", "b.e", "bca", "mca", "university", "college", "school", "high school"]
            keep_kws = ["intern", "assistant", "tutor", "fellow", "worker", "instructor"]
            if any(kw in check_context for kw in edu_kws):
                if not any(kw in check_context for kw in keep_kws):
                    is_edu = True
        
        if is_edu:
            continue

        # Get context around the date range
        prev_match_end = date_matches[i - 1].end() if i > 0 else 0
        start = max(prev_match_end, match.start() - 200)
        end = min(
            len(text),
            date_matches[i + 1].start() if i + 1 < len(date_matches) else match.end() + 300,
        )
        context = text[start:end]

        # Extract company using NER
        doc = nlp(context)
        company = None
        title = None

        for ent in doc.ents:
            if ent.label_ == "ORG" and not company:
                ent_text = ent.text.strip()
                ent_lower = ent_text.lower()
                if not any(x in ent_lower for x in ["university", "college", "school", "bachelor", "master", "january", "february", "march", "april", "june", "july", "august", "september", "october", "november", "december"]):
                    company = ent_text

        # Try to find job title
        lines = context.split("\n")
        for line in lines:
            line_lower = line.lower().strip()
            if any(kw in line_lower for kw in JOB_TITLE_KEYWORDS):
                if len(line.strip()) < 80:  # Job titles are usually short
                    title = line.strip()
                    break

        duration = f"{match.group(1)} - {match.group(2)}"

        # Get description (text after the date range)
        desc_start = match.end()
        desc_end = min(
            len(text),
            date_matches[i + 1].start() if i + 1 < len(date_matches) else match.end() + 500,
        )
        description = text[desc_start:desc_end].strip()[:300]

        # Clean up description to remove junk lines or section headers
        desc_lines = description.split("\n")
        cleaned_desc_lines = []
        for d_line in desc_lines:
            d_line_strip = d_line.strip()
            if not d_line_strip:
                continue
            if d_line_strip.lower() in ["education", "skills", "projects", "certifications", "languages", "interests", "awards"]:
                break
            cleaned_desc_lines.append(d_line_strip)
        description = " ".join(cleaned_desc_lines)[:300]

        experiences.append({
            "title": title or "Experience Entry",
            "company": company or "Various Projects / Freelance",
            "duration": duration,
            "description": description,
            "months": calculate_months_between(match.group(1), match.group(2))
        })

    return experiences


def estimate_total_experience_years(text: str) -> float:
    """Estimate total years of experience from resume text, strictly excluding internships."""
    # Extract work experience entries first
    experiences = extract_experience(text)
    if experiences:
        intern_keywords = ["intern", "internship", "trainee", "apprentice", "co-op", "coop"]
        non_intern_months = 0
        for exp in experiences:
            title = exp.get("title", "").lower()
            desc = exp.get("description", "").lower()
            company = exp.get("company", "").lower()
            
            is_intern = any(kw in title for kw in intern_keywords) or \
                        any(kw in desc for kw in intern_keywords) or \
                        any(kw in company for kw in intern_keywords)
            if not is_intern:
                non_intern_months += exp.get("months", 0)
                
        if non_intern_months > 0:
            return round(non_intern_months / 12.0, 1)

    # Fallback to explicit mentions if no experiences were parsed with durations
    explicit = []
    for match in EXPERIENCE_YEAR_PATTERN.finditer(text):
        # Filter out academic mentions like "5 Year Program"
        match_end = match.end()
        post_text = text[match_end:match_end+20].lower().strip()
        if any(post_text.startswith(w) for w in ["program", "degree", "course", "curriculum", "duration"]):
            continue

        match_start = match.start()
        pre_text = text[max(0, match_start-30):match_start].lower().strip()
        if any(w in pre_text for w in ["program", "degree", "course", "integrated"]):
            continue

        # Filter out mentions containing "internship" or "intern" in the surrounding text
        context_start = max(0, match_start-40)
        context_end = min(len(text), match_end+40)
        context_snippet = text[context_start:context_end].lower()
        if any(w in context_snippet for w in ["intern", "internship", "trainee", "apprentice"]):
            continue

        explicit.append(float(match.group(1)))

    if explicit:
        return max(explicit)

    return 0.0


# ─── Certification Extraction ───────────────────────────────────────────────

def extract_certifications(text: str) -> list[str]:
    """Extract certifications from resume text with word boundary checks for short acronyms."""
    db = _get_skills_db()
    cert_keywords = db.get("certifications_keywords", [])
    found = []
    text_lower = text.lower()

    for cert in cert_keywords:
        cert_lower = cert.lower()
        if len(cert_lower) <= 5:  # e.g., CKA, CKAD, CSM, PMP, CCNA, CCNP, ITIL
            pattern = r"\b" + re.escape(cert_lower) + r"\b"
            if re.search(pattern, text_lower):
                found.append(cert)
        else:
            if cert_lower in text_lower:
                found.append(cert)

    return found


def extract_projects(text: str) -> list[dict[str, Any]]:
    """Extract personal or academic projects from the resume."""
    lines = text.split("\n")
    project_indices = []
    
    project_headers = ["project", "academic work", "hackathon", "capstone", "academic project", "personal project"]
    start_idx = -1
    end_idx = len(lines)
    
    other_headers = ["experience", "work", "employment", "education", "skills", "certifications", "volunteer", "activities", "languages", "interests"]
    
    for i, line in enumerate(lines):
        line_clean = line.strip().lower().replace("#", "").replace(":", "").strip()
        if not line_clean:
            continue
        if any(h == line_clean or line_clean.startswith(h + "s") for h in project_headers) and len(line_clean) < 30:
            start_idx = i
            break
            
    if start_idx == -1:
        return []
        
    for i in range(start_idx + 1, len(lines)):
        line_clean = lines[i].strip().lower().replace("#", "").replace(":", "").strip()
        if not line_clean:
            continue
        if any(oh in line_clean or line_clean.startswith(oh + "s") for oh in other_headers) and len(line_clean) < 30:
            end_idx = i
            break
            
    project_lines = lines[start_idx+1 : end_idx]
    
    projects = []
    current_project = None
    
    for line in project_lines:
        line_strip = line.strip()
        if not line_strip:
            continue
            
        is_bullet = line_strip.startswith(("-", "*", "•", "o ", "+"))
        
        if not is_bullet and len(line_strip) < 80 and not any(line_strip.lower().startswith(x) for x in ["technologies", "tech stack", "tools used"]):
            if current_project:
                projects.append(current_project)
            
            title = line_strip.replace("|", "").strip()
            current_project = {
                "title": title,
                "description": "",
                "technologies": []
            }
        elif current_project:
            if any(term in line_strip.lower() for term in ["technology", "technologies", "tech stack", "tools", "built with", "used:"]):
                skills = extract_skills(line_strip)
                current_project["technologies"].extend(skills)
            else:
                clean_line = line_strip.lstrip("-*•o+ ").strip()
                if current_project["description"]:
                    current_project["description"] += " " + clean_line
                else:
                    current_project["description"] = clean_line
                    
                skills = extract_skills(clean_line)
                current_project["technologies"].extend(skills)
                
    if current_project:
        projects.append(current_project)
        
    for p in projects:
        p["technologies"] = sorted(list(set(p["technologies"])))
        p["description"] = p["description"][:500]
        
    return [p for p in projects if len(p["title"]) > 3]


def detect_is_student(text: str, experience_years: float, education_entries: list[dict]) -> bool:
    """Detect if the candidate is currently a student or a recent fresh graduate."""
    text_lower = text.lower()
    
    student_keywords = [
        "current student", "pursuing bsc", "pursuing ms", "pursuing btech",
        "student at", "candidate for", "expected graduation", "graduation date",
        "undergraduate", "fresh graduate", "class of 2025", "class of 2026", "class of 2027", "class of 2028", "class of 2024",
        "current year", "pursuing", "ongoing", "in progress", "in-progress",
        "1st year", "2nd year", "3rd year", "4th year", "5th year",
        "first year", "second year", "third year", "fourth year", "fifth year",
        "expected 20"
    ]
    if any(kw in text_lower for kw in student_keywords):
        return True
        
    current_year = datetime.now().year
    for edu in education_entries:
        yr_str = edu.get("year")
        if yr_str and yr_str.isdigit():
            yr = int(yr_str)
            if yr >= current_year - 1:
                if experience_years <= 1.5:
                    return True
                    
    if experience_years <= 1.0:
        general_student_terms = ["student", "internship", "university", "college", "school"]
        term_count = sum(1 for term in general_student_terms if term in text_lower)
        if term_count >= 3:
            return True
            
    return False


# ─── Main Extraction Functions ──────────────────────────────────────────────

def extract_resume_data(text: str) -> dict[str, Any]:
    """
    Extract all structured data from resume text.

    Returns a dictionary containing name, contact info, skills,
    education, experience, certifications, and estimated years.
    """
    contact = extract_contact_info(text)
    name = extract_name(text)
    skills = extract_skills(text)
    education = extract_education(text)
    experience = extract_experience(text)
    projects = extract_projects(text)
    certifications = extract_certifications(text)
    exp_years = estimate_total_experience_years(text)
    
    is_student = detect_is_student(text, exp_years, education)

    return {
        "name": name,
        "email": contact["email"],
        "phone": contact["phone"],
        "linkedin": contact["linkedin"],
        "skills": skills,
        "skills_categorized": categorize_skills(skills),
        "education": education,
        "experience": experience,
        "projects": projects,
        "certifications": certifications,
        "total_experience_years": exp_years,
        "is_student": is_student,
    }


def extract_job_description_data(text: str) -> dict[str, Any]:
    """
    Extract structured data from job description text.

    Returns a dictionary containing extracted skills (required/preferred),
    experience requirements, and education requirements.
    """
    all_skills = extract_skills(text)
    text_lower = text.lower()

    # Try to separate required vs preferred skills
    required_skills = []
    preferred_skills = []

    # Look for sections that indicate required vs preferred
    required_section = ""
    preferred_section = ""

    # Split by common section indicators
    sections = re.split(
        r"(?:required|must have|minimum|essential|qualifications)",
        text,
        flags=re.IGNORECASE,
    )
    pref_sections = re.split(
        r"(?:preferred|nice to have|bonus|desired|plus)",
        text,
        flags=re.IGNORECASE,
    )

    if len(pref_sections) > 1:
        preferred_section = pref_sections[-1][:500]
        preferred_skills = extract_skills(preferred_section)
        required_skills = [s for s in all_skills if s not in preferred_skills]
    else:
        required_skills = all_skills
        preferred_skills = []

    # Extract experience requirement
    # Check for range first, e.g., "3-6 years", "3 to 6 years"
    range_match = re.search(r"(\d+)\s*(?:-|–|—|to)\s*(\d+)\s*(?:years?|yrs?)\b", text, re.IGNORECASE)
    if range_match:
        min_experience = float(range_match.group(1))
    else:
        exp_match = EXPERIENCE_YEAR_PATTERN.search(text)
        min_experience = float(exp_match.group(1)) if exp_match else None

    # Extract education requirement
    education_req = []
    for pattern, level in DEGREE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            education_req.append(level)

    # Extract job title (usually in the first few lines)
    first_lines = text.split("\n")[:5]
    title = None
    for line in first_lines:
        line = line.strip()
        if line and len(line) < 100:
            title = line
            break

    return {
        "title": title,
        "all_skills": all_skills,
        "required_skills": required_skills if required_skills else all_skills,
        "preferred_skills": preferred_skills,
        "min_experience_years": min_experience,
        "education_requirements": list(set(education_req)),
    }
