"""Tests for the NLP extractor module."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nlp.extractor import (
    extract_skills,
    extract_contact_info,
    extract_resume_data,
    extract_job_description_data,
    categorize_skills,
)


class TestExtractSkills:
    """Tests for skill extraction from text."""

    def test_extracts_programming_languages(self):
        text = "Proficient in Python, Java, and JavaScript development"
        skills = extract_skills(text)
        assert "Python" in skills
        assert "Java" in skills
        assert "JavaScript" in skills

    def test_extracts_frameworks(self):
        text = "Built applications using React, Django, and FastAPI"
        skills = extract_skills(text)
        assert "React" in skills
        assert "Django" in skills
        assert "FastAPI" in skills

    def test_extracts_databases(self):
        text = "Experience with PostgreSQL, MongoDB, and Redis"
        skills = extract_skills(text)
        assert "PostgreSQL" in skills
        assert "MongoDB" in skills
        assert "Redis" in skills

    def test_extracts_cloud_platforms(self):
        text = "Deployed services on AWS and Google Cloud Platform"
        skills = extract_skills(text)
        assert "AWS" in skills

    def test_handles_case_insensitive(self):
        text = "python react mongodb docker"
        skills = extract_skills(text)
        assert "Python" in skills
        assert "React" in skills
        assert "MongoDB" in skills
        assert "Docker" in skills

    def test_empty_text_returns_empty(self):
        skills = extract_skills("")
        assert skills == []

    def test_no_skills_in_text(self):
        text = "I enjoy hiking and reading books"
        skills = extract_skills(text)
        # Some common words might match soft skills, but no technical skills
        assert len(skills) < 3


class TestExtractContactInfo:
    """Tests for contact information extraction."""

    def test_extracts_email(self):
        text = "Contact me at john.doe@example.com"
        info = extract_contact_info(text)
        assert info["email"] == "john.doe@example.com"

    def test_extracts_phone(self):
        text = "Phone: +1-555-123-4567"
        info = extract_contact_info(text)
        assert info["phone"] is not None

    def test_handles_missing_info(self):
        text = "No contact information here"
        info = extract_contact_info(text)
        assert info["email"] is None
        assert info["linkedin"] is None


class TestCategorizeSkills:
    """Tests for skill categorization."""

    def test_categorizes_programming_languages(self):
        skills = ["Python", "JavaScript", "Java"]
        categorized = categorize_skills(skills)
        assert "programming_languages" in categorized

    def test_categorizes_frameworks(self):
        skills = ["React", "Django"]
        categorized = categorize_skills(skills)
        assert "web_frameworks" in categorized

    def test_mixed_categories(self):
        skills = ["Python", "React", "Docker", "AWS"]
        categorized = categorize_skills(skills)
        assert len(categorized) >= 3


class TestExtractJobDescription:
    """Tests for job description parsing."""

    def test_extracts_required_skills(self):
        jd = """
        Senior Software Engineer

        Required Skills:
        - Python
        - React
        - PostgreSQL
        - Docker

        5+ years of experience required.
        Bachelor's degree in Computer Science.
        """
        data = extract_job_description_data(jd)
        assert "Python" in data["required_skills"] or "Python" in data["all_skills"]
        assert data["min_experience_years"] == 5.0

    def test_handles_empty_jd(self):
        data = extract_job_description_data("")
        assert data["all_skills"] == []


class TestExtractResumeData:
    """Tests for full resume extraction pipeline."""

    def test_full_extraction(self):
        resume_text = """
        John Doe
        john.doe@email.com | +1-555-0123

        SKILLS
        Python, JavaScript, React, Node.js, PostgreSQL, Docker, AWS

        EXPERIENCE
        Software Engineer at Google
        Jan 2020 - Present
        - Built scalable microservices using Python and FastAPI
        - Deployed applications on AWS ECS

        EDUCATION
        Bachelor's in Computer Science
        Stanford University, 2019

        CERTIFICATIONS
        AWS Certified Solutions Architect
        """
        data = extract_resume_data(resume_text)
        assert data["email"] == "john.doe@email.com"
        assert "Python" in data["skills"]
        assert "React" in data["skills"]
        assert len(data["skills"]) >= 5
        assert len(data["certifications"]) >= 1

    def test_student_resume_extraction_filters_projects_and_activities(self):
        resume_text = """
        Jane Smith
        jane.smith@email.com

        EDUCATION
        B.S. in Computer Science, University of California, Berkeley (Expected May 2026)

        PROJECTS
        AI Chatbot (Jan 2024 - Mar 2024)
        - Built chatbot using Python and PyTorch
        
        LEADERSHIP & ACTIVITIES
        Treasurer, Computer Science Club (Sep 2023 - Present)
        - Managed club budget and coordinated workshops
        
        EXPERIENCE
        Software Engineering Intern at Startup Corp (Jun 2024 - Aug 2024)
        - Developed REST APIs in Go
        """
        data = extract_resume_data(resume_text)
        assert data["name"] == "Jane Smith"
        # Since PROJECTS and LEADERSHIP & ACTIVITIES sections should be skipped in extract_experience,
        # only the Startup Corp experience should be extracted.
        assert len(data["experience"]) == 1
        assert data["experience"][0]["company"] == "Startup Corp"
        # Total experience years should be 0.0 because the only experience is an internship which is excluded from professional years count
        assert data["total_experience_years"] == 0.0

    def test_does_not_extract_degree_from_common_verb_be(self):
        resume_text = """
        Robert Cooper
        Full Responsible for completing reports and monitoring the grounds.
        be responsible for everything.
        To be admitted.
        """
        data = extract_resume_data(resume_text)
        # Should not extract any Bachelors degree from 'be' or 'be admitted'
        assert len(data["education"]) == 0

    def test_comma_and_hyphen_date_ranges(self):
        resume_text = """
        John Dev
        john@dev.com
        
        EXPERIENCE
        Software Engineer at Microsoft
        July, 2021 - August, 2023
        - Coding in C++
        
        Software Engineer at Amazon
        06-2019 - 05-2021
        - Coding in Python
        """
        data = extract_resume_data(resume_text)
        assert len(data["experience"]) == 2
        assert data["experience"][0]["months"] == 25
        assert data["experience"][1]["months"] == 23
        assert data["total_experience_years"] == 4.0
