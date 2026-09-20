"""Tests for the OpenAI-based high-precision matching pipeline."""

import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nlp.openai_analyzer import analyze_with_openai
from config import get_settings


def test_openai_fallback_when_no_key():
    """Verify that the analyzer returns None (triggering fallback) when OPENAI_API_KEY is not set."""
    settings = get_settings()
    original_key = settings.openai_api_key
    settings.openai_api_key = None
    try:
        result = analyze_with_openai("Resume text", "Job description")
        assert result is None
    finally:
        settings.openai_api_key = original_key


@patch("nlp.openai_analyzer.OpenAI")
def test_openai_success_mock(mock_openai_class):
    """Verify that the analyzer correctly initializes OpenAI, executes request, and returns structured result."""
    settings = get_settings()
    original_key = settings.openai_api_key
    settings.openai_api_key = "mock_key"

    try:
        # Mock Client and completions endpoint
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Structure the mocked response to look like OpenAI API structured parse result
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        
        # JSON representation matching the Pydantic schema
        mock_message.content = """{
            "resume_data": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "phone": "+123456789",
                "linkedin": "linkedin.com/in/johndoe",
                "skills": ["Python", "Docker", "FastAPI"],
                "skills_categorized": {
                    "programming_languages": ["Python"],
                    "devops_tools": ["Docker"],
                    "web_frameworks": ["FastAPI"]
                },
                "education": [
                    {
                        "degree": "Bachelor of Science in Computer Science",
                        "level": "Bachelors",
                        "institution": "State University",
                        "year": "2022"
                    }
                ],
                "experience": [
                    {
                        "title": "Software Engineer",
                        "company": "Tech Corp",
                        "duration": "Jun 2022 - Present",
                        "description": "Building APIs with FastAPI",
                        "months": 24
                     }
                ],
                "projects": [],
                "certifications": [],
                "total_experience_years": 2.0,
                "is_student": false
            },
            "job_data": {
                "title": "FastAPI Developer",
                "all_skills": ["Python", "FastAPI", "Docker", "AWS"],
                "required_skills": ["Python", "FastAPI"],
                "preferred_skills": ["Docker", "AWS"],
                "min_experience_years": 2.0,
                "education_requirements": ["Bachelors"]
            },
            "overall_score": 85.5,
            "grade": {
                "grade": "A",
                "label": "Excellent Match",
                "color": "#10b981"
            },
            "category_scores": {
                "skills_match": 90.0,
                "experience_match": 80.0,
                "education_match": 100.0,
                "semantic_similarity": 75.0
            },
            "skills_detail": {
                "matched_skills": ["Python", "FastAPI", "Docker"],
                "missing_required": [],
                "missing_preferred": ["AWS"],
                "extra_skills": []
            },
            "strengths": ["Strong FastAPI experience", "Has required Bachelors degree"],
            "recommendations": [],
            "negatives": [
                {
                    "category": "Missing Preferred Skills",
                    "missing_item": "AWS",
                    "impact": "low",
                    "suggestion": "Candidate has not shown capability with AWS."
                }
            ],
            "ats_suggestions": [
                {
                    "category": "AWS Cloud",
                    "text": "What AWS services have you deployed FastAPI apps to?"
                }
            ],
            "hiring_assessment": {
                "verdict": "Auto-Shortlist (Strong Match)",
                "verdict_color": "#10b981",
                "leveling": "Junior Level (0-2 years)",
                "summary": "Excellent FastAPI skills.",
                "checklist": [],
                "screening_actions": []
            }
        }"""
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.beta.chat.completions.parse.return_value = mock_response

        # Execute
        result = analyze_with_openai("Resume text", "Job description")
        
        # Asserts
        assert result is not None
        assert result["resume_data"]["name"] == "John Doe"
        assert result["overall_score"] == 86.0  # Certified/Project boosts apply
        assert result["grade"]["grade"] == "A"
        assert len(result["skills_detail"]["matched_skills"]) == 3
        
        # Verify call arguments
        mock_openai_class.assert_called_once_with(api_key="mock_key")
        mock_client.beta.chat.completions.parse.assert_called_once()
        call_kwargs = mock_client.beta.chat.completions.parse.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert "response_format" in call_kwargs
        
    finally:
        settings.openai_api_key = original_key
