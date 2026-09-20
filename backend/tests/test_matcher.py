"""Tests for the matching algorithm."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nlp.matcher import (
    compute_skills_match,
    compute_experience_match,
    compute_education_match,
)


class TestSkillsMatch:
    """Tests for skills matching algorithm."""

    def test_perfect_match(self):
        result = compute_skills_match(
            resume_skills=["Python", "React", "Docker"],
            required_skills=["Python", "React", "Docker"],
        )
        assert result["score"] == 100.0
        assert len(result["missing_required"]) == 0

    def test_no_match(self):
        result = compute_skills_match(
            resume_skills=["Java", "Angular"],
            required_skills=["Python", "React", "Docker"],
        )
        assert result["score"] < 35
        assert len(result["missing_required"]) == 3

    def test_partial_match(self):
        result = compute_skills_match(
            resume_skills=["Python", "React"],
            required_skills=["Python", "React", "Docker", "AWS"],
        )
        assert 40 < result["score"] < 80
        assert "Docker" in [s.lower() for s in result["missing_required"]] or "Docker" in result["missing_required"]

    def test_extra_skills_identified(self):
        result = compute_skills_match(
            resume_skills=["Python", "React", "Go", "Rust"],
            required_skills=["Python", "React"],
        )
        assert result["score"] == 100.0
        assert len(result["extra_skills"]) == 2

    def test_empty_required(self):
        result = compute_skills_match(
            resume_skills=["Python"],
            required_skills=[],
        )
        assert result["score"] >= 80

    def test_preferred_skills(self):
        result = compute_skills_match(
            resume_skills=["Python", "React", "TypeScript"],
            required_skills=["Python", "React"],
            preferred_skills=["TypeScript", "Docker"],
        )
        assert result["score"] > 80

    def test_zero_matched_skills_evaluates_to_zero(self):
        result = compute_skills_match(
            resume_skills=["Security Guarding", "Patrolling", "First Aid"],
            required_skills=["Python", "React", "Docker"],
            preferred_skills=["TypeScript"],
        )
        assert result["score"] == 0.0

    def test_js_and_json_normalization(self):
        result = compute_skills_match(
            resume_skills=["JS", "JSON"],
            required_skills=["JavaScript", "JSON"],
        )
        assert result["score"] == 100.0
        assert len(result["missing_required"]) == 0


class TestExperienceMatch:
    """Tests for experience matching."""

    def test_exceeds_requirement(self):
        result = compute_experience_match(resume_years=7, required_years=5)
        assert result["score"] == 100.0

    def test_meets_requirement(self):
        result = compute_experience_match(resume_years=5, required_years=5)
        assert result["score"] == 100.0

    def test_below_requirement(self):
        result = compute_experience_match(resume_years=2, required_years=5)
        assert result["score"] < 60

    def test_no_requirement(self):
        result = compute_experience_match(resume_years=3, required_years=None)
        assert result["score"] == 100.0

    def test_student_applying_to_senior_role_no_boost(self):
        # Student with 0 experience applying to a 5 year experience job should NOT get boosted
        result = compute_experience_match(
            resume_years=0,
            required_years=5,
            is_entry_level=False,
            projects=[{"title": "Project 1"}],
            is_student=True,
        )
        assert result["score"] < 20.0

    def test_student_applying_to_junior_role_gets_boost(self):
        # Student with 0 experience applying to a junior/entry-level job should get boosted
        result = compute_experience_match(
            resume_years=0,
            required_years=2,
            is_entry_level=True,
            projects=[{"title": "Project 1"}],
            is_student=True,
        )
        assert result["score"] >= 85.0


class TestEducationMatch:
    """Tests for education matching."""

    def test_meets_requirement(self):
        result = compute_education_match(
            resume_education=[{"level": "Bachelors", "degree": "B.S. in CS"}],
            jd_education_requirements=["Bachelors"],
        )
        assert result["score"] >= 90

    def test_exceeds_requirement(self):
        result = compute_education_match(
            resume_education=[{"level": "Masters", "degree": "M.S. in CS"}],
            jd_education_requirements=["Bachelors"],
        )
        assert result["score"] == 100.0

    def test_no_requirement(self):
        result = compute_education_match(
            resume_education=[{"level": "Bachelors", "degree": "B.S."}],
            jd_education_requirements=[],
        )
        assert result["score"] == 100.0

    def test_student_in_progress_fails_requirement(self):
        # Current student studying for Bachelors does not meet Bachelors requirement
        result = compute_education_match(
            resume_education=[{"level": "Bachelors", "degree": "B.S. in CS", "year": "2027"}],
            jd_education_requirements=["Bachelors"],
            is_student=True,
        )
        assert result["score"] == 0.0

    def test_student_completed_previous_meets_requirement(self):
        # Current student studying for Masters has completed Bachelors, meeting Bachelors requirement
        result = compute_education_match(
            resume_education=[
                {"level": "Masters", "degree": "M.S. in CS", "year": "2027"},
                {"level": "Bachelors", "degree": "B.S. in CS", "year": "2024"}
            ],
            jd_education_requirements=["Bachelors"],
            is_student=True,
        )
        assert result["score"] == 100.0

    def test_future_grad_year_in_progress_fails_requirement(self):
        # Candidate not explicitly marked is_student but with future graduation year does not meet requirement
        from datetime import datetime
        future_year = str(datetime.now().year + 2)
        result = compute_education_match(
            resume_education=[{"level": "Bachelors", "degree": "B.S. in CS", "year": future_year}],
            jd_education_requirements=["Bachelors"],
            is_student=False,
        )
        assert result["score"] == 0.0

    def test_student_in_progress_meets_entry_level(self):
        # Current student studying for Bachelors does not meet Bachelors requirement even for entry-level roles
        # (relying on recommender to offer pursuing guidelines)
        result = compute_education_match(
            resume_education=[{"level": "Bachelors", "degree": "B.S. in CS", "year": "2027"}],
            jd_education_requirements=["Bachelors"],
            is_student=True,
            is_entry_level=True,
        )
        assert result["score"] == 0.0

    def test_degree_with_in_progress_keyword_detected(self):
        # Degree with 'Current Year' in its title is detected as in-progress even if is_student=False
        result = compute_education_match(
            resume_education=[{"level": "Bachelors", "degree": "Btech cse\nVIT University, Vellore\nCurrent Year"}],
            jd_education_requirements=["Bachelors"],
            is_student=False,
            is_entry_level=False,
        )
        # Should be 0.0 because it's detected as in-progress (completed level becomes Associates) and not entry level
        assert result["score"] == 0.0

    def test_in_progress_integrated_course_rejected(self):
        # In-progress integrated B.Tech + M.Tech degree does not count as having completed a Bachelors degree
        result = compute_education_match(
            resume_education=[{"level": "Masters", "degree": "Integrated B.Tech + M.Tech", "year": "2027"}],
            jd_education_requirements=["Bachelors"],
            is_student=True,
        )
        # Since it is integrated and in-progress, completed_level is 0, so score should be 0.0
        assert result["score"] == 0.0

    def test_completed_integrated_course_accepted(self):
        # Completed integrated B.Tech + M.Tech degree counts as meeting Bachelors or Masters requirements
        result = compute_education_match(
            resume_education=[{"level": "Masters", "degree": "Integrated B.Tech + M.Tech", "year": "2023"}],
            jd_education_requirements=["Bachelors"],
            is_student=False,
        )
        assert result["score"] == 100.0

    def test_eee_degree_field_detected_as_unrelated(self):
        # Degree in EEE is detected as unrelated for a tech role
        result = compute_education_match(
            resume_education=[{"level": "Bachelors", "degree": "B.Tech in EEE", "year": "2023"}],
            jd_education_requirements=["Bachelors"],
            is_student=False,
        )
        assert result["score"] == 0.0
        assert "unrelated" in result["detail"].lower()

    def test_semantic_skill_match_partial_credit(self):
        # PyTorch is semantically similar to TensorFlow
        result = compute_skills_match(
            resume_skills=["PyTorch", "React"],
            required_skills=["TensorFlow", "React"],
        )
        # Score should be higher than a complete missing required skill (which would be 50%),
        # because PyTorch gets partial credit for TensorFlow
        assert result["score"] > 60.0
        # Semantic matches list should contain the details
        semantic_matches = result.get("semantic_matches", [])
        assert len(semantic_matches) >= 1
        assert any(m["missing"] == "TensorFlow" for m in semantic_matches)
