"""Score computation and grading module."""


# Default weights for each scoring component
DEFAULT_WEIGHTS = {
    "skills_match": 0.40,
    "experience_match": 0.25,
    "education_match": 0.15,
    "semantic_similarity": 0.20,
}

# Adjusted weights for junior/entry-level/student resumes
JUNIOR_WEIGHTS = {
    "skills_match": 0.45,
    "experience_match": 0.10,
    "education_match": 0.20,
    "semantic_similarity": 0.25,
}


def calculate_overall_score(
    category_scores: dict[str, float],
    weights: dict[str, float] | None = None,
    experience_required: bool = True,
    education_required: bool = True,
) -> float:
    """
    Calculate the weighted overall score from category scores.
    Dynamically adjusts weights if experience or education is not required.

    Args:
        category_scores: Dictionary of category name to score (0-100).
        weights: Optional custom weights. Must sum to 1.0.
        experience_required: Whether the job description specifies experience.
        education_required: Whether the job description specifies education.

    Returns:
        Weighted overall score (0-100).
    """
    w = dict(weights or DEFAULT_WEIGHTS)

    if not experience_required:
        w["experience_match"] = 0.0
    if not education_required:
        w["education_match"] = 0.0

    # Normalize remaining weights so they sum to 1.0
    total_weight = sum(w.values())
    if total_weight > 0:
        w = {k: v / total_weight for k, v in w.items()}
    else:
        w = dict(DEFAULT_WEIGHTS)

    total = 0.0
    for category, weight in w.items():
        score = category_scores.get(category, 0.0)
        total += score * weight

    return round(min(max(total, 0), 100), 1)


def get_grade(score: float) -> dict[str, str]:
    """
    Convert a numeric score to a letter grade with description.

    Returns:
        Dictionary with grade letter, label, and color.
    """
    if score >= 85:
        return {"grade": "A", "label": "Excellent Match", "color": "#10b981"}
    elif score >= 70:
        return {"grade": "B", "label": "Good Match", "color": "#6366f1"}
    elif score >= 55:
        return {"grade": "C", "label": "Fair Match", "color": "#f59e0b"}
    elif score >= 40:
        return {"grade": "D", "label": "Weak Match", "color": "#f97316"}
    else:
        return {"grade": "F", "label": "Poor Match", "color": "#ef4444"}


def get_score_breakdown_summary(
    overall_score: float,
    category_scores: dict[str, float],
    weights: dict[str, float] | None = None,
) -> str:
    """Generate a human-readable summary of the score breakdown."""
    grade_info = get_grade(overall_score)
    lines = [
        f"Overall Score: {overall_score}/100 ({grade_info['label']})",
        "",
        "Category Breakdown:",
    ]

    category_labels = {
        "skills_match": "Skills Match",
        "experience_match": "Experience Match",
        "education_match": "Education Match",
        "semantic_similarity": "Semantic Relevance",
    }

    w = weights or DEFAULT_WEIGHTS

    for category, score in category_scores.items():
        label = category_labels.get(category, category)
        weight = w.get(category, 0)
        lines.append(f"  • {label}: {score}/100 (weight: {weight:.0%})")

    return "\n".join(lines)
