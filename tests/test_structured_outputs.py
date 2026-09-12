import pytest
from pydantic import ValidationError

from real_estate_ai.structured_outputs import RecommendationExplanation


def test_valid_llm_json_is_parsed() -> None:
    raw_output = """
    {
        "summary": "A strong location match slightly above budget.",
        "strengths": [
            "Matches the preferred location",
            "Meets the bedroom requirement"
        ],
        "considerations": [
            "AED 50,000 above budget"
        ]
    }
    """

    explanation = RecommendationExplanation.model_validate_json(raw_output)

    assert explanation.summary == ("A strong location match slightly above budget.")
    assert explanation.strengths == [
        "Matches the preferred location",
        "Meets the bedroom requirement",
    ]
    assert explanation.considerations == [
        "AED 50,000 above budget",
    ]


def test_invalid_llm_json_is_rejected() -> None:
    raw_output = """
    {
        "summary": "",
        "strengths": [],
        "considerations": [],
        "confidence": 0.98
    }
    """

    with pytest.raises(ValidationError) as exception:
        RecommendationExplanation.model_validate_json(raw_output)

    error_types = {error["type"] for error in exception.value.errors()}

    assert "string_too_short" in error_types
    assert "extra_forbidden" in error_types
