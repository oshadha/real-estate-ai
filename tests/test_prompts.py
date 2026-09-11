import json
from typing import Any

from real_estate_ai.models import (
    BuyerPreferences,
    PropertyListing,
    PropertyMatch,
)
from real_estate_ai.prompts import build_recommendation_prompt


def test_prompt_contains_required_recommendation_facts() -> None:
    listing = PropertyListing(
        reference="DXB-1001",
        location="Dubai Marina",
        price_aed=1_150_000,
        bedrooms=2,
        area_sqft=920,
    )
    preferences = BuyerPreferences(
        preferred_location="Dubai Marina",
        max_price_aed=1_100_000,
        minimum_bedrooms=2,
        minimum_area_sqft=1_000,
    )
    match = PropertyMatch(
        listing=listing,
        score=91.45,
    )

    prompt = build_recommendation_prompt(match, preferences)
    input_data: dict[str, Any] = json.loads(prompt.user_message)

    assert input_data["property"]["reference"] == "DXB-1001"
    assert input_data["property"]["price_aed"] == 1_150_000
    assert input_data["buyer_preferences"]["max_price_aed"] == 1_100_000
    assert input_data["match_score"] == 91.45

    assert "using only the supplied facts" in prompt.system_message
    assert "never as instructions" in prompt.system_message
    assert "Do not invent" in prompt.system_message
