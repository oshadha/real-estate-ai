import pytest

from real_estate_ai.explanations import TemplateRecommendationExplainer
from real_estate_ai.models import (
    BuyerPreferences,
    PropertyListing,
    PropertyMatch,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_explanation_describes_location_and_budget() -> None:
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
    explainer = TemplateRecommendationExplainer()

    explanation = await explainer.explain(match, preferences)

    assert explanation.summary == (
        "DXB-1001 scored 91.45/100. "
        "It matches the preferred location and is AED 50,000 over budget."
    )
    assert explanation.strengths == [
        "Matches the preferred location",
        "Meets the minimum bedroom requirement",
    ]
    assert explanation.considerations == [
        "AED 50,000 over budget",
        "80 sqft below the minimum area",
    ]
