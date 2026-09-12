import pytest

from real_estate_ai.explanations import (
    ResilientRecommendationExplainer,
    TemplateRecommendationExplainer,
)
from real_estate_ai.language_models import StructuredRecommendationGenerator
from real_estate_ai.models import (
    BuyerPreferences,
    PropertyListing,
    PropertyMatch,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class SequenceLanguageModelClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.call_count = 0

    async def complete(
        self,
        *,
        system_message: str,
        user_message: str,
    ) -> str:
        response = self._responses[self.call_count]
        self.call_count += 1
        return response


def _match_and_preferences() -> tuple[
    PropertyMatch,
    BuyerPreferences,
]:
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

    return PropertyMatch(listing=listing, score=91.45), preferences


@pytest.mark.anyio
async def test_invalid_output_is_retried() -> None:
    client = SequenceLanguageModelClient(
        [
            '{"summary": "", "strengths": [], "considerations": []}',
            """
            {
                "summary": "A valid explanation after retry.",
                "strengths": ["Preferred location"],
                "considerations": ["Above budget"]
            }
            """,
        ]
    )
    generator = StructuredRecommendationGenerator(client)
    explainer = ResilientRecommendationExplainer(
        generator,
        TemplateRecommendationExplainer(),
    )
    match, preferences = _match_and_preferences()

    explanation = await explainer.explain(match, preferences)

    assert explanation == "A valid explanation after retry."
    assert client.call_count == 2


@pytest.mark.anyio
async def test_template_is_used_after_all_attempts_fail() -> None:
    invalid_output = '{"summary": "", "strengths": [], "considerations": []}'
    client = SequenceLanguageModelClient([invalid_output, invalid_output])
    generator = StructuredRecommendationGenerator(client)
    explainer = ResilientRecommendationExplainer(
        generator,
        TemplateRecommendationExplainer(),
    )
    match, preferences = _match_and_preferences()

    explanation = await explainer.explain(match, preferences)

    assert explanation == (
        "DXB-1001 scored 91.45/100. "
        "It matches the preferred location and is AED 50,000 over budget."
    )
    assert client.call_count == 2
