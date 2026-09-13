import pytest
from pydantic import BaseModel, ValidationError

from real_estate_ai.language_models import (
    ResponseModelT,
    StructuredRecommendationGenerator,
)
from real_estate_ai.models import (
    BuyerPreferences,
    PropertyListing,
    PropertyMatch,
)
from real_estate_ai.structured_outputs import (
    GeneratedRecommendationSummary,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class StubLanguageModelClient:
    def __init__(self, response: str) -> None:
        self._response = response
        self.system_message: str | None = None
        self.user_message: str | None = None
        self.response_model: type[BaseModel] | None = None

    async def complete(
        self,
        *,
        system_message: str,
        user_message: str,
        response_model: type[ResponseModelT],
    ) -> ResponseModelT:
        self.system_message = system_message
        self.user_message = user_message
        self.response_model = response_model

        return response_model.model_validate_json(self._response)


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

    return (
        PropertyMatch(listing=listing, score=91.45),
        preferences,
    )


@pytest.mark.anyio
async def test_generator_validates_model_output() -> None:
    client = StubLanguageModelClient(
        """
        {
            "summary": "A strong match slightly above budget."
        }
        """
    )
    generator = StructuredRecommendationGenerator(client)
    match, preferences = _match_and_preferences()

    explanation = await generator.generate(match, preferences)

    assert explanation.summary == "A strong match slightly above budget."
    assert explanation.strengths == [
        "Matches the preferred location",
        "Meets the minimum bedroom requirement",
    ]
    assert explanation.considerations == [
        "AED 50,000 over budget",
        "80 sqft below the minimum area",
    ]

    system_message = client.system_message
    user_message = client.user_message

    assert system_message is not None
    assert "Do not invent" in system_message

    assert user_message is not None
    assert '"reference": "DXB-1001"' in user_message
    assert '"verified_evidence"' in user_message

    assert client.response_model is GeneratedRecommendationSummary


@pytest.mark.anyio
async def test_generator_rejects_invalid_model_output() -> None:
    client = StubLanguageModelClient(
        """
        {
            "summary": ""
        }
        """
    )
    generator = StructuredRecommendationGenerator(client)
    match, preferences = _match_and_preferences()

    with pytest.raises(ValidationError):
        await generator.generate(match, preferences)
