import logging

import pytest

from real_estate_ai.explanations import (
    ResilientRecommendationExplainer,
    TemplateRecommendationExplainer,
)
from real_estate_ai.language_models import (
    ResponseModelT,
    StructuredRecommendationGenerator,
)
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
        self._responses = iter(responses)
        self.call_count = 0

    async def complete(
        self,
        *,
        system_message: str,
        user_message: str,
        response_model: type[ResponseModelT],
    ) -> ResponseModelT:
        self.call_count += 1
        response = next(self._responses)

        return response_model.model_validate_json(response)


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

    return PropertyMatch(
        listing=listing,
        score=91.45,
    ), preferences


@pytest.mark.anyio
async def test_invalid_output_is_retried(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = SequenceLanguageModelClient(
        [
            '{"summary": ""}',
            """
            {
                "summary": "A valid explanation after retry."
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

    with caplog.at_level(
        logging.INFO,
        logger="real_estate_ai.explanations",
    ):
        explanation = await explainer.explain(
            match,
            preferences,
        )

    assert explanation.summary == "A valid explanation after retry."
    assert client.call_count == 2

    failed_records = [
        record
        for record in caplog.records
        if getattr(record, "event_name", None) == "recommendation.explanation.attempt_failed"
    ]

    assert len(failed_records) == 1
    failed_records[0].__dict__["attempt_number"] == 1
    failed_records[0].__dict__["error_type"] == "ValidationError"

    generated_record = next(
        record
        for record in caplog.records
        if getattr(record, "event_name", None) == "recommendation.explanation.generated"
    )

    assert generated_record.__dict__["generation_source"] == "language_model"
    assert generated_record.__dict__["attempt_count"] == 2
    assert generated_record.__dict__["property_reference"] == "DXB-1001"


@pytest.mark.anyio
async def test_template_is_used_after_all_attempts_fail(
    caplog: pytest.LogCaptureFixture,
) -> None:
    invalid_output = '{"summary": ""}'
    client = SequenceLanguageModelClient(
        [
            invalid_output,
            invalid_output,
        ]
    )
    generator = StructuredRecommendationGenerator(client)
    explainer = ResilientRecommendationExplainer(
        generator,
        TemplateRecommendationExplainer(),
    )
    match, preferences = _match_and_preferences()

    with caplog.at_level(
        logging.INFO,
        logger="real_estate_ai.explanations",
    ):
        explanation = await explainer.explain(
            match,
            preferences,
        )

    assert explanation.summary == (
        "DXB-1001 scored 91.45/100. "
        "It matches the preferred location and is AED 50,000 over budget."
    )
    assert client.call_count == 2

    failed_records = [
        record
        for record in caplog.records
        if getattr(record, "event_name", None) == "recommendation.explanation.attempt_failed"
    ]

    assert len(failed_records) == 2
    assert [record.__dict__["attempt_number"] for record in failed_records] == [1, 2]

    fallback_record = next(
        record
        for record in caplog.records
        if getattr(record, "event_name", None) == "recommendation.explanation.fallback"
    )

    fallback_fields = vars(fallback_record)

    assert fallback_fields["generation_source"] == "template"
    assert fallback_fields["attempt_count"] == 2
    assert fallback_fields["property_reference"] == "DXB-1001"
