import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from openai import OpenAIError

from real_estate_ai.language_models import LanguageModelError
from real_estate_ai.openai_language_models import OpenAILanguageModelClient
from real_estate_ai.structured_outputs import (
    GeneratedRecommendationSummary,
    RecommendationExplanation,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_openai_client_returns_parsed_response() -> None:
    expected = RecommendationExplanation(
        summary="Strong match",
        strengths=["Within budget"],
        considerations=[],
    )

    with patch("real_estate_ai.openai_language_models.AsyncOpenAI") as openai_class:
        sdk_client = openai_class.return_value
        sdk_client.responses.parse = AsyncMock()
        sdk_client.responses.parse.return_value.output_parsed = expected

        client = OpenAILanguageModelClient(
            api_key="test-api-key",
            model="gpt-5.6-luna",
        )

        result = await client.complete(
            system_message="Return a structured response.",
            user_message="Explain this recommendation.",
            response_model=RecommendationExplanation,
        )

    assert result == expected

    sdk_client.responses.parse.assert_awaited_once_with(
        model="gpt-5.6-luna",
        instructions="Return a structured response.",
        input="Explain this recommendation.",
        text_format=RecommendationExplanation,
    )


@pytest.mark.anyio
async def test_openai_client_translates_sdk_errors() -> None:
    with patch("real_estate_ai.openai_language_models.AsyncOpenAI") as openai_class:
        sdk_client = openai_class.return_value
        sdk_client.responses.parse = AsyncMock(side_effect=OpenAIError("Connection failed"))

        client = OpenAILanguageModelClient(
            api_key="test-api-key",
            model="gpt-5.6-luna",
        )

        with pytest.raises(
            LanguageModelError,
            match="OpenAI request failed",
        ):
            await client.complete(
                system_message="Return a structured response.",
                user_message="Explain this recommendation.",
                response_model=RecommendationExplanation,
            )


@pytest.mark.anyio
async def test_openai_client_rejects_missing_parsed_output() -> None:
    with patch("real_estate_ai.openai_language_models.AsyncOpenAI") as openai_class:
        sdk_client = openai_class.return_value
        sdk_client.responses.parse = AsyncMock()
        sdk_client.responses.parse.return_value.output_parsed = None

        client = OpenAILanguageModelClient(
            api_key="test-api-key",
            model="gpt-5.6-luna",
        )

        with pytest.raises(
            LanguageModelError,
            match="no parsed response",
        ):
            await client.complete(
                system_message="Return a structured response.",
                user_message="Explain this recommendation.",
                response_model=RecommendationExplanation,
            )


@pytest.mark.anyio
async def test_openai_client_logs_safe_usage_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    expected = GeneratedRecommendationSummary(summary="A safe generated summary.")
    sdk_response = SimpleNamespace(
        output_parsed=expected,
        id="resp-test-123",
        usage=SimpleNamespace(
            input_tokens=120,
            output_tokens=30,
            total_tokens=150,
        ),
    )

    with (
        patch("real_estate_ai.openai_language_models.AsyncOpenAI") as openai_class,
        patch(
            "real_estate_ai.openai_language_models.perf_counter",
            side_effect=[100.0, 100.125],
        ),
    ):
        sdk_client = openai_class.return_value
        sdk_client.responses.parse = AsyncMock(return_value=sdk_response)

        client = OpenAILanguageModelClient(
            api_key="test-api-key",
            model="gpt-5.6-luna",
        )

        with caplog.at_level(
            logging.INFO,
            logger="real_estate_ai.openai_language_models",
        ):
            result = await client.complete(
                system_message="Private system instructions",
                user_message="Private buyer details",
                response_model=GeneratedRecommendationSummary,
            )

    assert result == expected

    record = next(
        record
        for record in caplog.records
        if getattr(record, "event_name", None) == "openai.request.completed"
    )

    fields = vars(record)

    assert fields["model"] == "gpt-5.6-luna"
    assert fields["response_id"] == "resp-test-123"
    assert fields["duration_ms"] == 125.0
    assert fields["input_tokens"] == 120
    assert fields["output_tokens"] == 30
    assert fields["total_tokens"] == 150

    assert "Private system instructions" not in caplog.text
    assert "Private buyer details" not in caplog.text
    assert "test-api-key" not in caplog.text
