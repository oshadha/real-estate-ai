from unittest.mock import AsyncMock, patch

import pytest
from openai import OpenAIError

from real_estate_ai.language_models import LanguageModelError
from real_estate_ai.openai_language_models import OpenAILanguageModelClient


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_openai_client_returns_response_output_text() -> None:
    with patch("real_estate_ai.openai_language_models.AsyncOpenAI") as openai_class:
        sdk_client = openai_class.return_value
        sdk_client.responses.create = AsyncMock()
        sdk_client.responses.create.return_value.output_text = (
            '{"summary":"Strong match","highlights":["Within budget"]}'
        )

        client = OpenAILanguageModelClient(
            api_key="test-api-key",
            model="gpt-5.6-luna",
        )

        result = await client.complete(
            system_message="Return JSON.",
            user_message="Explain this recommendation.",
        )

    assert result == ('{"summary":"Strong match","highlights":["Within budget"]}')

    sdk_client.responses.create.assert_awaited_once_with(
        model="gpt-5.6-luna",
        instructions="Return JSON.",
        input="Explain this recommendation.",
    )


@pytest.mark.anyio
async def test_openai_client_translates_sdk_errors() -> None:
    with patch("real_estate_ai.openai_language_models.AsyncOpenAI") as openai_class:
        sdk_client = openai_class.return_value
        sdk_client.responses.create = AsyncMock(side_effect=OpenAIError("Connection failed"))

        client = OpenAILanguageModelClient(
            api_key="test-api-key",
            model="gpt-5.6-luna",
        )

        with pytest.raises(
            LanguageModelError,
            match="OpenAI request failed",
        ):
            await client.complete(
                system_message="Return JSON.",
                user_message="Explain this recommendation.",
            )
