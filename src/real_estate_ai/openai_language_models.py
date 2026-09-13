import logging
from time import perf_counter

from openai import AsyncOpenAI, OpenAIError

from real_estate_ai.language_models import (
    LanguageModelError,
    ResponseModelT,
)

logger = logging.getLogger(__name__)


class OpenAILanguageModelClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            timeout=30.0,
            max_retries=0,
        )

    async def complete(
        self,
        *,
        system_message: str,
        user_message: str,
        response_model: type[ResponseModelT],
    ) -> ResponseModelT:
        started_at = perf_counter()

        try:
            response = await self._client.responses.parse(
                model=self._model,
                instructions=system_message,
                input=user_message,
                text_format=response_model,
            )
        except OpenAIError as error:
            duration_ms = round(
                (perf_counter() - started_at) * 1_000,
                2,
            )

            logger.warning(
                "OpenAI request failed",
                extra={
                    "event_name": "openai.request.failed",
                    "model": self._model,
                    "duration_ms": duration_ms,
                    "error_type": type(error).__name__,
                    "response_id": getattr(error, "request_id", None),
                },
            )

            raise LanguageModelError("OpenAI request failed") from error

        duration_ms = round(
            (perf_counter() - started_at) * 1_000,
            2,
        )
        usage = response.usage

        logger.info(
            "OpenAI request completed",
            extra={
                "event_name": "openai.request.completed",
                "model": self._model,
                "response_id": response.id,
                "duration_ms": duration_ms,
                "input_tokens": (usage.input_tokens if usage is not None else None),
                "output_tokens": (usage.output_tokens if usage is not None else None),
                "total_tokens": (usage.total_tokens if usage is not None else None),
            },
        )

        parsed_output = response.output_parsed

        if parsed_output is None:
            raise LanguageModelError("OpenAI returned no parsed response")

        return parsed_output
