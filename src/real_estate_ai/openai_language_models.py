from openai import AsyncOpenAI, OpenAIError

from real_estate_ai.language_models import LanguageModelError


class OpenAILanguageModelClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            max_retries=0,
            timeout=timeout_seconds,
        )
        self._model = model

    async def complete(
        self,
        *,
        system_message: str,
        user_message: str,
    ) -> str:
        try:
            response = await self._client.responses.create(
                model=self._model,
                instructions=system_message,
                input=user_message,
            )
        except OpenAIError as error:
            raise LanguageModelError("OpenAI request failed.") from error

        output = response.output_text

        if not output.strip():
            raise LanguageModelError("OpenAI returned an empty response.")

        return output
