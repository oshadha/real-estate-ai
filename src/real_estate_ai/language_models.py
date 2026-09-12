from typing import Protocol

from real_estate_ai.models import BuyerPreferences, PropertyMatch
from real_estate_ai.prompts import build_recommendation_prompt
from real_estate_ai.structured_outputs import RecommendationExplanation


class LanguageModelError(RuntimeError):
    """Raised when a language-model provider call fails."""


class LanguageModelClient(Protocol):
    async def complete(
        self,
        *,
        system_message: str,
        user_message: str,
    ) -> str: ...


class StructuredRecommendationGenerator:
    def __init__(self, client: LanguageModelClient) -> None:
        self._client = client

    async def generate(
        self,
        match: PropertyMatch,
        preferences: BuyerPreferences,
    ) -> RecommendationExplanation:
        prompt = build_recommendation_prompt(match, preferences)

        raw_output = await self._client.complete(
            system_message=prompt.system_message,
            user_message=prompt.user_message,
        )

        return RecommendationExplanation.model_validate_json(raw_output)
