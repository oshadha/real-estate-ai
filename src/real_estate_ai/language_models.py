from typing import Protocol, TypeVar

from pydantic import BaseModel

from real_estate_ai.evidence import build_recommendation_evidence
from real_estate_ai.models import BuyerPreferences, PropertyMatch
from real_estate_ai.prompts import build_recommendation_prompt
from real_estate_ai.structured_outputs import (
    GeneratedRecommendationSummary,
    RecommendationExplanation,
)

ResponseModelT = TypeVar(
    "ResponseModelT",
    bound=BaseModel,
)


class LanguageModelError(RuntimeError):
    """Raised when a language-model provider call fails."""


class LanguageModelClient(Protocol):
    async def complete(
        self,
        *,
        system_message: str,
        user_message: str,
        response_model: type[ResponseModelT],
    ) -> ResponseModelT: ...


class StructuredRecommendationGenerator:
    def __init__(self, client: LanguageModelClient) -> None:
        self._client = client

    async def generate(
        self,
        match: PropertyMatch,
        preferences: BuyerPreferences,
    ) -> RecommendationExplanation:
        evidence = build_recommendation_evidence(
            match,
            preferences,
        )
        prompt = build_recommendation_prompt(
            match,
            preferences,
            evidence,
        )

        generated_summary = await self._client.complete(
            system_message=prompt.system_message,
            user_message=prompt.user_message,
            response_model=GeneratedRecommendationSummary,
        )

        return RecommendationExplanation(
            summary=generated_summary.summary,
            strengths=list(evidence.strengths),
            considerations=list(evidence.considerations),
        )
