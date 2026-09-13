import logging
from typing import Protocol

from pydantic import ValidationError

from real_estate_ai.evidence import build_recommendation_evidence
from real_estate_ai.language_models import LanguageModelError
from real_estate_ai.models import (
    BuyerPreferences,
    PropertyMatch,
)
from real_estate_ai.structured_outputs import RecommendationExplanation

logger = logging.getLogger(__name__)


class RecommendationExplainer(Protocol):
    async def explain(
        self,
        match: PropertyMatch,
        preferences: BuyerPreferences,
    ) -> RecommendationExplanation: ...


class RecommendationExplanationGenerator(Protocol):
    async def generate(
        self,
        match: PropertyMatch,
        preferences: BuyerPreferences,
    ) -> RecommendationExplanation: ...


class TemplateRecommendationExplainer:
    async def explain(
        self,
        match: PropertyMatch,
        preferences: BuyerPreferences,
    ) -> RecommendationExplanation:
        listing = match.listing
        evidence = build_recommendation_evidence(
            match,
            preferences,
        )

        if listing.location.casefold() == preferences.preferred_location.casefold():
            location_text = "matches the preferred location"
        else:
            location_text = "is outside the preferred location"

        if listing.price_aed <= preferences.max_price_aed:
            budget_text = "is within budget"
        else:
            amount_over_budget = listing.price_aed - preferences.max_price_aed
            budget_text = f"is AED {amount_over_budget:,.0f} over budget"

        summary = (
            f"{listing.reference} scored {match.score:.2f}/100. "
            f"It {location_text} and {budget_text}."
        )

        return RecommendationExplanation(
            summary=summary,
            strengths=list(evidence.strengths),
            considerations=list(evidence.considerations),
        )


class ResilientRecommendationExplainer:
    def __init__(
        self,
        generator: RecommendationExplanationGenerator,
        fallback: RecommendationExplainer,
        max_attempts: int = 2,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        self._generator = generator
        self._fallback = fallback
        self._max_attempts = max_attempts

    async def explain(
    self,
    match: PropertyMatch,
    preferences: BuyerPreferences,
) -> RecommendationExplanation:
        for attempt in range(1, self._max_attempts + 1):
            try:
                explanation = await self._generator.generate(
                    match,
                    preferences,
                )

                logger.info(
                    "Recommendation explanation generated",
                    extra={
                        "event_name": (
                            "recommendation.explanation.generated"
                        ),
                        "property_reference": match.listing.reference,
                        "generation_source": "language_model",
                        "attempt_count": attempt,
                    },
                )

                return explanation
            except (LanguageModelError, ValidationError) as error:
                logger.warning(
                    "Recommendation explanation attempt failed",
                    extra={
                        "event_name": (
                            "recommendation.explanation.attempt_failed"
                        ),
                        "property_reference": match.listing.reference,
                        "attempt_number": attempt,
                        "error_type": type(error).__name__,
                    },
                )

        fallback_explanation = await self._fallback.explain(
            match,
            preferences,
        )

        logger.error(
            "Using template recommendation explanation",
            extra={
                "event_name": "recommendation.explanation.fallback",
                "property_reference": match.listing.reference,
                "generation_source": "template",
                "attempt_count": self._max_attempts,
            },
        )

        return fallback_explanation
