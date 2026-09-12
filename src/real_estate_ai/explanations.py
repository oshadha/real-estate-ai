import logging
from typing import Protocol

from pydantic import ValidationError

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
    ) -> str: ...


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
    ) -> str:
        listing = match.listing

        if listing.location.casefold() == preferences.preferred_location.casefold():
            location_text = "matches the preferred location"
        else:
            location_text = "is outside the preferred location"

        if listing.price_aed <= preferences.max_price_aed:
            budget_text = "is within budget"
        else:
            amount_over_budget = listing.price_aed - preferences.max_price_aed
            budget_text = f"is AED {amount_over_budget:,.0f} over budget"

        return (
            f"{listing.reference} scored {match.score:.2f}/100. "
            f"It {location_text} and {budget_text}."
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
    ) -> str:
        for attempt in range(1, self._max_attempts + 1):
            try:
                explanation = await self._generator.generate(
                    match,
                    preferences,
                )
                return explanation.summary
            except (LanguageModelError, ValidationError) as error:
                logger.warning(
                    "Explanation attempt %d failed for property %s: %s",
                    attempt,
                    match.listing.reference,
                    type(error).__name__,
                )

        logger.error(
            "Using template explanation for property %s",
            match.listing.reference,
        )

        return await self._fallback.explain(match, preferences)
