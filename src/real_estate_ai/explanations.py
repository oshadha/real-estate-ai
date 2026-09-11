from typing import Protocol

from real_estate_ai.models import (
    BuyerPreferences,
    PropertyMatch,
)


class RecommendationExplainer(Protocol):
    async def explain(
        self,
        match: PropertyMatch,
        preferences: BuyerPreferences,
    ) -> str: ...


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
