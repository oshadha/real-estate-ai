from dataclasses import dataclass

from real_estate_ai.models import BuyerPreferences, PropertyMatch


@dataclass(frozen=True, slots=True)
class RecommendationEvidence:
    strengths: tuple[str, ...]
    considerations: tuple[str, ...]


def build_recommendation_evidence(
    match: PropertyMatch,
    preferences: BuyerPreferences,
) -> RecommendationEvidence:
    listing = match.listing

    strengths: list[str] = []
    considerations: list[str] = []

    if listing.location.casefold() == preferences.preferred_location.casefold():
        strengths.append("Matches the preferred location")
    else:
        considerations.append(
            f"Located in {listing.location} instead of {preferences.preferred_location}"
        )

    if listing.price_aed <= preferences.max_price_aed:
        strengths.append("Within the buyer's budget")
    else:
        amount_over_budget = listing.price_aed - preferences.max_price_aed
        considerations.append(f"AED {amount_over_budget:,.0f} over budget")

    meets_bedroom_requirement = listing.bedrooms >= preferences.minimum_bedrooms
    meets_area_requirement = listing.area_sqft >= preferences.minimum_area_sqft

    if meets_bedroom_requirement and meets_area_requirement:
        strengths.append("Meets the minimum bedroom and area requirements")
    elif not meets_bedroom_requirement and not meets_area_requirement:
        bedroom_shortfall = preferences.minimum_bedrooms - listing.bedrooms
        area_shortfall = preferences.minimum_area_sqft - listing.area_sqft
        bedroom_label = "bedroom" if bedroom_shortfall == 1 else "bedrooms"

        considerations.append(
            f"{bedroom_shortfall} {bedroom_label} below the minimum "
            f"and {area_shortfall:,} sqft below the minimum area"
        )
    elif not meets_bedroom_requirement:
        strengths.append("Meets the minimum area requirement")

        bedroom_shortfall = preferences.minimum_bedrooms - listing.bedrooms
        bedroom_label = "bedroom" if bedroom_shortfall == 1 else "bedrooms"

        considerations.append(f"{bedroom_shortfall} {bedroom_label} below the minimum requirement")
    else:
        strengths.append("Meets the minimum bedroom requirement")

        area_shortfall = preferences.minimum_area_sqft - listing.area_sqft
        considerations.append(f"{area_shortfall:,} sqft below the minimum area")

    return RecommendationEvidence(
        strengths=tuple(strengths),
        considerations=tuple(considerations),
    )
