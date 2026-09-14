from real_estate_ai.models import (
    BuyerPreferences,
    PropertyListing,
)
from real_estate_ai.scoring import calculate_match_score


def main() -> None:
    listing = PropertyListing(
        reference="DXB-1001",
        location="Dubai Marina",
        price_aed=1_150_000,
        bedrooms=2,
        area_sqft=920,
    )

    preferences = BuyerPreferences(
        preferred_location="Dubai Marina",
        max_price_aed=1_100_000,
        minimum_bedrooms=2,
        minimum_area_sqft=1_000,
    )

    score = calculate_match_score(
        listing,
        preferences,
    )

    print(f"Property: {listing.reference}")
    print(f"Match score: {score:.2f}/100")
