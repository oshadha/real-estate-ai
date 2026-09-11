from real_estate_ai.models import BuyerPreferences, PropertyListing
from real_estate_ai.recommendations import rank_properties


def test_properties_are_ranked_by_score_descending() -> None:
    listings = [
        PropertyListing("A", "Dubai Marina", 1_150_000, 2, 920),
        PropertyListing("B", "JVC", 950_000, 2, 1_100),
        PropertyListing("C", "Dubai Marina", 1_050_000, 1, 850),
    ]

    preferences = BuyerPreferences(
        preferred_location="Dubai Marina",
        max_price_aed=1_100_000,
        minimum_bedrooms=2,
        minimum_area_sqft=1_000,
    )

    matches = rank_properties(listings, preferences)

    references = [match.listing.reference for match in matches]

    assert references == ["A", "B", "C"]
