from real_estate_ai.models import BuyerPreferences, PropertyListing
from real_estate_ai.scoring import calculate_match_score


def test_perfect_match_returns_100() -> None:
    listing = PropertyListing(
        reference="DXB-1001",
        location="Dubai Marina",
        price_aed=1_000_000,
        bedrooms=2,
        area_sqft=1_100,
    )

    preferences = BuyerPreferences(
        preferred_location="Dubai Marina",
        max_price_aed=1_100_000,
        minimum_bedrooms=2,
        minimum_area_sqft=1_000,
    )

    result = calculate_match_score(listing, preferences)

    assert result == 100.0


def test_penalties_are_applied() -> None:
    listing = PropertyListing(
        reference="DXB-1002",
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

    result = calculate_match_score(listing, preferences)

    assert result == 91.45


def test_score_cannot_fall_below_zero() -> None:
    listing = PropertyListing(
        reference="DXB-1003",
        location="JVC",
        price_aed=10_000_000,
        bedrooms=0,
        area_sqft=1,
    )

    preferences = BuyerPreferences(
        preferred_location="Dubai Marina",
        max_price_aed=100_000,
        minimum_bedrooms=10,
        minimum_area_sqft=10_000,
    )

    result = calculate_match_score(listing, preferences)

    assert result == 0.0
