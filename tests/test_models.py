import pytest

from real_estate_ai.models import BuyerPreferences, PropertyListing


def test_listing_rejects_non_positive_price() -> None:
    with pytest.raises(
        ValueError,
        match="price_aed must be greater than zero",
    ):
        PropertyListing(
            reference="DXB-1001",
            location="Dubai Marina",
            price_aed=0,
            bedrooms=2,
            area_sqft=1_000,
        )


def test_preferences_reject_non_positive_budget() -> None:
    with pytest.raises(
        ValueError,
        match="max_price_aed must be greater than zero",
    ):
        BuyerPreferences(
            preferred_location="Dubai Marina",
            max_price_aed=0,
            minimum_bedrooms=2,
            minimum_area_sqft=1_000,
        )


def test_listing_rejects_blank_reference() -> None:
    with pytest.raises(
        ValueError,
        match="reference is required",
    ):
        PropertyListing(
            reference="   ",
            location="Dubai Marina",
            price_aed=1_000_000,
            bedrooms=2,
            area_sqft=1_000,
        )
