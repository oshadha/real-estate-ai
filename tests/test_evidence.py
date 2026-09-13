from real_estate_ai.evidence import build_recommendation_evidence
from real_estate_ai.models import (
    BuyerPreferences,
    PropertyListing,
    PropertyMatch,
)


def _preferences() -> BuyerPreferences:
    return BuyerPreferences(
        preferred_location="Dubai Marina",
        max_price_aed=1_100_000,
        minimum_bedrooms=2,
        minimum_area_sqft=1_000,
    )


def test_evidence_identifies_budget_consideration() -> None:
    listing = PropertyListing(
        reference="DXB-1001",
        location="Dubai Marina",
        price_aed=1_150_000,
        bedrooms=2,
        area_sqft=1_100,
    )
    match = PropertyMatch(
        listing=listing,
        score=95.45,
    )

    evidence = build_recommendation_evidence(
        match,
        _preferences(),
    )

    assert evidence.strengths == (
        "Matches the preferred location",
        "Meets the minimum bedroom and area requirements",
    )
    assert evidence.considerations == ("AED 50,000 over budget",)


def test_evidence_identifies_location_consideration() -> None:
    listing = PropertyListing(
        reference="DXB-1002",
        location="JVC",
        price_aed=950_000,
        bedrooms=2,
        area_sqft=1_000,
    )
    match = PropertyMatch(
        listing=listing,
        score=90.0,
    )

    evidence = build_recommendation_evidence(
        match,
        _preferences(),
    )

    assert evidence.strengths == (
        "Within the buyer's budget",
        "Meets the minimum bedroom and area requirements",
    )
    assert evidence.considerations == ("Located in JVC instead of Dubai Marina",)


def test_evidence_handles_partially_met_space_requirements() -> None:
    listing = PropertyListing(
        reference="DXB-1003",
        location="Dubai Marina",
        price_aed=1_150_000,
        bedrooms=2,
        area_sqft=920,
    )
    match = PropertyMatch(
        listing=listing,
        score=91.45,
    )

    evidence = build_recommendation_evidence(
        match,
        _preferences(),
    )

    assert evidence.strengths == (
        "Matches the preferred location",
        "Meets the minimum bedroom requirement",
    )
    assert evidence.considerations == (
        "AED 50,000 over budget",
        "80 sqft below the minimum area",
    )
