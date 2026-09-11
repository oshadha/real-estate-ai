from real_estate_ai.models import BuyerPreferences, PropertyListing
from real_estate_ai.recommendations import rank_properties


def main() -> None:
    listings = [
        PropertyListing(
            reference="DXB-1001",
            location="Dubai Marina",
            price_aed=1_150_000,
            bedrooms=2,
            area_sqft=920,
        ),
        PropertyListing(
            reference="DXB-1002",
            location="JVC",
            price_aed=950_000,
            bedrooms=2,
            area_sqft=1_100,
        ),
        PropertyListing(
            reference="DXB-1003",
            location="Dubai Marina",
            price_aed=1_050_000,
            bedrooms=1,
            area_sqft=850,
        ),
    ]

    preferences = BuyerPreferences(
        preferred_location="Dubai Marina",
        max_price_aed=1_100_000,
        minimum_bedrooms=2,
        minimum_area_sqft=1_000,
    )

    matches = rank_properties(listings, preferences)

    for position, match in enumerate(matches, start=1):
        print(f"{position}. {match.listing.reference}: {match.score:.2f}/100")
