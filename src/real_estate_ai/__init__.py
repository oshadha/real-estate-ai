from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PropertyListing:
    reference: str
    location: str
    price_aed: int
    bedrooms: int
    area_sqft: int


@dataclass(frozen=True, slots=True)
class BuyerPreferences:
    preferred_location: str
    max_price_aed: int
    minimum_bedrooms: int
    minimum_area_sqft: int


def calculate_match_score(
    listing: PropertyListing,
    preferences: BuyerPreferences,
) -> float:
    score = 100.0

    if listing.price_aed > preferences.max_price_aed:
        excess_ratio = (listing.price_aed - preferences.max_price_aed) / preferences.max_price_aed

        score -= min(excess_ratio * 100, 40)

    if listing.bedrooms < preferences.minimum_bedrooms:
        missing_bedrooms = preferences.minimum_bedrooms - listing.bedrooms
        score -= missing_bedrooms * 15

    if listing.area_sqft < preferences.minimum_area_sqft:
        missing_area_ratio = (
            preferences.minimum_area_sqft - listing.area_sqft
        ) / preferences.minimum_area_sqft

        score -= min(missing_area_ratio * 50, 20)

    if listing.location.casefold() != preferences.preferred_location.casefold():
        score -= 10

    return round(max(score, 0.0), 2)


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

    score = calculate_match_score(listing, preferences)

    print(f"Property: {listing.reference}")
    print(f"Match score: {score:.2f}/100")
