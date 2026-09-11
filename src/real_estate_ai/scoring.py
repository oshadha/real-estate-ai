from real_estate_ai.models import BuyerPreferences, PropertyListing


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
