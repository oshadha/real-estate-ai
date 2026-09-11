from collections.abc import Iterable

from real_estate_ai.models import (
    BuyerPreferences,
    PropertyListing,
    PropertyMatch,
)
from real_estate_ai.scoring import calculate_match_score


def rank_properties(
    listings: Iterable[PropertyListing],
    preferences: BuyerPreferences,
) -> list[PropertyMatch]:
    matches: list[PropertyMatch] = []

    for listing in listings:
        score = calculate_match_score(listing, preferences)
        matches.append(PropertyMatch(listing=listing, score=score))

    return sorted(
        matches,
        key=lambda match: match.score,
        reverse=True,
    )
