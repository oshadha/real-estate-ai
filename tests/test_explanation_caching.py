import pytest

from real_estate_ai.explanation_caching import (
    CachingRecommendationExplanationGenerator,
)
from real_estate_ai.models import (
    BuyerPreferences,
    PropertyListing,
    PropertyMatch,
)
from real_estate_ai.structured_outputs import RecommendationExplanation


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class CountingGenerator:
    def __init__(self) -> None:
        self.call_count = 0

    async def generate(
        self,
        match: PropertyMatch,
        preferences: BuyerPreferences,
    ) -> RecommendationExplanation:
        self.call_count += 1

        return RecommendationExplanation(
            summary=f"Explanation for {match.listing.reference}",
            strengths=[],
            considerations=[],
        )


class FakeClock:
    def __init__(self) -> None:
        self.current_time = 0.0

    def __call__(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += seconds


def _match(reference: str) -> PropertyMatch:
    return PropertyMatch(
        listing=PropertyListing(
            reference=reference,
            location="Dubai Marina",
            price_aed=1_150_000,
            bedrooms=2,
            area_sqft=1_100,
        ),
        score=95.45,
    )


def _preferences() -> BuyerPreferences:
    return BuyerPreferences(
        preferred_location="Dubai Marina",
        max_price_aed=1_100_000,
        minimum_bedrooms=2,
        minimum_area_sqft=1_000,
    )


@pytest.mark.anyio
async def test_repeated_explanation_uses_cache() -> None:
    inner = CountingGenerator()
    generator = CachingRecommendationExplanationGenerator(inner)

    first = await generator.generate(
        _match("DXB-1001"),
        _preferences(),
    )
    second = await generator.generate(
        _match("DXB-1001"),
        _preferences(),
    )

    assert inner.call_count == 1
    assert first == second
    assert first is not second


@pytest.mark.anyio
async def test_expired_explanation_is_regenerated() -> None:
    clock = FakeClock()
    inner = CountingGenerator()
    generator = CachingRecommendationExplanationGenerator(
        inner,
        ttl_seconds=60,
        clock=clock,
    )
    match = _match("DXB-1001")
    preferences = _preferences()

    await generator.generate(match, preferences)

    clock.advance(61)

    await generator.generate(match, preferences)

    assert inner.call_count == 2


@pytest.mark.anyio
async def test_oldest_entry_is_evicted_when_cache_is_full() -> None:
    inner = CountingGenerator()
    generator = CachingRecommendationExplanationGenerator(
        inner,
        max_entries=1,
    )
    preferences = _preferences()

    await generator.generate(
        _match("DXB-1001"),
        preferences,
    )
    await generator.generate(
        _match("DXB-1002"),
        preferences,
    )
    await generator.generate(
        _match("DXB-1001"),
        preferences,
    )

    assert inner.call_count == 3
