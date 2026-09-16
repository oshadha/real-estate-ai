import asyncio

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


def _explanation(reference: str) -> RecommendationExplanation:
    return RecommendationExplanation(
        summary=f"Explanation for {reference}",
        strengths=[],
        considerations=[],
    )


class CountingGenerator:
    def __init__(self) -> None:
        self.call_count = 0

    async def generate(
        self,
        match: PropertyMatch,
        preferences: BuyerPreferences,
    ) -> RecommendationExplanation:
        self.call_count += 1

        return _explanation(match.listing.reference)


class BlockingCountingGenerator:
    def __init__(self) -> None:
        self.call_count = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(
        self,
        match: PropertyMatch,
        preferences: BuyerPreferences,
    ) -> RecommendationExplanation:
        self.call_count += 1
        self.started.set()

        await self.release.wait()

        return _explanation(match.listing.reference)


class FailOnceGenerator:
    def __init__(self) -> None:
        self.call_count = 0

    async def generate(
        self,
        match: PropertyMatch,
        preferences: BuyerPreferences,
    ) -> RecommendationExplanation:
        self.call_count += 1

        if self.call_count == 1:
            raise RuntimeError("Generation failed")

        return _explanation(match.listing.reference)


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
async def test_concurrent_identical_requests_share_generation() -> None:
    inner = BlockingCountingGenerator()
    generator = CachingRecommendationExplanationGenerator(inner)
    match = _match("DXB-1001")
    preferences = _preferences()

    tasks = [
        asyncio.create_task(
            generator.generate(
                match,
                preferences,
            )
        )
        for _ in range(10)
    ]

    await inner.started.wait()
    await asyncio.sleep(0)

    assert inner.call_count == 1

    inner.release.set()

    explanations = await asyncio.gather(*tasks)

    assert inner.call_count == 1
    assert all(explanation.summary == "Explanation for DXB-1001" for explanation in explanations)
    assert len({id(explanation) for explanation in explanations}) == 10


@pytest.mark.anyio
async def test_failed_generation_is_not_cached() -> None:
    inner = FailOnceGenerator()
    generator = CachingRecommendationExplanationGenerator(inner)
    match = _match("DXB-1001")
    preferences = _preferences()

    with pytest.raises(
        RuntimeError,
        match="Generation failed",
    ):
        await generator.generate(
            match,
            preferences,
        )

    explanation = await generator.generate(
        match,
        preferences,
    )

    assert inner.call_count == 2
    assert explanation.summary == "Explanation for DXB-1001"


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


def test_cache_rejects_non_positive_ttl() -> None:
    with pytest.raises(
        ValueError,
        match="ttl_seconds must be positive",
    ):
        CachingRecommendationExplanationGenerator(
            CountingGenerator(),
            ttl_seconds=0,
        )


def test_cache_rejects_non_positive_max_entries() -> None:
    with pytest.raises(
        ValueError,
        match="max_entries must be at least 1",
    ):
        CachingRecommendationExplanationGenerator(
            CountingGenerator(),
            max_entries=0,
        )
