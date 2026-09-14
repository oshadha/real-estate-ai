import asyncio

import pytest

from real_estate_ai.explanations import (
    ConcurrencyLimitedRecommendationExplainer,
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


class TrackingExplainer:
    def __init__(self) -> None:
        self.active_count = 0
        self.maximum_active_count = 0
        self.two_calls_started = asyncio.Event()
        self.release_calls = asyncio.Event()

    async def explain(
        self,
        match: PropertyMatch,
        preferences: BuyerPreferences,
    ) -> RecommendationExplanation:
        self.active_count += 1
        self.maximum_active_count = max(
            self.maximum_active_count,
            self.active_count,
        )

        if self.active_count == 2:
            self.two_calls_started.set()

        try:
            await self.release_calls.wait()

            return RecommendationExplanation(
                summary=f"Explanation for {match.listing.reference}",
                strengths=[],
                considerations=[],
            )
        finally:
            self.active_count -= 1


def _match_and_preferences() -> tuple[
    PropertyMatch,
    BuyerPreferences,
]:
    listing = PropertyListing(
        reference="DXB-1001",
        location="Dubai Marina",
        price_aed=1_150_000,
        bedrooms=2,
        area_sqft=1_100,
    )
    preferences = BuyerPreferences(
        preferred_location="Dubai Marina",
        max_price_aed=1_100_000,
        minimum_bedrooms=2,
        minimum_area_sqft=1_000,
    )

    return (
        PropertyMatch(
            listing=listing,
            score=95.45,
        ),
        preferences,
    )


@pytest.mark.anyio
async def test_explainer_limits_concurrent_calls() -> None:
    inner = TrackingExplainer()
    explainer = ConcurrencyLimitedRecommendationExplainer(
        inner,
        max_concurrency=2,
    )
    match, preferences = _match_and_preferences()

    tasks = [
        asyncio.create_task(
            explainer.explain(
                match,
                preferences,
            )
        )
        for _ in range(3)
    ]

    try:
        await asyncio.wait_for(
            inner.two_calls_started.wait(),
            timeout=1,
        )

        assert inner.maximum_active_count == 2
    finally:
        inner.release_calls.set()
        results = await asyncio.gather(*tasks)

    assert len(results) == 3
    assert inner.maximum_active_count == 2


def test_explainer_rejects_invalid_concurrency() -> None:
    with pytest.raises(
        ValueError,
        match="max_concurrency must be at least 1",
    ):
        ConcurrencyLimitedRecommendationExplainer(
            TrackingExplainer(),
            max_concurrency=0,
        )
