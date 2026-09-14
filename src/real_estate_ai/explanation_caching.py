import asyncio
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

from real_estate_ai.explanations import (
    RecommendationExplanationGenerator,
)
from real_estate_ai.models import BuyerPreferences, PropertyMatch
from real_estate_ai.structured_outputs import RecommendationExplanation

type CacheKey = tuple[
    PropertyMatch,
    BuyerPreferences,
]


@dataclass(frozen=True, slots=True)
class CacheEntry:
    explanation: RecommendationExplanation
    expires_at: float


class CachingRecommendationExplanationGenerator:
    def __init__(
        self,
        inner: RecommendationExplanationGenerator,
        *,
        ttl_seconds: float = 300,
        max_entries: int = 1_000,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")

        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")

        self._inner = inner
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._cache: OrderedDict[CacheKey, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

    async def generate(
        self,
        match: PropertyMatch,
        preferences: BuyerPreferences,
    ) -> RecommendationExplanation:
        key = (match, preferences)
        current_time = self._clock()

        async with self._lock:
            entry = self._cache.get(key)

            if entry is not None:
                if entry.expires_at > current_time:
                    self._cache.move_to_end(key)

                    return entry.explanation.model_copy(deep=True)

                del self._cache[key]

        explanation = await self._inner.generate(
            match,
            preferences,
        )

        async with self._lock:
            self._cache[key] = CacheEntry(
                explanation=explanation.model_copy(deep=True),
                expires_at=(self._clock() + self._ttl_seconds),
            )
            self._cache.move_to_end(key)

            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)

        return explanation
