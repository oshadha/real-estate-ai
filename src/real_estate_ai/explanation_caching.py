import asyncio
import logging
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

from real_estate_ai.explanations import (
    RecommendationExplanationGenerator,
)
from real_estate_ai.models import BuyerPreferences, PropertyMatch
from real_estate_ai.structured_outputs import RecommendationExplanation

logger = logging.getLogger(__name__)

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
        self._in_flight: dict[
            CacheKey,
            asyncio.Task[RecommendationExplanation],
        ] = {}

        self._lock = asyncio.Lock()

    async def generate(
        self,
        match: PropertyMatch,
        preferences: BuyerPreferences,
    ) -> RecommendationExplanation:
        key = (match, preferences)

        async with self._lock:
            entry = self._cache.get(key)

            if entry is not None:
                if entry.expires_at > self._clock():
                    self._cache.move_to_end(key)

                    logger.info(
                        "Recommendation explanation cache hit",
                        extra={
                            "event_name": "explanation_cache.hit",
                            "property_reference": match.listing.reference,
                        },
                    )

                    return entry.explanation.model_copy(deep=True)

                del self._cache[key]

                logger.info(
                    "Recommendation explanation cache entry expired",
                    extra={
                        "event_name": "explanation_cache.expired",
                        "property_reference": match.listing.reference,
                    },
                )

            task = self._in_flight.get(key)

            if task is None:
                logger.info(
                    "Recommendation explanation cache miss",
                    extra={
                        "event_name": "explanation_cache.miss",
                        "property_reference": match.listing.reference,
                    },
                )

                task = asyncio.create_task(
                    self._generate_and_cache(
                        key,
                        match,
                        preferences,
                    )
                )
                self._in_flight[key] = task
            else:
                logger.info(
                    "Joining in-flight recommendation generation",
                    extra={
                        "event_name": "explanation_cache.in_flight_join",
                        "property_reference": match.listing.reference,
                    },
                )

        explanation = await asyncio.shield(task)

        return explanation.model_copy(deep=True)

    async def _generate_and_cache(
        self,
        key: CacheKey,
        match: PropertyMatch,
        preferences: BuyerPreferences,
    ) -> RecommendationExplanation:
        try:
            explanation = await self._inner.generate(
                match,
                preferences,
            )

            async with self._lock:
                self._cache[key] = CacheEntry(
                    explanation=explanation.model_copy(deep=True),
                    expires_at=self._clock() + self._ttl_seconds,
                )
                self._cache.move_to_end(key)

                while len(self._cache) > self._max_entries:
                    evicted_key, _ = self._cache.popitem(last=False)

                    logger.info(
                        "Recommendation explanation cache entry evicted",
                        extra={
                            "event_name": "explanation_cache.evicted",
                            "property_reference": (evicted_key[0].listing.reference),
                        },
                    )

            return explanation
        finally:
            async with self._lock:
                self._in_flight.pop(key, None)
