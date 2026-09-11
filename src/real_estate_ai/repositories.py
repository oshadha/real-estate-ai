from collections.abc import Iterable
from typing import Protocol

from real_estate_ai.models import PropertyListing


class PropertyRepository(Protocol):
    async def get_all(self) -> list[PropertyListing]: ...


class InMemoryPropertyRepository:
    def __init__(self, listings: Iterable[PropertyListing]) -> None:
        self._listings = list(listings)

    async def get_all(self) -> list[PropertyListing]:
        return list(self._listings)
