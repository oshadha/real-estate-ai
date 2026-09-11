import pytest

from real_estate_ai.models import PropertyListing
from real_estate_ai.repositories import InMemoryPropertyRepository


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_repository_returns_property_listings() -> None:
    listing = PropertyListing(
        reference="DXB-1001",
        location="Dubai Marina",
        price_aed=1_150_000,
        bedrooms=2,
        area_sqft=1_100,
    )
    repository = InMemoryPropertyRepository([listing])

    listings = await repository.get_all()

    assert listings == [listing]


@pytest.mark.anyio
async def test_returned_collection_cannot_modify_repository() -> None:
    listing = PropertyListing(
        reference="DXB-1001",
        location="Dubai Marina",
        price_aed=1_150_000,
        bedrooms=2,
        area_sqft=1_100,
    )
    repository = InMemoryPropertyRepository([listing])

    listings = await repository.get_all()
    listings.clear()

    assert await repository.get_all() == [listing]
