from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from real_estate_ai.api import (
    app,
    get_property_repository,
    get_recommendation_explainer,
)
from real_estate_ai.explanations import (
    RecommendationExplainer,
    TemplateRecommendationExplainer,
)
from real_estate_ai.models import (
    BuyerPreferences,
    PropertyListing,
    PropertyMatch,
)
from real_estate_ai.repositories import (
    InMemoryPropertyRepository,
    PropertyRepository,
)
from real_estate_ai.structured_outputs import RecommendationExplanation

client = TestClient(app)


def override_recommendation_explainer() -> RecommendationExplainer:
    return TemplateRecommendationExplainer()


@pytest.fixture(autouse=True)
def use_template_recommendation_explainer() -> Iterator[None]:
    app.dependency_overrides[get_recommendation_explainer] = override_recommendation_explainer

    try:
        yield
    finally:
        app.dependency_overrides.pop(
            get_recommendation_explainer,
            None,
        )


class StubRecommendationExplainer:
    async def explain(
        self,
        match: PropertyMatch,
        preferences: BuyerPreferences,
    ) -> RecommendationExplanation:
        return RecommendationExplanation(
            summary=f"Test explanation for {match.listing.reference}",
            strengths=["Test strength"],
            considerations=[],
        )


def _valid_request() -> dict[str, Any]:
    return {
        "preferences": {
            "preferred_location": "Dubai Marina",
            "max_price_aed": 1_100_000,
            "minimum_bedrooms": 2,
            "minimum_area_sqft": 1_000,
        }
    }


def _recommendation_repository() -> InMemoryPropertyRepository:
    return InMemoryPropertyRepository(
        [
            PropertyListing(
                reference="DXB-1001",
                location="Dubai Marina",
                price_aed=1_150_000,
                bedrooms=2,
                area_sqft=920,
            ),
            PropertyListing(
                reference="DXB-1002",
                location="JVC",
                price_aed=950_000,
                bedrooms=2,
                area_sqft=1_100,
            ),
        ]
    )


def test_health_endpoint_returns_healthy() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_recommendations_are_returned_in_ranked_order() -> None:
    repository = _recommendation_repository()

    explainer = StubRecommendationExplainer()

    def override_property_repository() -> PropertyRepository:
        return repository

    def override_recommendation_explainer() -> RecommendationExplainer:
        return explainer

    app.dependency_overrides[get_property_repository] = override_property_repository

    app.dependency_overrides[get_recommendation_explainer] = override_recommendation_explainer

    try:
        response = client.post(
            "/recommendations",
            json=_valid_request(),
        )
    finally:
        app.dependency_overrides.pop(get_property_repository, None)
        app.dependency_overrides.pop(get_recommendation_explainer, None)

    assert response.status_code == 200

    body = response.json()

    assert [item["reference"] for item in body] == [
        "DXB-1001",
        "DXB-1002",
    ]
    assert body[0]["score"] == 91.45
    assert body[1]["score"] == 90.0
    assert body[0]["explanation"] == {
        "summary": "Test explanation for DXB-1001",
        "strengths": ["Test strength"],
        "considerations": [],
    }


def test_invalid_budget_returns_validation_error() -> None:
    request = _valid_request()
    request["preferences"]["max_price_aed"] = 0

    response = client.post(
        "/recommendations",
        json=request,
    )

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(error["loc"] == ["body", "preferences", "max_price_aed"] for error in errors)


def test_properties_are_loaded_from_injected_repository() -> None:
    repository = InMemoryPropertyRepository(
        [
            PropertyListing(
                reference="TEST-1001",
                location="Downtown Dubai",
                price_aed=2_000_000,
                bedrooms=3,
                area_sqft=1_500,
            )
        ]
    )

    def override_property_repository() -> PropertyRepository:
        return repository

    app.dependency_overrides[get_property_repository] = override_property_repository

    try:
        response = client.get("/properties")
    finally:
        app.dependency_overrides.pop(get_property_repository, None)

    assert response.status_code == 200
    assert response.json() == [
        {
            "reference": "TEST-1001",
            "location": "Downtown Dubai",
            "price_aed": 2_000_000,
            "bedrooms": 3,
            "area_sqft": 1_500,
        }
    ]
