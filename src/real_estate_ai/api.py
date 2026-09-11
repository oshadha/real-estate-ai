import asyncio
from typing import Annotated

from fastapi import Depends, FastAPI

from real_estate_ai.api_models import (
    PropertyListingResponse,
    PropertyRecommendationResponse,
    RecommendationRequest,
)
from real_estate_ai.explanations import (
    RecommendationExplainer,
    TemplateRecommendationExplainer,
)
from real_estate_ai.models import PropertyListing
from real_estate_ai.recommendations import rank_properties
from real_estate_ai.repositories import (
    InMemoryPropertyRepository,
    PropertyRepository,
)

app = FastAPI(
    title="Real Estate AI API",
    version="0.1.0",
)

property_repository: PropertyRepository = InMemoryPropertyRepository(
    [
        PropertyListing(
            reference="DXB-1001",
            location="Dubai Marina",
            price_aed=1_150_000,
            bedrooms=2,
            area_sqft=1_100,
        ),
        PropertyListing(
            reference="DXB-1002",
            location="JVC",
            price_aed=950_000,
            bedrooms=2,
            area_sqft=1_000,
        ),
    ]
)

recommendation_explainer: RecommendationExplainer = TemplateRecommendationExplainer()


def get_recommendation_explainer() -> RecommendationExplainer:
    return recommendation_explainer


RecommendationExplainerDependency = Annotated[
    RecommendationExplainer,
    Depends(get_recommendation_explainer),
]


def get_property_repository() -> PropertyRepository:
    return property_repository


PropertyRepositoryDependency = Annotated[
    PropertyRepository,
    Depends(get_property_repository),
]


@app.get("/health", tags=["Operations"])
async def get_health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post(
    "/recommendations",
    response_model=list[PropertyRecommendationResponse],
    tags=["Recommendations"],
)
async def create_recommendations(
    request: RecommendationRequest,
    repository: PropertyRepositoryDependency,
    explainer: RecommendationExplainerDependency,
) -> list[PropertyRecommendationResponse]:
    listings = await repository.get_all()
    preferences = request.preferences.to_domain()

    matches = rank_properties(listings, preferences)[: request.limit]

    explanations = await asyncio.gather(
        *(explainer.explain(match, preferences) for match in matches)
    )

    return [
        PropertyRecommendationResponse(
            reference=match.listing.reference,
            location=match.listing.location,
            price_aed=match.listing.price_aed,
            score=match.score,
            explanation=explanation,
        )
        for match, explanation in zip(
            matches,
            explanations,
            strict=True,
        )
    ]


@app.get(
    "/properties",
    response_model=list[PropertyListingResponse],
    tags=["Properties"],
)
async def get_properties(
    repository: PropertyRepositoryDependency,
) -> list[PropertyListingResponse]:
    listings = await repository.get_all()

    return [
        PropertyListingResponse(
            reference=listing.reference,
            location=listing.location,
            price_aed=listing.price_aed,
            bedrooms=listing.bedrooms,
            area_sqft=listing.area_sqft,
        )
        for listing in listings
    ]
