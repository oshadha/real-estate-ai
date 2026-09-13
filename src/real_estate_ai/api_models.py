from pydantic import BaseModel, ConfigDict, Field

from real_estate_ai.models import BuyerPreferences, PropertyListing


class ApiModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        strict=True,
    )


class PropertyListingRequest(ApiModel):
    reference: str = Field(min_length=1, max_length=50)
    location: str = Field(min_length=1, max_length=100)
    price_aed: int = Field(gt=0)
    bedrooms: int = Field(ge=0)
    area_sqft: int = Field(gt=0)

    def to_domain(self) -> PropertyListing:
        return PropertyListing(
            reference=self.reference,
            location=self.location,
            price_aed=self.price_aed,
            bedrooms=self.bedrooms,
            area_sqft=self.area_sqft,
        )


class BuyerPreferencesRequest(ApiModel):
    preferred_location: str = Field(min_length=1, max_length=100)
    max_price_aed: int = Field(gt=0)
    minimum_bedrooms: int = Field(ge=0)
    minimum_area_sqft: int = Field(gt=0)

    def to_domain(self) -> BuyerPreferences:
        return BuyerPreferences(
            preferred_location=self.preferred_location,
            max_price_aed=self.max_price_aed,
            minimum_bedrooms=self.minimum_bedrooms,
            minimum_area_sqft=self.minimum_area_sqft,
        )


class RecommendationRequest(ApiModel):
    preferences: BuyerPreferencesRequest
    limit: int = Field(default=3, ge=1, le=10)


class PropertyListingResponse(ApiModel):
    reference: str
    location: str
    price_aed: int
    bedrooms: int
    area_sqft: int


class RecommendationExplanationResponse(ApiModel):
    summary: str = Field(min_length=1, max_length=300)
    strengths: list[str] = Field(max_length=3)
    considerations: list[str] = Field(max_length=3)


class PropertyRecommendationResponse(ApiModel):
    reference: str
    location: str
    price_aed: int
    score: float
    explanation: RecommendationExplanationResponse
