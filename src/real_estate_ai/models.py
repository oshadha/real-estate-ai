from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PropertyListing:
    reference: str
    location: str
    price_aed: int
    bedrooms: int
    area_sqft: int

    def __post_init__(self) -> None:
        if not self.reference.strip():
            raise ValueError("reference is required")

        if not self.location.strip():
            raise ValueError("location is required")

        if self.price_aed <= 0:
            raise ValueError("price_aed must be greater than zero")

        if self.bedrooms < 0:
            raise ValueError("bedrooms cannot be negative")

        if self.area_sqft <= 0:
            raise ValueError("area_sqft must be greater than zero")


@dataclass(frozen=True, slots=True)
class BuyerPreferences:
    preferred_location: str
    max_price_aed: int
    minimum_bedrooms: int
    minimum_area_sqft: int

    def __post_init__(self) -> None:
        if not self.preferred_location.strip():
            raise ValueError("preferred_location is required")

        if self.max_price_aed <= 0:
            raise ValueError("max_price_aed must be greater than zero")

        if self.minimum_bedrooms < 0:
            raise ValueError("minimum_bedrooms cannot be negative")

        if self.minimum_area_sqft <= 0:
            raise ValueError("minimum_area_sqft must be greater than zero")


@dataclass(frozen=True, slots=True)
class PropertyMatch:
    listing: PropertyListing
    score: float
