from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

ExplanationPoint = Annotated[
    str,
    Field(min_length=1, max_length=200),
]


class RecommendationExplanation(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        strict=True,
    )

    summary: str = Field(min_length=1, max_length=300)
    strengths: list[ExplanationPoint] = Field(
        min_length=1,
        max_length=3,
    )
    considerations: list[ExplanationPoint] = Field(
        default_factory=list,
        max_length=3,
    )
