from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    openai_api_key: SecretStr = Field(min_length=1)
    openai_model: str = Field(min_length=1)

    openai_max_concurrency: int = Field(
        default=2,
        ge=1,
        le=20,
    )

    explanation_cache_ttl_seconds: int = Field(
        default=300,
        ge=1,
        le=86_400,
    )

    explanation_cache_max_entries: int = Field(
        default=1_000,
        ge=1,
        le=100_000,
    )
