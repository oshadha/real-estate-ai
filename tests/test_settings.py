import pytest

from real_estate_ai.settings import Settings


def test_settings_loads_openai_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-luna")
    monkeypatch.setenv("OPENAI_MAX_CONCURRENCY", "4")

    settings = Settings()

    assert settings.openai_api_key.get_secret_value() == "test-api-key"
    assert settings.openai_model == "gpt-5.6-luna"
    assert settings.openai_max_concurrency == 4
    assert "test-api-key" not in repr(settings)
