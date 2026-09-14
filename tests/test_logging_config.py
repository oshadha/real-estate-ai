import json
import logging

from real_estate_ai.logging_config import JsonFormatter


def test_json_formatter_includes_only_safe_metadata() -> None:
    record = logging.LogRecord(
        name="real_estate_ai.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="OpenAI request completed",
        args=(),
        exc_info=None,
    )
    record.__dict__.update(
        {
            "event_name": "openai.request.completed",
            "correlation_id": "test-request-123",
            "model": "gpt-5.6-luna",
            "response_id": "resp-test-123",
            "duration_ms": 125.0,
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
            "user_message": "Private buyer details",
            "api_key": "secret-api-key",
        }
    )

    payload: dict[str, object] = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["event_name"] == "openai.request.completed"
    assert payload["correlation_id"] == "test-request-123"
    assert payload["model"] == "gpt-5.6-luna"
    assert payload["response_id"] == "resp-test-123"
    assert payload["duration_ms"] == 125.0
    assert payload["input_tokens"] == 120
    assert payload["output_tokens"] == 30
    assert payload["total_tokens"] == 150

    assert "user_message" not in payload
    assert "api_key" not in payload
