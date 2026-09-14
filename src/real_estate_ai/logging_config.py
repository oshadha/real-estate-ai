import json
import logging
from datetime import UTC, datetime

_safe_extra_fields = (
    "event_name",
    "correlation_id",
    "model",
    "response_id",
    "duration_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "property_reference",
    "generation_source",
    "attempt_number",
    "attempt_count",
    "error_type",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for field_name in _safe_extra_fields:
            value = record.__dict__.get(field_name)

            if value is not None:
                payload[field_name] = value

        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )


def configure_logging(
    level: int = logging.INFO,
) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
