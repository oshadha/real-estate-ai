import uvicorn

from real_estate_ai.logging_config import configure_logging


def main() -> None:
    configure_logging()

    uvicorn.run(
        "real_estate_ai.api:app",
        host="127.0.0.1",
        port=8000,
        log_config=None,
    )
