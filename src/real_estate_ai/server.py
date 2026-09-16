import os

import uvicorn

from real_estate_ai.logging_config import configure_logging


def main() -> None:
    configure_logging()

    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8000"))

    uvicorn.run(
        "real_estate_ai.api:app",
        host=host,
        port=port,
        log_config=None,
    )


if __name__ == "__main__":
    main()
