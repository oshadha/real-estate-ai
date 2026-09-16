# Real Estate AI

A production-oriented property recommendation API built with Python, FastAPI,
Pydantic, and OpenAI.

The application combines deterministic business rules with grounded AI-generated
summaries. Python owns scoring and evidence classification; the language model
only converts verified facts into natural language.

## Features

- Deterministic property scoring and ranking
- Async repository abstraction
- FastAPI dependency injection
- Pydantic request and response validation
- OpenAI Structured Outputs
- Deterministic strengths and considerations
- Retry and template fallback
- Bounded TTL/LRU explanation cache
- Single-flight duplicate-request protection
- OpenAI concurrency limiting
- Correlation IDs
- Structured JSON logging
- Liveness and readiness endpoints
- Strict mypy type checking
- Ruff formatting and linting
- Automated pytest suite
- Non-root production Docker image

## Architecture

```text
FastAPI endpoint
    -> Repository
    -> Deterministic ranking
    -> Concurrency limiter
    -> Retry and fallback
    -> Successful-result cache
    -> Structured OpenAI generator
    -> Grounded explanation response
```

## AI responsibility boundary

The application calculates these values deterministically:

- Match score
- Budget status
- Location match
- Bedroom match
- Area match
- Strengths
- Considerations

The language model generates only the natural-language summary from verified
facts. It does not calculate scores or classify evidence.

## Requirements

- Python 3.13+
- uv
- OpenAI API key
- Docker Desktop, if running the container

## Local setup

Clone the repository and install dependencies:

```powershell
uv sync
```

Copy the example configuration:

```powershell
Copy-Item .env.example .env
```

Replace the placeholder API key in `.env` with your actual key. Never commit
`.env`.

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Run locally

```powershell
python -m real_estate_ai.server
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Operational endpoints

Liveness:

```http
GET /health
```

Readiness:

```http
GET /ready
```

Readiness validates application configuration and repository access. It does
not call OpenAI.

## Request recommendations

```powershell
$body = @{
    preferences = @{
        preferred_location = "Dubai Marina"
        max_price_aed = 1100000
        minimum_bedrooms = 2
        minimum_area_sqft = 1000
    }
    limit = 2
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/recommendations" `
    -Headers @{ "X-Request-ID" = "example-request-001" } `
    -ContentType "application/json" `
    -Body $body
```

## Quality checks

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

On Windows systems that block generated command launchers, use:

```powershell
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest
```

## Docker

Build the image:

```powershell
docker build -t real-estate-ai:0.1.0 .
```

Run it using the local environment file:

```powershell
docker run `
    --rm `
    --name real-estate-ai `
    --env-file .env `
    --env APP_HOST=0.0.0.0 `
    --publish 8000:8000 `
    real-estate-ai:0.1.0
```

Test the container:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
```

Never bake `.env` or an API key into a container image. Production deployments
should obtain secrets from an approved secret manager.

## Tests

The test suite covers:

- Domain validation
- Scoring and ranking
- Repository isolation
- API validation and dependency injection
- Structured AI outputs
- Evidence grounding
- Retry and fallback behavior
- OpenAI adapter behavior
- Safe structured logging
- Correlation propagation
- Concurrency limiting
- Cache expiration and eviction
- Single-flight generation
- Liveness and readiness

## Current scope

This project is a production-oriented portfolio API. The current repository is
in-memory and the cache is local to one process.

Potential future extensions include:

- Persistent property storage
- Authentication and authorization
- Redis distributed caching
- Distributed rate limiting
- Metrics and tracing
- Cloud deployment