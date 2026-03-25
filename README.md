# Neural Hub — Kiro Template Integration API

FastAPI service that generates ML project templates by querying PostgreSQL for project metadata, orchestrating `requirements.md` and `model_details.md` documents, and uploading customized templates to S3 for Kiro IDE integration.

## Project Structure

```
src/neural_hub/
├── main.py                  # FastAPI app entry point
├── settings.py              # Pydantic Settings + constants
├── models/
│   ├── database.py          # SQLAlchemy ORM models
│   └── schemas.py           # Pydantic request/response models
├── repositories/
│   ├── database.py          # Async session management
│   ├── generation_repository.py
│   ├── model_recommendation_repository.py
│   └── s3_repository.py
├── routes/
│   ├── default.py           # Health/readiness endpoints
│   └── template.py          # Template generation endpoints
├── services/
│   ├── dependencies.py      # FastAPI dependency injection
│   ├── document_service.py  # Markdown document orchestration
│   ├── kiro_service.py      # Kiro CLI/API/deep-link integration
│   └── template_service.py  # Workflow orchestration
└── utils/
    ├── logger.py            # Structured JSON logging
    └── exceptions/
        ├── error_codes.py
        ├── error_responses.py
        └── exceptions.py
```

## Setup

```bash
# Install dependencies
make setup

# Copy and configure environment
cp .env.sample .env
# Edit .env with your database and AWS credentials
```

## Run

```bash
make run
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/template-download` | Generate template, return S3 URI |
| POST | `/api/v1/kiro-integration` | Generate template + open in Kiro |
| GET | `/health` | Liveness probe |
| GET | `/ready` | Readiness probe (checks DB + S3) |

## Development

```bash
make lint          # Ruff linter
make format        # Ruff formatter
make test          # Run tests
make test-cov      # Tests with coverage
make clean         # Remove caches
```

## Docker

```bash
docker build -t neural-hub .
docker run -p 8000:8000 --env-file .env neural-hub
```
