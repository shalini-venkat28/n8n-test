"""Default health check and readiness endpoints."""

from datetime import datetime, timezone

from botocore.exceptions import ClientError, BotoCoreError
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy.exc import DBAPIError, SQLAlchemyError

from neural_hub.models.schemas import HealthResponse, ReadinessResponse
from neural_hub.repositories.database import get_engine
from neural_hub.repositories.s3_repository import S3Repository
from neural_hub.settings import APP_NAME, APP_VERSION
from neural_hub.utils.logger import logger

router = APIRouter(tags=["Health"])


@router.get("/")
async def root():
    """Root endpoint returning app name and version."""
    return {"app": APP_NAME, "version": APP_VERSION}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness probe — returns 200 if the application is running."""
    return HealthResponse(status="healthy", timestamp=datetime.now(timezone.utc))


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    status_code=200,
    responses={
        200: {"description": "All dependencies are reachable"},
        503: {"description": "One or more dependencies are unreachable", "model": ReadinessResponse},
    },
)
async def readiness_check():
    """Readiness probe — checks database and S3 connectivity."""
    checks = {"database": "disconnected", "s3": "inaccessible"}

    # Check database connectivity
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        checks["database"] = "connected"
    except DBAPIError as e:
        logger.warning("Database readiness check failed: connection error", extra={"error": str(e)})
    except SQLAlchemyError as e:
        logger.warning("Database readiness check failed: SQLAlchemy error", extra={"error": str(e)})
    except OSError as e:
        logger.warning("Database readiness check failed: network error", extra={"error": str(e)})
    except Exception as e:
        logger.error("Database readiness check failed: unexpected error", extra={"error": type(e).__name__})

    # Check S3 connectivity
    try:
        s3_repo = S3Repository()
        if await s3_repo.check_accessibility():
            checks["s3"] = "accessible"
    except ClientError as e:
        logger.warning("S3 readiness check failed: client error", extra={"error_code": e.response["Error"]["Code"]})
    except BotoCoreError as e:
        logger.warning("S3 readiness check failed: AWS SDK error", extra={"error": str(e)})
    except OSError as e:
        logger.warning("S3 readiness check failed: network error", extra={"error": str(e)})
    except Exception as e:
        logger.error("S3 readiness check failed: unexpected error", extra={"error": type(e).__name__})

    all_ready = checks["database"] == "connected" and checks["s3"] == "accessible"
    status_code = 200 if all_ready else 503

    response = ReadinessResponse(
        status="ready" if all_ready else "not_ready",
        timestamp=datetime.now(timezone.utc),
        checks=checks,
    )
    return JSONResponse(content=response.model_dump(mode="json"), status_code=status_code)
