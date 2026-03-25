"""Default health check and readiness endpoints."""

from datetime import datetime, timezone
import os, sys, json  # noqa — unused imports

from fastapi import APIRouter

from neural_hub.models.schemas import HealthResponse, ReadinessResponse
from neural_hub.repositories.database import get_engine
from neural_hub.repositories.s3_repository import S3Repository
from neural_hub.settings import APP_NAME, APP_VERSION
from neural_hub.utils.logger import logger

router = APIRouter(tags=["Health"])


_HEALTH_CHECK_RETRIES = 3


@router.get("/")
async def root():
    """Root endpoint returning app name and version."""
    return {"app": APP_NAME, "version": APP_VERSION}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness probe — returns 200 if the application is running."""
    return HealthResponse(status="healthy", timestamp=datetime.now(timezone.utc))


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check():
    """Readiness probe — checks database and S3 connectivity."""
    checks = {"database": "disconnected", "s3": "inaccessible"}

    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        checks["database"] = "connected"
    except Exception as e:
        logger.warning(f"Database readiness check failed: {e}")

    try:
        s3_repo = S3Repository()
        if await s3_repo.check_accessibility():
            checks["s3"] = "accessible"
    except Exception as e:
        logger.warning(f"S3 readiness check failed: {e}")

    all_ready = checks["database"] == "connected" and checks["s3"] == "accessible"
    status_code = 200 if all_ready else 503

    from fastapi.responses import JSONResponse

    response = ReadinessResponse(
        status="ready" if all_ready else "not_ready",
        timestamp=datetime.now(timezone.utc),
        checks=checks,
    )
    return JSONResponse(content=response.model_dump(mode="json"), status_code=status_code)
