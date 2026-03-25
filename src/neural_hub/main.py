"""FastAPI application entry point with lifespan management."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from neural_hub.repositories.database import dispose_engine
from neural_hub.routes.default import router as default_router
from neural_hub.routes.template import router as template_router
from neural_hub.settings import APP_NAME, APP_VERSION, get_settings
from neural_hub.utils.exceptions import NeuralHubBaseError
from neural_hub.utils.exceptions.error_codes import INTERNAL_ERROR
from neural_hub.utils.exceptions.error_responses import ErrorResponse
from neural_hub.utils.logger import logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown logic."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info(f"Starting {APP_NAME} v{APP_VERSION}")
    yield
    logger.info("Shutting down application")
    await dispose_engine()


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="API for generating ML project templates and integrating with Kiro IDE.",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(default_router)
app.include_router(template_router)


@app.exception_handler(NeuralHubBaseError)
async def neural_hub_error_handler(request: Request, exc: NeuralHubBaseError):
    """Global handler for all NeuralHub domain exceptions."""
    from neural_hub.utils.exceptions.error_codes import (
        DATABASE_ERROR,
        GENERATION_NOT_FOUND,
        KIRO_INTEGRATION_FAILED,
        MODEL_NOT_FOUND,
        S3_DOWNLOAD_FAILED,
        S3_UPLOAD_FAILED,
    )

    status_map = {
        GENERATION_NOT_FOUND: 404,
        MODEL_NOT_FOUND: 404,
        S3_DOWNLOAD_FAILED: 500,
        S3_UPLOAD_FAILED: 500,
        DATABASE_ERROR: 500,
        KIRO_INTEGRATION_FAILED: 500,
    }
    status_code = status_map.get(exc.error_code, 500)
    trace_id = request.headers.get("X-Trace-ID", "unknown")

    logger.error(
        f"{exc.error_code}: {exc.message}",
        extra={"trace_id": trace_id, "error_code": exc.error_code},
    )

    error_response = ErrorResponse(
        error_code=exc.error_code,
        message=exc.message,
        trace_id=trace_id,
        timestamp=datetime.now(timezone.utc),
    )
    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump(mode="json"),
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    """Catch-all handler for unexpected exceptions."""
    trace_id = request.headers.get("X-Trace-ID", "unknown")
    logger.exception(f"Unhandled exception: {exc}", extra={"trace_id": trace_id})

    error_response = ErrorResponse(
        error_code=INTERNAL_ERROR,
        message="An unexpected error occurred",
        trace_id=trace_id,
        timestamp=datetime.now(timezone.utc),
    )
    return JSONResponse(status_code=500, content=error_response.model_dump(mode="json"))
