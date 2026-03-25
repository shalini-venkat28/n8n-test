"""Template generation and Kiro integration API endpoints."""

import uuid

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from neural_hub.models.schemas import (
    KiroIntegrationRequest,
    KiroIntegrationResponse,
    TemplateDownloadRequest,
    TemplateDownloadResponse,
)
from neural_hub.repositories.s3_repository import S3Repository
from neural_hub.services.dependencies import get_kiro_service, get_s3_repository, get_session
from neural_hub.services.kiro_service import KiroService
from neural_hub.services.template_service import TemplateService
from neural_hub.utils.logger import logger

router = APIRouter(prefix="/api/v1", tags=["Template Generation"])


def _get_trace_id(request: Request) -> str:
    """Extract or generate a trace ID for the request."""
    return request.headers.get("X-Trace-ID", str(uuid.uuid4()))


@router.post("/template-download", response_model=TemplateDownloadResponse)
async def template_download(
    body: TemplateDownloadRequest,
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_session),
    s3_repo: S3Repository = Depends(get_s3_repository),
):
    """Generate a customized Kiro project template and return the S3 URI."""
    trace_id = _get_trace_id(request)
    response.headers["X-Trace-ID"] = trace_id

    logger.info(
        "Template download request received",
        extra={
            "trace_id": trace_id,
            "generation_details_id": str(body.generation_details_id),
            "model_recommendation_id": str(body.model_recommendation_id),
        },
    )

    service = TemplateService(session=session, s3_repo=s3_repo)
    result = await service.generate_template(
        generation_details_id=body.generation_details_id,
        model_recommendation_id=body.model_recommendation_id,
        trace_id=trace_id,
    )

    return TemplateDownloadResponse(**result)


@router.post("/kiro-integration", response_model=KiroIntegrationResponse)
async def kiro_integration(
    body: KiroIntegrationRequest,
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_session),
    s3_repo: S3Repository = Depends(get_s3_repository),
    kiro_svc: KiroService = Depends(get_kiro_service),
):
    """Generate template and open in Kiro app with graceful fallback."""
    trace_id = _get_trace_id(request)
    response.headers["X-Trace-ID"] = trace_id

    logger.info(
        "Kiro integration request received",
        extra={
            "trace_id": trace_id,
            "generation_details_id": str(body.generation_details_id),
            "model_recommendation_id": str(body.model_recommendation_id),
        },
    )

    # Step 1: Generate template (same as template-download)
    template_svc = TemplateService(session=session, s3_repo=s3_repo)
    result = await template_svc.generate_template(
        generation_details_id=body.generation_details_id,
        model_recommendation_id=body.model_recommendation_id,
        trace_id=trace_id,
    )

    # Step 2: Kiro integration with fallback
    kiro_result = {
        "kiro_workspace_url": None,
        "kiro_integration_status": "failed",
        "kiro_integration_method": None,
    }
    warning = None

    try:
        kiro_result = await kiro_svc.integrate(s3_uri=result["s3_uri"], trace_id=trace_id)
    except Exception as e:
        logger.error(
            f"Kiro integration failed: {e}",
            extra={"trace_id": trace_id},
            exc_info=True,
        )
        warning = "Template generated successfully but Kiro integration failed. Use S3 URI to manually open template."

    status = "success" if kiro_result["kiro_integration_status"] != "failed" else "partial_success"

    return KiroIntegrationResponse(
        s3_uri=result["s3_uri"],
        template_id=result["template_id"],
        generated_at=result["generated_at"],
        kiro_workspace_url=kiro_result.get("kiro_workspace_url"),
        kiro_integration_status=kiro_result["kiro_integration_status"],
        kiro_integration_method=kiro_result.get("kiro_integration_method"),
        status=status,
        warning=warning,
    )
