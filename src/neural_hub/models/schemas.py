"""Pydantic models for API request/response validation."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

# --- Request Models ---

class TemplateDownloadRequest(BaseModel):
    """Request body for template download endpoint."""

    model_recommendation_id: UUID
    """UUID of the model recommendation record."""

    generation_details_id: UUID
    """UUID of the generation details record."""


class KiroIntegrationRequest(BaseModel):
    """Request body for Kiro integration endpoint."""

    model_recommendation_id: UUID
    """UUID of the model recommendation record."""

    generation_details_id: UUID
    """UUID of the generation details record."""


# --- Response Models ---

class TemplateDownloadResponse(BaseModel):
    """Response for template download endpoint."""

    model_config = ConfigDict(from_attributes=True)

    s3_uri: str
    """S3 URI of the uploaded template folder."""

    template_id: str
    """Unique identifier for the generated template."""

    generated_at: datetime
    """Timestamp when the template was generated."""

    status: str = "success"
    """Status of the operation."""


class KiroIntegrationResponse(BaseModel):
    """Response for Kiro integration endpoint."""

    model_config = ConfigDict(from_attributes=True)

    s3_uri: str
    """S3 URI of the uploaded template folder."""

    template_id: str
    """Unique identifier for the generated template."""

    generated_at: datetime
    """Timestamp when the template was generated."""

    kiro_workspace_url: str | None = None
    """Kiro workspace URL or deep link."""

    kiro_integration_status: str
    """Status of Kiro integration (success, fallback, failed)."""

    kiro_integration_method: str | None = None
    """Method used for Kiro integration."""

    status: str = "success"
    """Overall operation status."""

    warning: str | None = None
    """Warning message if Kiro integration failed."""


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    timestamp: datetime


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    status: str
    timestamp: datetime
    checks: dict
