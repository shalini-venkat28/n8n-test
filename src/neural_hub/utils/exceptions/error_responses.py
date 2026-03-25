"""Pydantic models for standardized error responses."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ErrorResponse(BaseModel):
    """Standardized API error response."""

    model_config = ConfigDict(from_attributes=True)

    error_code: str
    """Machine-readable error code."""

    message: str
    """Human-readable error message."""

    trace_id: str
    """Unique trace identifier for request correlation."""

    timestamp: datetime
    """When the error occurred."""

    details: dict | None = None
    """Additional error context."""
