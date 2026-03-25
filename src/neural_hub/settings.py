"""Centralized application settings and constants."""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings

# S3 Constants
S3_TEMPLATE_BASE_PREFIX = "template/"
S3_TEMPLATES_OUTPUT_PREFIX = "templates/"
S3_DESIGN_FOLDER = "template/design/"

# Document Names
REQUIREMENTS_DOC = "requirements.md"
MODEL_DETAILS_DOC = "model_details.md"

# Application Constants
APP_NAME = "Neural Hub - Kiro Template Integration API"
APP_VERSION = "1.0.0"
API_V1_PREFIX = "/api/v1"

# Kiro Integration
KIRO_DEEP_LINK_SCHEME = "kiro://open?workspace="


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    database_url: str
    """PostgreSQL async connection string."""

    aws_region: str = "us-east-1"
    """AWS region for S3 operations."""

    s3_bucket_name: str = "neuralhub-eus1-data-source-dev"
    """S3 bucket for template storage."""

    aws_access_key_id: str | None = None
    """AWS access key (optional if using IAM roles)."""

    aws_secret_access_key: str | None = None
    """AWS secret key (optional if using IAM roles)."""

    log_level: str = "INFO"
    """Application log level."""

    kiro_cli_path: str | None = None
    """Path to Kiro CLI executable."""

    kiro_api_url: str | None = None
    """Kiro API endpoint URL."""

    # Template settings
    max_template_size_mb: int = 50
    """Maximum template size in megabytes."""

    # Database pool settings
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30
    db_pool_recycle: int = 3600

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Ensure database_url uses an async driver."""
        if not v.startswith(("postgresql+asyncpg://", "sqlite+aiosqlite://")):
            raise ValueError(
                "database_url must use an async driver "
                "(e.g. postgresql+asyncpg:// or sqlite+aiosqlite://)"
            )
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log_level is a recognized Python logging level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return v.upper()

    @field_validator("max_template_size_mb")
    @classmethod
    def validate_max_template_size(cls, v: int) -> int:
        """Ensure max_template_size_mb is a positive value."""
        if v <= 0:
            raise ValueError("max_template_size_mb must be greater than 0")
        return v

    @field_validator("db_pool_size")
    @classmethod
    def validate_db_pool_size(cls, v: int) -> int:
        """Ensure db_pool_size is a positive value."""
        if v <= 0:
            raise ValueError("db_pool_size must be greater than 0")
        return v


@lru_cache
def get_settings() -> Settings:
    """Return cached singleton Settings instance."""
    return Settings()
