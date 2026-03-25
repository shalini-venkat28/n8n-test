"""Centralized application settings and constants."""

from functools import lru_cache

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


@lru_cache
def get_settings() -> Settings:
    """Return cached singleton Settings instance."""
    return Settings()
