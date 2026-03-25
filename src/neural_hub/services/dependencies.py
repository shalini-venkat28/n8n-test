"""Dependency injection configuration for FastAPI."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from neural_hub.repositories.database import get_db_session
from neural_hub.repositories.s3_repository import S3Repository
from neural_hub.services.kiro_service import KiroService
from neural_hub.services.template_service import TemplateService


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a database session via FastAPI dependency injection."""
    async for session in get_db_session():
        yield session


def get_s3_repository() -> S3Repository:
    """Provide an S3Repository instance."""
    return S3Repository()


def get_kiro_service() -> KiroService:
    """Provide a KiroService instance."""
    return KiroService()


async def get_template_service(
    session: AsyncSession,
    s3_repo: S3Repository,
) -> TemplateService:
    """Provide a TemplateService with injected dependencies."""
    return TemplateService(session=session, s3_repo=s3_repo)
