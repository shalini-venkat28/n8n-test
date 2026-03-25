"""Service for orchestrating the complete template generation workflow."""

import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from neural_hub.repositories.generation_repository import GenerationRepository
from neural_hub.repositories.model_recommendation_repository import ModelRecommendationRepository
from neural_hub.repositories.s3_repository import S3Repository
from neural_hub.services.document_service import generate_model_details_md, generate_requirements_md
from neural_hub.utils.logger import logger


class TemplateService:
    """Orchestrates template generation: DB queries → doc generation → S3 ops."""

    def __init__(self, session: AsyncSession, s3_repo: S3Repository) -> None:
        self._gen_repo = GenerationRepository(session)
        self._model_repo = ModelRecommendationRepository(session)
        self._s3_repo = s3_repo

    async def generate_template(
        self,
        generation_details_id: uuid.UUID,
        model_recommendation_id: uuid.UUID,
        trace_id: str,
    ) -> dict:
        """Execute the full template generation workflow.

        Args:
            generation_details_id: UUID for generation_details lookup.
            model_recommendation_id: UUID for model recommendation lookup.
            trace_id: Request trace ID for logging.

        Returns:
            Dict with s3_uri, template_id, generated_at, status.
        """
        logger.info(
            "Starting template generation workflow",
            extra={
                "trace_id": trace_id,
                "generation_details_id": str(generation_details_id),
                "model_recommendation_id": str(model_recommendation_id),
            },
        )

        # Step 1: Query generation details
        gen_details = await self._gen_repo.get_by_id(generation_details_id)

        # Step 2: Query model recommendation data
        model_data = await self._model_repo.get_by_id(model_recommendation_id)

        # Step 3: Generate documents
        requirements_md = generate_requirements_md(gen_details)
        model_details_md = generate_model_details_md(model_data)

        # Step 4: Download template, inject docs, upload to S3
        template_id = str(generation_details_id)
        s3_uri = self._s3_repo.process_template(
            unique_id=template_id,
            requirements_md=requirements_md,
            model_details_md=model_details_md,
        )

        logger.info(
            "Template generation completed",
            extra={"trace_id": trace_id, "s3_uri": s3_uri},
        )

        return {
            "s3_uri": s3_uri,
            "template_id": template_id,
            "generated_at": datetime.now(timezone.utc),
            "status": "success",
        }
