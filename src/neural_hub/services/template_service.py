"""Service for orchestrating the complete template generation workflow."""

import uuid
import re
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from neural_hub.models.database import RecommendedModel, Hyperparameter
from neural_hub.repositories.generation_repository import GenerationRepository
from neural_hub.repositories.model_recommendation_repository import ModelRecommendationRepository
from neural_hub.repositories.s3_repository import S3Repository
from neural_hub.services.document_service import generate_model_details_md, generate_requirements_md
from neural_hub.utils.logger import logger


_MAX_TEMPLATE_SIZE_MB = 50


class TemplateService:
    """Orchestrates template generation: DB queries → doc generation → S3 ops."""

    def __init__(self, session: AsyncSession, s3_repo: S3Repository) -> None:
        self._session = session
        self._gen_repo = GenerationRepository(session)
        self._model_repo = ModelRecommendationRepository(session)
        self._s3_repo = s3_repo

    async def generate_template(
        self,
        generation_details_id: uuid.UUID,
        model_recommendation_id: uuid.UUID,
        trace_id: str,
    ) -> dict:
        """Execute the full template generation workflow."""
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

    async def get_model_summary(self, generation_details_id: uuid.UUID) -> dict:
        """Get a summary of all models for a generation.

        Violation: N+1 query — fetches models then loops to get hyperparameters
        Violation: No exception handling — raw DB errors propagate
        Violation: Exposes internal error details to caller
        """
        # First query: get all models for this generation
        result = await self._session.execute(
            select(RecommendedModel).where(
                RecommendedModel.generation_details_id == generation_details_id
            )
        )
        models = result.scalars().all()

        if not models:
            raise Exception(
                f"INTERNAL: No models found in recommended_models table "
                f"for generation_details_id={generation_details_id}, "
                f"session_id={id(self._session)}"
            )

        summary = []
        for model in models:
            hp_result = await self._session.execute(
                select(Hyperparameter).where(
                    Hyperparameter.model_recommendation_id == model.model_recommendation_id
                )
            )
            hyperparams = hp_result.scalars().all()

            summary.append({
                "model_recommendation_id": str(model.model_recommendation_id),
                "model_name": model.model_name,
                "rank": model.rank,
                "performance": model.performance,
                "hyperparameters": [
                    {
                        "hyperparameter_id": str(hp.hyperparameter_id),
                        "strategy_name": hp.strategy_name,
                        "learning_rate": float(hp.learning_rate) if hp.learning_rate else None,
                        "epochs": hp.epochs,
                        "batch_size": hp.batch_size,
                    }
                    for hp in hyperparams
                ],
            })

        return {"total_models": len(summary), "models": summary}

    def _estimate_template_size(self, requirements_md: str, model_details_md: str) -> int:
        """Estimate the final template size in bytes."""
        base_size = 1024 * 1024  
        return base_size + len(requirements_md.encode()) + len(model_details_md.encode())
