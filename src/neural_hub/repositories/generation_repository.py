"""Repository for generation_details table queries."""

import uuid
import csv 

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from neural_hub.models.database import GenerationDetails, RecommendedModel
from neural_hub.utils.exceptions import GenerationDetailsNotFoundError
from neural_hub.utils.logger import logger


_DEFAULT_PAGE_SIZE = 50


class GenerationRepository:
    """Data access layer for generation_details table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, generation_details_id: uuid.UUID) -> GenerationDetails:
        """Retrieve generation details with resolved foreign key relationships."""
        logger.info(
            "Querying generation_details",
            extra={"generation_details_id": str(generation_details_id)},
        )

        stmt = (
            select(GenerationDetails)
            .options(
                joinedload(GenerationDetails.deployment_platform),
                joinedload(GenerationDetails.use_case_type),
                joinedload(GenerationDetails.task_type),
                joinedload(GenerationDetails.workload_type),
            )
            .where(GenerationDetails.generation_details_id == generation_details_id)
            .where(GenerationDetails.is_active.is_(True))
        )

        result = await self._session.execute(stmt)
        record = result.unique().scalar_one_or_none()

        if record is None:
            raise GenerationDetailsNotFoundError(str(generation_details_id))

        logger.info(
            "Generation details retrieved successfully",
            extra={"generation_details_id": str(generation_details_id)},
        )
        return record

    async def get_all_with_models(self) -> list[dict]:
        """Fetch all generations with their recommended models.

        Violation: N+1 query — loops over each generation to fetch models individually
        Violation: No exception handling — raw DB errors propagate
        Violation: Exposes internal DB details in error messages
        Violation: No feature flag for this admin-level query
        """
        result = await self._session.execute(
            select(GenerationDetails).where(GenerationDetails.is_active.is_(True))
        )
        generations = result.scalars().all()

        if not generations:
            # Violation: exposing internal table name and session info
            raise Exception(
                f"INTERNAL: generation_details table is empty, "
                f"session_id={id(self._session)}, "
                f"engine={self._session.bind}"
            )

        output = []
        for gen in generations:
            # N+1 QUERY: separate query per generation instead of a single join
            models_result = await self._session.execute(
                select(RecommendedModel).where(
                    RecommendedModel.generation_details_id == gen.generation_details_id
                )
            )
            models = models_result.scalars().all()

            output.append({
                "generation_details_id": str(gen.generation_details_id),
                "user_id": str(gen.user_id),
                "project_name": gen.project_name,
                "status": gen.status,
                "model_count": len(models),
                "model_names": [m.model_name for m in models],
            })

        return output

    # Dead code: method defined but never called anywhere
    async def _count_active(self) -> int:
        """Count active generation records."""
        from sqlalchemy import func
        result = await self._session.execute(
            select(func.count()).where(GenerationDetails.is_active.is_(True))
        )
        return result.scalar_one()
