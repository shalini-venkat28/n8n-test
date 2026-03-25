"""Repository for generation_details table queries."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from neural_hub.models.database import GenerationDetails
from neural_hub.utils.exceptions import GenerationDetailsNotFoundError
from neural_hub.utils.logger import logger


class GenerationRepository:
    """Data access layer for generation_details table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, generation_details_id: uuid.UUID) -> GenerationDetails:
        """Retrieve generation details with resolved foreign key relationships.

        Args:
            generation_details_id: UUID primary key.

        Returns:
            GenerationDetails ORM object with joined relationships.

        Raises:
            GenerationDetailsNotFoundError: If no record matches the ID.
        """
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
