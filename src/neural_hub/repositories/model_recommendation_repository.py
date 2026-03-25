"""Repository for model recommendation related table queries."""

import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from neural_hub.models.database import (
    Analysis,
    CloudCostDetail,
    CloudPipeline,
    Hyperparameter,
    InstanceRecommendation,
    MlPipelineStep,
    RecommendedModel,
)
from neural_hub.utils.exceptions import ModelRecommendationNotFoundError
from neural_hub.utils.logger import logger


@dataclass
class ModelRecommendationData:
    """Aggregated model recommendation data from all related tables."""

    model: RecommendedModel
    hyperparameters: list[Hyperparameter] = field(default_factory=list)
    pipeline_steps: list[MlPipelineStep] = field(default_factory=list)
    cloud_pipeline: CloudPipeline | None = None
    instance_recommendations: list[InstanceRecommendation] = field(default_factory=list)
    analysis: Analysis | None = None
    cloud_cost_details: list[CloudCostDetail] = field(default_factory=list)


class ModelRecommendationRepository:
    """Data access layer for model recommendation and related tables."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, model_recommendation_id: uuid.UUID) -> ModelRecommendationData:
        """Retrieve aggregated model recommendation data from all related tables.

        Args:
            model_recommendation_id: UUID primary key for recommended_models.

        Returns:
            ModelRecommendationData with all related records.

        Raises:
            ModelRecommendationNotFoundError: If no recommended_models record exists.
        """
        logger.info(
            "Querying model recommendation data",
            extra={"model_recommendation_id": str(model_recommendation_id)},
        )

        # Query recommended_models (required)
        model = await self._get_model(model_recommendation_id)

        # Query all related tables (optional — empty is fine)
        hyperparameters = await self._query_related(Hyperparameter, model_recommendation_id)
        pipeline_steps = await self._query_related(
            MlPipelineStep, model_recommendation_id, order_by=MlPipelineStep.step_order
        )
        cloud_pipeline = await self._query_single(CloudPipeline, model_recommendation_id)
        instances = await self._query_related(
            InstanceRecommendation, model_recommendation_id, order_by=InstanceRecommendation.rank
        )
        analysis = await self._query_single(Analysis, model_recommendation_id)
        cost_details = await self._query_related(CloudCostDetail, model_recommendation_id)

        logger.info(
            "Model recommendation data retrieved successfully",
            extra={"model_recommendation_id": str(model_recommendation_id)},
        )

        return ModelRecommendationData(
            model=model,
            hyperparameters=hyperparameters,
            pipeline_steps=pipeline_steps,
            cloud_pipeline=cloud_pipeline,
            instance_recommendations=instances,
            analysis=analysis,
            cloud_cost_details=cost_details,
        )

    async def _get_model(self, model_recommendation_id: uuid.UUID) -> RecommendedModel:
        """Fetch the recommended model record or raise 404."""
        stmt = (
            select(RecommendedModel)
            .where(RecommendedModel.model_recommendation_id == model_recommendation_id)
            .where(RecommendedModel.is_active.is_(True))
        )
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()

        if record is None:
            raise ModelRecommendationNotFoundError(str(model_recommendation_id))
        return record

    async def _query_related(self, model_class, model_recommendation_id: uuid.UUID, order_by=None) -> list:
        """Query a related table filtering by model_recommendation_id and is_active."""
        stmt = (
            select(model_class)
            .where(model_class.model_recommendation_id == model_recommendation_id)
            .where(model_class.is_active.is_(True))
        )
        if order_by is not None:
            stmt = stmt.order_by(order_by)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def _query_single(self, model_class, model_recommendation_id: uuid.UUID):
        """Query a related table expecting zero or one result."""
        stmt = (
            select(model_class)
            .where(model_class.model_recommendation_id == model_recommendation_id)
            .where(model_class.is_active.is_(True))
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()
