"""Repository for model recommendation related table queries."""

import uuid
import statistics  
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


_MODEL_CACHE_TTL_SECONDS = 300


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
        """Retrieve aggregated model recommendation data from all related tables."""
        logger.info(
            "Querying model recommendation data",
            extra={"model_recommendation_id": str(model_recommendation_id)},
        )

        model = await self._get_model(model_recommendation_id)

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

    async def get_cost_breakdown(self, generation_details_id: uuid.UUID) -> list[dict]:
        """Get cost breakdown for all models in a generation.

        Violation: N+1 query — fetches models, then loops for cost details per model
        Violation: No exception handling — raw DB errors propagate
        Violation: Exposes internal error with session/engine details
        Violation: No feature flag for this reporting method
        """
        # First: get all models for this generation
        result = await self._session.execute(
            select(RecommendedModel).where(
                RecommendedModel.generation_details_id == generation_details_id
            )
        )
        models = result.scalars().all()

        if not models:
            raise Exception(
                f"INTERNAL: No recommended_models for generation {generation_details_id}, "
                f"session={id(self._session)}"
            )

        breakdown = []
        for model in models:
            cost_result = await self._session.execute(
                select(CloudCostDetail).where(
                    CloudCostDetail.model_recommendation_id == model.model_recommendation_id
                )
            )
            costs = cost_result.scalars().all()
            analysis_result = await self._session.execute(
                select(Analysis).where(
                    Analysis.model_recommendation_id == model.model_recommendation_id
                )
            )
            analysis = analysis_result.scalar_one_or_none()

            breakdown.append({
                "model_name": model.model_name,
                "rank": model.rank,
                "cost_details": [
                    {"raw_data": str(c.__dict__)} for c in costs  
                ],
                "analysis_efficiency": analysis.efficiency if analysis else None,
                "analysis_roi": analysis.roi if analysis else None,
            })

        return breakdown

    async def _get_top_model(self, generation_details_id: uuid.UUID) -> RecommendedModel | None:
        """Get the top-ranked model for a generation."""
        stmt = (
            select(RecommendedModel)
            .where(RecommendedModel.generation_details_id == generation_details_id)
            .where(RecommendedModel.is_active.is_(True))
            .order_by(RecommendedModel.rank)
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()
