"""SQLAlchemy ORM models for PostgreSQL database."""

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class DeploymentPlatform(Base):
    __tablename__ = "deployment_platform"

    platform_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    platform_name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class UseCaseType(Base):
    __tablename__ = "use_case_types"

    use_case_type_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    use_case_type_name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class TaskType(Base):
    __tablename__ = "task_types"

    task_type_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    use_case_type_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("use_case_types.use_case_type_id"))
    task_type_name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class WorkloadType(Base):
    __tablename__ = "workload_types"

    workload_type_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workload_type_name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    project_structure: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class GenerationDetails(Base):
    __tablename__ = "generation_details"

    generation_details_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("user.user_id"))
    deployment_platform_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("deployment_platform.platform_id"))
    use_case_type_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("use_case_types.use_case_type_id"))
    task_type_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("task_types.task_type_id"))
    workload_type_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("workload_types.workload_type_id"))
    from_generation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("generation_details.generation_details_id"))
    project_description: Mapped[str] = mapped_column(Text, nullable=False)
    project_name: Mapped[str] = mapped_column(String(100), nullable=False)
    uploaded_file_link: Mapped[str | None] = mapped_column(Text)
    uploaded_file_summary: Mapped[str | None] = mapped_column(Text)
    configuration_values: Mapped[dict | None] = mapped_column(JSON)
    cost_simulation_parameter_values: Mapped[dict | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(15), nullable=False)
    created_by: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Relationships
    deployment_platform: Mapped[DeploymentPlatform | None] = relationship(lazy="joined")
    use_case_type: Mapped[UseCaseType | None] = relationship(lazy="joined")
    task_type: Mapped[TaskType | None] = relationship(lazy="joined")
    workload_type: Mapped[WorkloadType | None] = relationship(lazy="joined")


class RecommendedModel(Base):
    __tablename__ = "recommended_models"

    model_recommendation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    generation_details_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("generation_details.generation_details_id"))
    workload_type_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("workload_types.workload_type_id"))
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    use_case: Mapped[str | None] = mapped_column(Text)
    why_these_model: Mapped[str | None] = mapped_column(Text)
    features: Mapped[dict | None] = mapped_column(JSON)
    performance: Mapped[float | None] = mapped_column(Float)
    latency: Mapped[int | None] = mapped_column(Integer)
    ml_pipeline_duration: Mapped[str | None] = mapped_column(String(50))
    ml_pipeline_complexity: Mapped[str | None] = mapped_column(String(50))
    is_generated: Mapped[bool] = mapped_column(Boolean, nullable=False)
    cost_category: Mapped[str | None] = mapped_column(String(50))
    created_by: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_by: Mapped[str | None] = mapped_column(String(50))
    updated_at: Mapped[datetime | None] = mapped_column(DateTime)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class Hyperparameter(Base):
    __tablename__ = "hyperparameters"

    hyperparameter_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_recommendation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("recommended_models.model_recommendation_id"))
    strategy_name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    learning_rate: Mapped[float | None] = mapped_column(Numeric(10, 8))
    epochs: Mapped[int | None] = mapped_column(Integer)
    batch_size: Mapped[int | None] = mapped_column(Integer)
    optimizer: Mapped[str | None] = mapped_column(String(50))
    dropout: Mapped[float | None] = mapped_column(Numeric(3, 2))
    weight_decay: Mapped[float | None] = mapped_column(Numeric(10, 8))
    created_by: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class MlPipelineStep(Base):
    __tablename__ = "ml_pipeline_steps"

    pipeline_step_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_recommendation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("recommended_models.model_recommendation_id"))
    step_name: Mapped[str] = mapped_column(String(100), nullable=False)
    step_order: Mapped[int] = mapped_column(Integer, nullable=False)
    step_description: Mapped[str | None] = mapped_column(String(100))
    tools_used: Mapped[dict | None] = mapped_column(JSON)
    compute: Mapped[dict | None] = mapped_column(JSON)
    prerequisite_warning: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class CloudPipeline(Base):
    __tablename__ = "cloud_pipeline"

    pipeline_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_recommendation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("recommended_models.model_recommendation_id"))
    data_layer: Mapped[dict | None] = mapped_column(JSON)
    processing_layer: Mapped[dict | None] = mapped_column(JSON)
    model_layer: Mapped[dict | None] = mapped_column(JSON)
    api_layer: Mapped[dict | None] = mapped_column(JSON)
    architecture_notes: Mapped[dict | None] = mapped_column(JSON)
    draw_img: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class InstanceRecommendation(Base):
    __tablename__ = "instance_recommendations"

    instance_recommendation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_recommendation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("recommended_models.model_recommendation_id"))
    service_name: Mapped[str | None] = mapped_column(String(100))
    instance_type: Mapped[str | None] = mapped_column(String(100))
    instance_name: Mapped[str] = mapped_column(String(100), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    vcpu: Mapped[int | None] = mapped_column(Integer)
    memory: Mapped[int | None] = mapped_column(Integer)
    gpu: Mapped[str | None] = mapped_column(String(100))
    estimated_cost_per_hour: Mapped[float | None] = mapped_column(Numeric(10, 4))
    on_demand_price: Mapped[float | None] = mapped_column(Numeric(10, 4))
    spot_price: Mapped[float | None] = mapped_column(Numeric(10, 4))
    recommended: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_by: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class Analysis(Base):
    __tablename__ = "analysis"

    analysis_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_recommendation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("recommended_models.model_recommendation_id"))
    benefits: Mapped[dict | None] = mapped_column(JSON)
    risks: Mapped[dict | None] = mapped_column(JSON)
    model_description: Mapped[str | None] = mapped_column(Text)
    cost_optimization_tips: Mapped[str | None] = mapped_column(Text)
    gpu_count: Mapped[int | None] = mapped_column(Integer)
    training_time_in_hours: Mapped[int | None] = mapped_column(Integer)
    cost_per_hour: Mapped[float | None] = mapped_column(Float)
    inference_cost_per_hour: Mapped[float | None] = mapped_column(Float)
    efficiency: Mapped[str | None] = mapped_column(String(50))
    scalability: Mapped[str | None] = mapped_column(String(50))
    roi: Mapped[str | None] = mapped_column(String(50))
    created_by: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class CloudCostDetail(Base):
    __tablename__ = "cloud_cost_details"

    cost_detail_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_recommendation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("recommended_models.model_recommendation_id"))
    service_name: Mapped[str] = mapped_column(String(100), nullable=False)
    service_id: Mapped[str | None] = mapped_column(String(10))
    cost: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    cost_unit: Mapped[str | None] = mapped_column(String(100))
    created_by: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
