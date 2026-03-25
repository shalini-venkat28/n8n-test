"""Shared test fixtures."""

import os
import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# Set env vars before importing app
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/testdb")
os.environ.setdefault("S3_BUCKET_NAME", "test-bucket")
os.environ.setdefault("AWS_REGION", "us-east-1")

from neural_hub.models.database import (
    DeploymentPlatform,
    GenerationDetails,
    RecommendedModel,
    TaskType,
    UseCaseType,
    WorkloadType,
)


@pytest.fixture
def gen_details_id():
    return uuid.UUID("039f74eb-e370-4daf-87d2-8b1de4efdf4c")


@pytest.fixture
def model_rec_id():
    return uuid.UUID("ae9ed59a-25b4-4bb5-a15e-0da18638c5b8")


@pytest.fixture
def sample_generation_details(gen_details_id):
    """Create a mock GenerationDetails ORM object."""
    details = MagicMock(spec=GenerationDetails)
    details.generation_details_id = gen_details_id
    details.project_name = "Test ML Project"
    details.project_description = "A test ML project for image classification."
    details.status = "processing"
    details.uploaded_file_summary = "Technical analysis of the project."
    details.configuration_values = {"framework": "pytorch", "gpu": True}
    details.cost_simulation_parameter_values = {
        "training_hours_per_month": 100,
        "inference_hours_per_day": 8,
    }

    # Mock relationships
    platform = MagicMock(spec=DeploymentPlatform)
    platform.platform_name = "AWS"
    details.deployment_platform = platform

    use_case = MagicMock(spec=UseCaseType)
    use_case.use_case_type_name = "Computer Vision"
    details.use_case_type = use_case

    task = MagicMock(spec=TaskType)
    task.task_type_name = "Image Classification"
    details.task_type = task

    workload = MagicMock(spec=WorkloadType)
    workload.workload_type_name = "Training"
    workload.project_structure = "src/\n  model/\n  data/"
    details.workload_type = workload

    return details


@pytest.fixture
def sample_recommended_model(model_rec_id):
    """Create a mock RecommendedModel ORM object."""
    model = MagicMock(spec=RecommendedModel)
    model.model_recommendation_id = model_rec_id
    model.model_name = "ResNet-50"
    model.rank = 1
    model.use_case = "Image classification"
    model.why_these_model = "High accuracy with reasonable compute"
    model.features = ["transfer_learning", "batch_norm"]
    model.performance = 0.95
    model.latency = 50
    model.ml_pipeline_duration = "4 hours"
    model.ml_pipeline_complexity = "Medium"
    model.cost_category = "Medium"
    return model


@pytest.fixture
def test_client():
    """Create a FastAPI test client with mocked dependencies."""
    from neural_hub.main import app

    return TestClient(app)
