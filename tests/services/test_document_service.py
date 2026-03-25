"""Tests for document orchestration service."""

from unittest.mock import MagicMock

from neural_hub.models.database import (
    Analysis,
    CloudCostDetail,
    Hyperparameter,
    InstanceRecommendation,
    MlPipelineStep,
)
from neural_hub.repositories.model_recommendation_repository import ModelRecommendationData
from neural_hub.services.document_service import (
    generate_model_details_md,
    generate_requirements_md,
)


class TestGenerateRequirementsMd:
    """Tests for requirements.md generation."""

    def test_generate_requirements_success(self, sample_generation_details):
        result = generate_requirements_md(sample_generation_details)

        assert "# Test ML Project" in result
        assert "## Project Overview" in result
        assert "## Business Context" in result
        assert "AWS" in result
        assert "Computer Vision" in result
        assert "Image Classification" in result
        assert "Training" in result
        assert "pytorch" in result

    def test_generate_requirements_null_relationships(self, sample_generation_details):
        sample_generation_details.deployment_platform = None
        sample_generation_details.use_case_type = None
        sample_generation_details.task_type = None
        sample_generation_details.workload_type = None
        sample_generation_details.uploaded_file_summary = None
        sample_generation_details.configuration_values = None
        sample_generation_details.cost_simulation_parameter_values = None

        result = generate_requirements_md(sample_generation_details)

        assert "Not specified" in result
        assert "No specific configuration values" in result
        assert "No workload parameters" in result

    def test_generate_requirements_with_project_structure(self, sample_generation_details):
        result = generate_requirements_md(sample_generation_details)
        assert "## Project Structure" in result
        assert "src/" in result


class TestGenerateModelDetailsMd:
    """Tests for model_details.md generation."""

    def _make_data(self, sample_recommended_model, **kwargs):
        return ModelRecommendationData(model=sample_recommended_model, **kwargs)

    def test_generate_model_details_minimal(self, sample_recommended_model):
        data = self._make_data(sample_recommended_model)
        result = generate_model_details_md(data)

        assert "# ResNet-50" in result
        assert "## Model Overview" in result
        assert "No pipeline steps defined" in result
        assert "No hyperparameters defined" in result

    def test_generate_model_details_with_pipeline_steps(self, sample_recommended_model):
        step = MagicMock(spec=MlPipelineStep)
        step.step_order = 1
        step.step_name = "Data Preprocessing"
        step.step_description = "Clean and normalize data"
        step.tools_used = ["pandas", "numpy"]
        step.compute = {"gpu": False}
        step.prerequisite_warning = "Ensure data is available"

        data = self._make_data(sample_recommended_model, pipeline_steps=[step])
        result = generate_model_details_md(data)

        assert "Step 1: Data Preprocessing" in result
        assert "Clean and normalize data" in result

    def test_generate_model_details_with_hyperparameters(self, sample_recommended_model):
        hp = MagicMock(spec=Hyperparameter)
        hp.strategy_name = "Fine-tuning"
        hp.learning_rate = 0.001
        hp.epochs = 50
        hp.batch_size = 32
        hp.optimizer = "Adam"
        hp.dropout = 0.3
        hp.weight_decay = 0.0001
        hp.description = "Standard fine-tuning strategy"

        data = self._make_data(sample_recommended_model, hyperparameters=[hp])
        result = generate_model_details_md(data)

        assert "Fine-tuning" in result
        assert "Adam" in result

    def test_generate_model_details_with_cost_details(self, sample_recommended_model):
        cd = MagicMock(spec=CloudCostDetail)
        cd.service_name = "EC2"
        cd.cost = 10.50
        cd.cost_unit = "per hour"

        data = self._make_data(sample_recommended_model, cloud_cost_details=[cd])
        result = generate_model_details_md(data)

        assert "EC2" in result
        assert "$10.50" in result

    def test_generate_model_details_with_analysis(self, sample_recommended_model):
        analysis = MagicMock(spec=Analysis)
        analysis.model_description = "A deep CNN model"
        analysis.benefits = ["High accuracy", "Fast inference"]
        analysis.risks = ["Overfitting risk"]
        analysis.gpu_count = 2
        analysis.training_time_in_hours = 8
        analysis.cost_per_hour = 3.5
        analysis.inference_cost_per_hour = 0.5
        analysis.efficiency = "High"
        analysis.scalability = "Medium"
        analysis.roi = "Positive"
        analysis.cost_optimization_tips = "Use spot instances"

        data = self._make_data(sample_recommended_model, analysis=analysis)
        result = generate_model_details_md(data)

        assert "A deep CNN model" in result
        assert "High accuracy" in result
        assert "Use spot instances" in result

    def test_generate_model_details_with_instances(self, sample_recommended_model):
        inst = MagicMock(spec=InstanceRecommendation)
        inst.rank = 1
        inst.instance_name = "ml.g4dn.xlarge"
        inst.vcpu = 4
        inst.memory = 16
        inst.gpu = "T4"
        inst.on_demand_price = 0.526
        inst.spot_price = 0.158

        data = self._make_data(sample_recommended_model, instance_recommendations=[inst])
        result = generate_model_details_md(data)

        assert "ml.g4dn.xlarge" in result
        assert "T4" in result
