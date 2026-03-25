"""Tests for template service orchestration."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_hub.repositories.model_recommendation_repository import ModelRecommendationData
from neural_hub.services.template_service import TemplateService


class TestTemplateService:
    """Tests for TemplateService.generate_template."""

    @pytest.fixture
    def mock_session(self):
        return AsyncMock()

    @pytest.fixture
    def mock_s3_repo(self):
        repo = MagicMock()
        repo.process_template.return_value = "s3://test-bucket/templates/test-id/"
        return repo

    @pytest.fixture
    def service(self, mock_session, mock_s3_repo):
        return TemplateService(session=mock_session, s3_repo=mock_s3_repo)

    @pytest.mark.asyncio
    @patch("neural_hub.services.template_service.generate_model_details_md")
    @patch("neural_hub.services.template_service.generate_requirements_md")
    @patch("neural_hub.services.template_service.ModelRecommendationRepository")
    @patch("neural_hub.services.template_service.GenerationRepository")
    async def test_generate_template_success(
        self, mock_gen_repo_cls, mock_model_repo_cls,
        mock_gen_req, mock_gen_model, service, mock_s3_repo,
        sample_generation_details, sample_recommended_model,
    ):
        mock_gen_repo_cls.return_value.get_by_id = AsyncMock(return_value=sample_generation_details)
        mock_model_repo_cls.return_value.get_by_id = AsyncMock(
            return_value=ModelRecommendationData(model=sample_recommended_model)
        )
        mock_gen_req.return_value = "# Requirements"
        mock_gen_model.return_value = "# Model Details"

        gen_id = uuid.UUID("039f74eb-e370-4daf-87d2-8b1de4efdf4c")
        model_id = uuid.UUID("ae9ed59a-25b4-4bb5-a15e-0da18638c5b8")

        result = await service.generate_template(gen_id, model_id, "trace-123")

        assert result["status"] == "success"
        assert "s3://" in result["s3_uri"]
        assert result["template_id"] == str(gen_id)
        mock_s3_repo.process_template.assert_called_once()

    @pytest.mark.asyncio
    @patch("neural_hub.services.template_service.GenerationRepository")
    async def test_generate_template_gen_not_found(
        self, mock_gen_repo_cls, service,
    ):
        from neural_hub.utils.exceptions import GenerationDetailsNotFoundError

        mock_gen_repo_cls.return_value.get_by_id = AsyncMock(
            side_effect=GenerationDetailsNotFoundError("bad-id")
        )

        with pytest.raises(GenerationDetailsNotFoundError):
            await service.generate_template(uuid.uuid4(), uuid.uuid4(), "trace-123")
