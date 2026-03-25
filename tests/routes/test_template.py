"""Tests for template generation API endpoints."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch


class TestTemplateDownloadEndpoint:
    """Tests for POST /api/v1/template-download."""

    def test_invalid_uuid_returns_422(self, test_client):
        response = test_client.post(
            "/api/v1/template-download",
            json={
                "model_recommendation_id": "not-a-uuid",
                "generation_details_id": "not-a-uuid",
            },
        )
        assert response.status_code == 422

    def test_missing_body_returns_422(self, test_client):
        response = test_client.post("/api/v1/template-download", json={})
        assert response.status_code == 422

    @patch("neural_hub.routes.template.TemplateService")
    def test_template_download_success(self, mock_svc_cls, test_client):
        mock_svc = mock_svc_cls.return_value
        mock_svc.generate_template = AsyncMock(return_value={
            "s3_uri": "s3://bucket/templates/test/",
            "template_id": "039f74eb-e370-4daf-87d2-8b1de4efdf4c",
            "generated_at": datetime.now(timezone.utc),
            "status": "success",
        })

        response = test_client.post(
            "/api/v1/template-download",
            json={
                "model_recommendation_id": "ae9ed59a-25b4-4bb5-a15e-0da18638c5b8",
                "generation_details_id": "039f74eb-e370-4daf-87d2-8b1de4efdf4c",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "s3_uri" in data
        assert "X-Trace-ID" in response.headers
