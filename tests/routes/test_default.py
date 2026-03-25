"""Tests for default health check endpoints."""

from neural_hub.settings import APP_NAME, APP_VERSION


class TestRootEndpoint:
    def test_root_returns_app_info(self, test_client):
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["app"] == APP_NAME
        assert data["version"] == APP_VERSION


class TestHealthEndpoint:
    def test_health_returns_healthy(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestReadinessEndpoint:
    def test_ready_returns_checks(self, test_client):
        # Without real DB/S3, this will return not_ready
        response = test_client.get("/ready")
        data = response.json()
        assert "checks" in data
        assert "database" in data["checks"]
        assert "s3" in data["checks"]
