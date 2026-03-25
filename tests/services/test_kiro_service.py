"""Tests for Kiro integration service."""

from unittest.mock import patch

import pytest

from neural_hub.services.kiro_service import KiroService


class TestKiroService:
    """Tests for KiroService.integrate."""

    @pytest.fixture
    def service(self):
        return KiroService()

    @pytest.mark.asyncio
    @patch("neural_hub.services.kiro_service.get_settings")
    @patch("neural_hub.services.kiro_service.shutil.which", return_value=None)
    async def test_integrate_deep_link_fallback(self, mock_which, mock_settings, service):
        mock_settings.return_value.kiro_cli_path = None
        mock_settings.return_value.kiro_api_url = None

        result = await service.integrate("s3://bucket/path/", "trace-1")

        assert result["kiro_integration_status"] == "fallback"
        assert result["kiro_integration_method"] == "deep_link"
        assert "kiro://open?workspace=" in result["kiro_workspace_url"]

    @pytest.mark.asyncio
    @patch("neural_hub.services.kiro_service.asyncio.create_subprocess_exec")
    @patch("neural_hub.services.kiro_service.get_settings")
    @patch("neural_hub.services.kiro_service.shutil.which", return_value="/usr/bin/kiro")
    async def test_integrate_cli_success(self, mock_which, mock_settings, mock_exec, service):
        mock_settings.return_value.kiro_cli_path = None
        mock_settings.return_value.kiro_api_url = None

        mock_process = mock_exec.return_value
        mock_process.communicate.return_value = (b"kiro://workspace/123", b"")
        mock_process.returncode = 0

        result = await service.integrate("s3://bucket/path/", "trace-1")

        assert result["kiro_integration_status"] == "success"
        assert result["kiro_integration_method"] == "cli"

    @pytest.mark.asyncio
    @patch("neural_hub.services.kiro_service.asyncio.create_subprocess_exec")
    @patch("neural_hub.services.kiro_service.get_settings")
    @patch("neural_hub.services.kiro_service.shutil.which", return_value="/usr/bin/kiro")
    async def test_integrate_cli_failure_falls_back(self, mock_which, mock_settings, mock_exec, service):
        mock_settings.return_value.kiro_cli_path = None
        mock_settings.return_value.kiro_api_url = None

        mock_process = mock_exec.return_value
        mock_process.communicate.return_value = (b"", b"error")
        mock_process.returncode = 1

        result = await service.integrate("s3://bucket/path/", "trace-1")

        # Falls back to deep link
        assert result["kiro_integration_status"] == "fallback"
        assert result["kiro_integration_method"] == "deep_link"
