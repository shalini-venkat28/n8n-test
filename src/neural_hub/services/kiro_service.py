"""Service for Kiro app integration with fallback logic."""

import asyncio
import shutil

from neural_hub.settings import KIRO_DEEP_LINK_SCHEME, get_settings
from neural_hub.utils.logger import logger


class KiroService:
    """Handles Kiro CLI/API/deep-link integration with graceful fallback."""

    async def integrate(self, s3_uri: str, trace_id: str) -> dict:
        """Attempt to open Kiro workspace using CLI → API → deep link fallback.

        Args:
            s3_uri: S3 URI of the uploaded template.
            trace_id: Request trace ID for logging.

        Returns:
            Dict with kiro_workspace_url, kiro_integration_status, kiro_integration_method.
        """
        settings = get_settings()

        # Primary: Kiro CLI
        if settings.kiro_cli_path or shutil.which("kiro"):
            cli_path = settings.kiro_cli_path or "kiro"
            result = await self._try_cli(cli_path, s3_uri, trace_id)
            if result:
                return result

        # Secondary: Kiro REST API
        if settings.kiro_api_url:
            result = await self._try_api(settings.kiro_api_url, s3_uri, trace_id)
            if result:
                return result

        # Fallback: Deep link URL
        deep_link = f"{KIRO_DEEP_LINK_SCHEME}{s3_uri}"
        logger.info(
            "Using deep link fallback for Kiro integration",
            extra={"trace_id": trace_id, "deep_link": deep_link},
        )
        return {
            "kiro_workspace_url": deep_link,
            "kiro_integration_status": "fallback",
            "kiro_integration_method": "deep_link",
        }

    async def _try_cli(self, cli_path: str, s3_uri: str, trace_id: str) -> dict | None:
        """Attempt Kiro CLI integration."""
        try:
            logger.info(f"Attempting Kiro CLI: {cli_path} open --workspace {s3_uri}", extra={"trace_id": trace_id})
            process = await asyncio.create_subprocess_exec(
                cli_path, "open", "--workspace", s3_uri,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)

            if process.returncode == 0:
                workspace_url = stdout.decode().strip() or f"kiro://workspace/{s3_uri}"
                logger.info("Kiro CLI integration successful", extra={"trace_id": trace_id})
                return {
                    "kiro_workspace_url": workspace_url,
                    "kiro_integration_status": "success",
                    "kiro_integration_method": "cli",
                }
            logger.warning(
                f"Kiro CLI failed with return code {process.returncode}: {stderr.decode()}",
                extra={"trace_id": trace_id},
            )
        except (asyncio.TimeoutError, FileNotFoundError, OSError) as e:
            logger.warning(f"Kiro CLI integration failed: {e}", extra={"trace_id": trace_id})
        return None

    async def _try_api(self, api_url: str, s3_uri: str, trace_id: str) -> dict | None:
        """Attempt Kiro REST API integration."""
        try:
            import httpx

            logger.info(f"Attempting Kiro API: {api_url}", extra={"trace_id": trace_id})
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    f"{api_url}/workspaces",
                    json={"s3_uri": s3_uri},
                )
                if response.status_code in (200, 201):
                    data = response.json()
                    workspace_url = data.get("workspace_url", f"kiro://workspace/{s3_uri}")
                    logger.info("Kiro API integration successful", extra={"trace_id": trace_id})
                    return {
                        "kiro_workspace_url": workspace_url,
                        "kiro_integration_status": "success",
                        "kiro_integration_method": "api",
                    }
                logger.warning(f"Kiro API returned {response.status_code}", extra={"trace_id": trace_id})
        except Exception as e:
            logger.warning(f"Kiro API integration failed: {e}", extra={"trace_id": trace_id})
        return None
