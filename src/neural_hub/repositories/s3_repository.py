"""Repository for S3 template download and upload operations."""

import os
import tempfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from neural_hub.settings import (
    S3_DESIGN_FOLDER,
    S3_TEMPLATE_BASE_PREFIX,
    S3_TEMPLATES_OUTPUT_PREFIX,
    get_settings,
)
from neural_hub.utils.exceptions import S3OperationError
from neural_hub.utils.logger import logger


class S3Repository:
    """Data access layer for S3 template operations."""

    def __init__(self) -> None:
        settings = get_settings()
        session_kwargs: dict = {"region_name": settings.aws_region}
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = settings.aws_access_key_id
            session_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key

        session = boto3.Session(**session_kwargs)
        self._s3 = session.client("s3")
        self._bucket = settings.s3_bucket_name

    def download_base_template(self, local_dir: str) -> None:
        """Download the base template folder from S3 to a local directory.

        Args:
            local_dir: Local directory path to download files into.

        Raises:
            S3OperationError: If download fails.
        """
        logger.info("Downloading base template from S3", extra={"bucket": self._bucket})
        try:
            paginator = self._s3.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self._bucket, Prefix=S3_TEMPLATE_BASE_PREFIX)

            file_count = 0
            for page in pages:
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    # Skip directory markers
                    if key.endswith("/"):
                        continue

                    local_path = os.path.join(local_dir, key)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    self._s3.download_file(self._bucket, key, local_path)
                    file_count += 1

            logger.info(f"Downloaded {file_count} files from base template")
        except ClientError as e:
            raise S3OperationError(
                message=f"Failed to download base template from S3: {e}",
                error_code="S3_DOWNLOAD_FAILED",
            ) from e

    def upload_template(self, local_dir: str, unique_id: str) -> str:
        """Upload the complete template folder to S3 with a unique prefix.

        Args:
            local_dir: Local directory containing the template files.
            unique_id: Unique identifier for the S3 prefix path.

        Returns:
            S3 URI of the uploaded template.

        Raises:
            S3OperationError: If upload fails.
        """
        s3_prefix = f"{S3_TEMPLATES_OUTPUT_PREFIX}{unique_id}/"
        logger.info(f"Uploading template to S3 prefix: {s3_prefix}")

        try:
            template_root = os.path.join(local_dir, "template")
            file_count = 0

            for root, _dirs, files in os.walk(template_root):
                for filename in files:
                    local_path = os.path.join(root, filename)
                    # Build relative key from template root
                    relative_path = os.path.relpath(local_path, template_root)
                    s3_key = f"{s3_prefix}{relative_path}".replace("\\", "/")
                    self._s3.upload_file(local_path, self._bucket, s3_key)
                    file_count += 1

            s3_uri = f"s3://{self._bucket}/{s3_prefix}"
            logger.info(f"Uploaded {file_count} files to {s3_uri}")
            return s3_uri

        except ClientError as e:
            raise S3OperationError(
                message=f"Failed to upload template to S3: {e}",
                error_code="S3_UPLOAD_FAILED",
            ) from e

    def process_template(
        self,
        unique_id: str,
        requirements_md: str,
        model_details_md: str,
    ) -> str:
        """Download base template, inject documents, upload, and return S3 URI.

        Args:
            unique_id: Unique identifier for the template (generation_details_id).
            requirements_md: Generated requirements.md content.
            model_details_md: Generated model_details.md content.

        Returns:
            S3 URI of the uploaded template.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Step 1: Download base template
            self.download_base_template(tmp_dir)

            # Step 2: Inject generated documents into template/design/
            design_dir = os.path.join(tmp_dir, S3_DESIGN_FOLDER)
            os.makedirs(design_dir, exist_ok=True)

            requirements_path = os.path.join(design_dir, "requirements.md")
            Path(requirements_path).write_text(requirements_md, encoding="utf-8")

            model_details_path = os.path.join(design_dir, "model_details.md")
            Path(model_details_path).write_text(model_details_md, encoding="utf-8")

            logger.info("Injected requirements.md and model_details.md into template/design/")

            # Step 3: Upload complete template
            return self.upload_template(tmp_dir, unique_id)

    async def check_accessibility(self) -> bool:
        """Check if the S3 bucket is accessible."""
        try:
            self._s3.head_bucket(Bucket=self._bucket)
            return True
        except ClientError:
            return False
