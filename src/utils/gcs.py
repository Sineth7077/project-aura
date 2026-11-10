"""Google Cloud Storage utilities for Project Aura 2.0.

Provides helper functions for uploading/downloading models, datasets,
and artifacts to/from GCS. Gracefully degrades when the google-cloud-
storage package is not installed (e.g., minimal Colab runtime).
"""

from __future__ import annotations

import os
from typing import Optional


class GCSClient:
    """Wrapper around Google Cloud Storage client."""

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
    ):
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.credentials_path = credentials_path
        self._client = None
        self._bucket = None

    def _ensure_client(self) -> bool:
        """Lazily initialize the GCS client. Returns True if successful."""
        if self._client is not None:
            return True

        try:
            from google.cloud import storage  # type: ignore

            if self.credentials_path and os.path.exists(self.credentials_path):
                self._client = storage.Client.from_service_account_json(
                    self.credentials_path, project=self.project_id
                )
            else:
                self._client = storage.Client(project=self.project_id)

            if self.bucket_name:
                self._bucket = self._client.bucket(self.bucket_name)
            return True
        except ImportError:
            # google-cloud-storage not installed — Colab minimal environment
            return False
        except Exception:
            return False

    def upload_file(
        self,
        local_path: str,
        destination_blob_name: str,
        bucket_name: Optional[str] = None,
    ) -> bool:
        """Upload a file to GCS.

        Args:
            local_path: Path to the local file.
            destination_blob_name: Destination blob name in the bucket.
            bucket_name: Bucket name (uses instance default if not provided).

        Returns:
            True if upload succeeded, False otherwise.
        """
        if not self._ensure_client():
            return False

        bucket = self._client.bucket(bucket_name or self.bucket_name)
        if not bucket:
            return False

        try:
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(local_path)
            return True
        except Exception:
            return False

    def download_file(
        self,
        source_blob_name: str,
        local_path: str,
        bucket_name: Optional[str] = None,
    ) -> bool:
        """Download a file from GCS.

        Args:
            source_blob_name: Source blob name in the bucket.
            local_path: Local path to save the file.
            bucket_name: Bucket name (uses instance default if not provided).

        Returns:
            True if download succeeded, False otherwise.
        """
        if not self._ensure_client():
            return False

        bucket = self._client.bucket(bucket_name or self.bucket_name)
        if not bucket:
            return False

        try:
            blob = bucket.blob(source_blob_name)
            blob.download_to_filename(local_path)
            return True
        except Exception:
            return False

    def list_blobs(
        self,
        prefix: str = "",
        bucket_name: Optional[str] = None,
    ) -> list[str]:
        """List blob names in the bucket with an optional prefix.

        Args:
            prefix: Blob name prefix to filter.
            bucket_name: Bucket name (uses instance default if not provided).

        Returns:
            List of blob names.
        """
        if not self._ensure_client():
            return []

        bucket = self._client.bucket(bucket_name or self.bucket_name)
        if not bucket:
            return []

        try:
            blobs = bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs]
        except Exception:
            return []


def get_gcs_client(
    bucket_name: Optional[str] = None,
    project_id: Optional[str] = None,
    credentials_path: Optional[str] = None,
) -> GCSClient:
    """Factory function to create a GCSClient instance."""
    return GCSClient(
        bucket_name=bucket_name,
        project_id=project_id,
        credentials_path=credentials_path,
    )