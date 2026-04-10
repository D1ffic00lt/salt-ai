from __future__ import annotations

from saltai.artifacts.store.local import LocalArtifactStore
from saltai.artifacts.store.s3 import S3ArtifactStore

__all__ = (
    "LocalArtifactStore",
    "S3ArtifactStore"
)
