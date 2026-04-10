from __future__ import annotations

from saltai.artifacts.refs import (
    artifact_local_path,
    artifact_ref_scheme,
    artifact_uri_scheme,
    is_local_artifact_ref,
    make_artifact_ref,
    validate_artifact_key,
    validate_artifact_ref,
)
from saltai.artifacts.store import LocalArtifactStore, S3ArtifactStore

__all__ = (
    "LocalArtifactStore",
    "S3ArtifactStore",
    "artifact_uri_scheme",
    "artifact_ref_scheme",
    "artifact_local_path",
    "is_local_artifact_ref",
    "validate_artifact_key",
    "validate_artifact_ref",
    "make_artifact_ref",
)
