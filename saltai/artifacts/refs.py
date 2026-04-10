from __future__ import annotations

import os

from saltai.utils.errors.base import ArtifactError
from saltai.utils.errors.codes import EC
from saltai.utils.typing.core import ArtifactId, ArtifactRef
from saltai.utils.typing.json_types import JSONObject

__all__ = (
    "artifact_uri_scheme",
    "artifact_ref_scheme",
    "artifact_local_path",
    "is_local_artifact_ref",
    "validate_artifact_key",
    "validate_artifact_ref",
    "make_artifact_ref",
)


def _invalid_ref(message: str, *, context: dict) -> None:
    raise ArtifactError(
        EC.ARTIFACT_INVALID_REF,
        message,
        hint="Check artifact id, kind, name, uri, sha256 and size_bytes",
        context=context,
    )


def artifact_uri_scheme(uri: str) -> str:
    uri_s = str(uri).strip()
    if "://" not in uri_s:
        return ""
    return uri_s.split("://", 1)[0].lower()


def artifact_ref_scheme(ref: ArtifactRef) -> str:
    return artifact_uri_scheme(ref.uri)


def is_local_artifact_ref(ref: ArtifactRef) -> bool:
    return artifact_ref_scheme(ref) == "file"


def artifact_local_path(ref: ArtifactRef) -> str:
    validate_artifact_ref(ref, allowed_schemes=("file",))
    return ref.uri[7:]


def validate_artifact_key(*, kind: str, name: str) -> tuple[str, str]:
    kind_s = _validate_segment(kind, field="kind")
    name_s = _validate_segment(name, field="name")
    return kind_s, name_s


def validate_artifact_ref(
        ref: ArtifactRef,
        *,
        allowed_schemes: tuple[str, ...] | list[str] | None = None,
) -> ArtifactRef:
    if not isinstance(ref, ArtifactRef):
        _invalid_ref(
            "Invalid artifact ref object",
            context={"type": type(ref).__name__},
        )

    kind = str(ref.kind).strip()
    name = str(ref.name).strip()
    uri = str(ref.uri).strip()

    if not kind:
        _invalid_ref("Artifact kind must be non-empty", context={"field": "kind", "value": ref.kind})
    if not name:
        _invalid_ref("Artifact name must be non-empty", context={"field": "name", "value": ref.name})
    if not uri:
        _invalid_ref("Artifact uri must be non-empty", context={"field": "uri", "value": ref.uri})

    scheme = artifact_uri_scheme(uri)
    if not scheme:
        _invalid_ref("Artifact uri must include a scheme", context={"uri": uri})

    if allowed_schemes is not None:
        allowed = tuple(str(s).lower() for s in allowed_schemes)
        if scheme not in allowed:
            _invalid_ref(
                "Artifact uri scheme is not supported here",
                context={"uri": uri, "scheme": scheme, "allowed_schemes": list(allowed)},
            )

    if ref.size_bytes is not None and int(ref.size_bytes) < 0:
        _invalid_ref(
            "Artifact size_bytes must be non-negative",
            context={"size_bytes": ref.size_bytes},
        )

    if ref.sha256 is not None:
        sha = str(ref.sha256).strip().lower()
        if len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
            _invalid_ref(
                "Artifact sha256 must be a 64-character hex string",
                context={"sha256": ref.sha256},
            )

    return ref


def make_artifact_ref(
        *,
        id: ArtifactId | str,
        kind: str,
        name: str,
        uri: str,
        sha256: str | None = None,
        size_bytes: int | None = None,
        meta: JSONObject | None = None,
) -> ArtifactRef:
    ref = ArtifactRef(
        id=ArtifactId(str(id)),
        kind=str(kind),
        name=str(name),
        uri=str(uri),
        sha256=sha256,
        size_bytes=size_bytes,
        meta=meta or {},
    )
    return validate_artifact_ref(ref)


def _validate_segment(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        _invalid_ref(
            "Artifact key segment must be a string",
            context={"field": field, "type": type(value).__name__},
        )

    value_s = value.strip()
    if not value_s:
        _invalid_ref(
            "Artifact key segment must be non-empty",
            context={"field": field, "value": value},
        )

    if value_s in (".", ".."):
        _invalid_ref(
            "Artifact key segment must not be a relative path marker",
            context={"field": field, "value": value_s},
        )

    if os.sep in value_s or (os.altsep is not None and os.altsep in value_s) or "\x00" in value_s:
        _invalid_ref(
            "Artifact key segment must not contain path separators",
            context={"field": field, "value": value_s},
        )

    return value_s
