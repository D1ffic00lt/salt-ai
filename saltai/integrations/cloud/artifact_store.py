from __future__ import annotations

import hashlib
import mimetypes
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import quote

from saltai.integrations.cloud.client import CloudApiError, CloudClient
from saltai.logging.utils.jsonable import to_jsonable
from saltai.utils.typing.core import ArtifactId, ArtifactRef
from saltai.utils.typing.json_types import JSONObject

__all__ = (
    "CloudArtifactStore",
    "CloudArtifactStoreError",
)

StorageUriBuilder = Callable[[Path, str, str, dict[str, Any]], str]


class CloudArtifactStoreError(RuntimeError):
    pass


class CloudArtifactStore(object):
    def __init__(
            self,
            *,
            client: CloudClient,
            run_id: str,
            storage_uri_builder: StorageUriBuilder | None = None,
            upload_files: bool = False,
    ):
        if not run_id:
            raise ValueError("run_id is required")

        self.client = client
        self.run_id = str(run_id)
        self.storage_uri_builder = storage_uri_builder or _default_storage_uri
        self.upload_files = bool(upload_files)

    def put(
            self,
            local_path: str | os.PathLike[str],
            *,
            kind: str,
            name: str,
            meta: JSONObject | None = None,
    ) -> ArtifactRef:
        path = Path(local_path).expanduser().resolve()

        if not path.exists():
            raise CloudArtifactStoreError(f"Local artifact file not found: {path}")
        if not path.is_file():
            raise CloudArtifactStoreError(f"Local artifact path is not a file: {path}")

        kind_s = str(kind).strip()
        name_s = str(name).strip()

        if not kind_s:
            raise ValueError("kind is required")
        if not name_s:
            raise ValueError("name is required")

        size_bytes = path.stat().st_size
        sha256 = _sha256_file(path)
        content_type = mimetypes.guess_type(str(path))[0]

        storage_mode = "local_upload" if self.upload_files else "metadata_only"
        base_meta = _json_object(meta)
        request_meta = {
            **base_meta,
            "local_path": str(path),
            "storage_mode": storage_mode,
        }

        artifact = self.client.create_artifact(
            self.run_id,
            name_s,
            kind=kind_s,
            size_bytes=size_bytes,
            content_type=content_type,
            hash=sha256,
            meta=request_meta,
        )

        if self.upload_files:
            uploaded = self.client.upload_artifact_file(str(artifact["id"]), path)
            payload = _merge_artifact_payloads(artifact, uploaded)

            return _artifact_ref_from_payload(
                payload,
                fallback_id=str(artifact["id"]),
                fallback_kind=kind_s,
                fallback_name=name_s,
                fallback_uri=str(uploaded.get("storage_uri") or artifact.get("storage_uri") or ""),
                fallback_sha256=uploaded.get("hash") or sha256,
                fallback_size_bytes=uploaded.get("size_bytes") or size_bytes,
                fallback_meta=request_meta,
            )

        storage_uri = self.storage_uri_builder(path, kind_s, name_s, artifact)

        completed = self.client.complete_artifact(
            str(artifact["id"]),
            storage_uri=storage_uri,
            size_bytes=size_bytes,
            content_type=content_type,
            hash=sha256,
            meta=request_meta,
        )

        payload = _merge_artifact_payloads(artifact, completed)

        return _artifact_ref_from_payload(
            payload,
            fallback_id=str(artifact["id"]),
            fallback_kind=kind_s,
            fallback_name=name_s,
            fallback_uri=storage_uri,
            fallback_sha256=sha256,
            fallback_size_bytes=size_bytes,
            fallback_meta=request_meta,
        )

    def get(self, ref: ArtifactRef, *, dst_dir: str | os.PathLike[str]) -> str:
        payload = self.client.get_artifact(str(ref.id))
        storage_uri = str(payload.get("storage_uri") or payload.get("uri") or ref.uri)

        dst_root = Path(dst_dir).expanduser().resolve()
        dst_root.mkdir(parents=True, exist_ok=True)

        if storage_uri.startswith("file://"):
            src = Path(storage_uri[7:]).expanduser().resolve()
            if src.exists() and src.is_file():
                dst = dst_root / src.name
                shutil.copy2(src, dst)
                return str(dst)

        name = str(payload.get("name") or ref.name or ref.id)
        dst = dst_root / Path(name).name

        data = self.client.download_artifact_content(str(ref.id))
        dst.write_bytes(data)

        return str(dst)

    def exists(self, ref: ArtifactRef) -> bool:
        try:
            payload = self.client.get_artifact(str(ref.id))
        except CloudApiError as e:
            if e.status_code == 404:
                return False
            raise

        status = str(payload.get("status") or "").lower()
        return status not in {"deleted", "failed"}

    def list(self, *, kind: str | None = None) -> Sequence[ArtifactRef]:
        artifacts = self.client.list_artifacts(self.run_id)

        out: list[ArtifactRef] = []
        for item in artifacts:
            if kind is not None and str(item.get("kind")) != str(kind):
                continue

            out.append(
                _artifact_ref_from_payload(
                    item,
                    fallback_id=str(item.get("id") or ""),
                    fallback_kind=str(item.get("kind") or kind or "other"),
                    fallback_name=str(item.get("name") or item.get("id") or "artifact"),
                    fallback_uri=str(item.get("storage_uri") or item.get("uri") or ""),
                    fallback_sha256=item.get("hash") or item.get("sha256"),
                    fallback_size_bytes=item.get("size_bytes"),
                    fallback_meta=item.get("meta") or {},
                )
            )

        return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def _default_storage_uri(path: Path, kind: str, name: str, artifact: dict[str, Any]) -> str:
    return f"file://{path}"


def _artifact_ref_from_payload(
        payload: dict[str, Any],
        *,
        fallback_id: str,
        fallback_kind: str,
        fallback_name: str,
        fallback_uri: str,
        fallback_sha256: str | None,
        fallback_size_bytes: int | None,
        fallback_meta: dict[str, Any],
) -> ArtifactRef:
    artifact_id = str(payload.get("id") or fallback_id)
    kind = str(payload.get("kind") or fallback_kind)
    name = str(payload.get("name") or fallback_name)

    uri = str(payload.get("storage_uri") or payload.get("uri") or fallback_uri)
    if not uri:
        uri = f"saltai-cloud://artifacts/{quote(artifact_id, safe='')}"

    sha256 = payload.get("hash") or payload.get("sha256") or fallback_sha256
    size_bytes = payload.get("size_bytes", fallback_size_bytes)

    meta = _json_object(payload.get("meta") or payload.get("metadata") or fallback_meta)
    meta["cloud_artifact"] = to_jsonable(payload)

    return ArtifactRef(
        id=ArtifactId(artifact_id),
        kind=kind,
        name=name,
        uri=uri,
        sha256=str(sha256) if sha256 else None,
        size_bytes=int(size_bytes) if size_bytes is not None else None,
        meta=meta,
    )


def _merge_artifact_payloads(
        created: dict[str, Any],
        completed: Any,
) -> dict[str, Any]:
    if isinstance(completed, dict):
        return {
            **created,
            **completed,
        }

    return created


def _json_object(value: Any) -> dict[str, Any]:
    value = to_jsonable(value or {})

    if isinstance(value, dict):
        return value

    return {
        "value": value,
    }
