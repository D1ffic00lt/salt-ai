from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from saltai.utils.errors.base import ArtifactError
from saltai.utils.errors.codes import EC
from saltai.utils.typing.core import ArtifactId, ArtifactRef
from saltai.utils.typing.json_types import JSONObject


class Boto3NotInstalledError(ImportError):
    pass


def _load_boto3_client() -> Any:
    try:
        import boto3
    except ImportError as e:
        raise Boto3NotInstalledError(
            "boto3 is not installed. Install it with `pip install salt-ai[s3]` "
            "or `poetry install -E s3`."
        ) from e
    return boto3.client


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    key = parsed.path.lstrip("/")
    if parsed.scheme != "s3" or not parsed.netloc or not key:
        raise ValueError(f"Invalid S3 artifact URI: {uri}")
    return parsed.netloc, key


class S3ArtifactStore(object):
    def __init__(
            self,
            *,
            bucket: str,
            prefix: str = "",
            client: Any | None = None,
            client_kwargs: Mapping[str, Any] | None = None,
    ):
        if not bucket:
            raise ValueError("bucket is required")

        self.bucket = bucket
        self.prefix = prefix.strip("/")

        if client is None:
            boto3_client = _load_boto3_client()
            client = boto3_client("s3", **dict(client_kwargs or {}))

        self.client = client

    def put(self, local_path: str, *, kind: str, name: str, meta: JSONObject | None = None) -> ArtifactRef:
        if not os.path.exists(local_path):
            raise ArtifactError(
                EC.ARTIFACT_NOT_FOUND,
                "Local artifact file not found",
                hint="Check the path you pass to put()",
                context={"path": local_path, "kind": kind, "name": name},
            )

        aid = ArtifactId(uuid.uuid4().hex)
        ext = Path(local_path).suffix
        key = self._key_for(kind=kind, name=name, artifact_id=aid, ext=ext)

        try:
            size = os.path.getsize(local_path)
            sha = _sha256_file(local_path)
            self.client.upload_file(local_path, self.bucket, key)
        except BaseException as e:
            raise ArtifactError(
                EC.ARTIFACT_WRITE_FAILED,
                "Failed to store artifact in S3",
                hint="Check S3 credentials, bucket permissions, and object key",
                context={"path": local_path, "bucket": self.bucket, "key": key, "kind": kind, "name": name},
                cause=e,
            ) from e

        return ArtifactRef(
            id=aid,
            kind=kind,
            name=name,
            uri=f"s3://{self.bucket}/{key}",
            sha256=sha,
            size_bytes=size,
            meta=meta or {},
        )

    def get(self, ref: ArtifactRef, *, dst_dir: str) -> str:
        try:
            bucket, key = _parse_s3_uri(ref.uri)
        except ValueError as e:
            raise ArtifactError(
                EC.ARTIFACT_READ_FAILED,
                "Invalid S3 artifact URI",
                hint="Expected URI format: s3://bucket/key",
                context={"uri": ref.uri, "kind": ref.kind, "name": ref.name},
                cause=e,
            ) from e

        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(key))

        try:
            self.client.download_file(bucket, key, dst)
        except BaseException as e:
            raise ArtifactError(
                EC.ARTIFACT_READ_FAILED,
                "Failed to retrieve artifact from S3",
                hint="Check S3 credentials, bucket permissions, and artifact URI",
                context={"bucket": bucket, "key": key, "dst": dst, "kind": ref.kind, "name": ref.name},
                cause=e,
            ) from e

        return dst

    def exists(self, ref: ArtifactRef) -> bool:
        try:
            bucket, key = _parse_s3_uri(ref.uri)
            self.client.head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False

    def list(self, *, kind: str | None = None) -> list[ArtifactRef]:
        prefix = self._list_prefix(kind)
        out: list[ArtifactRef] = []
        token: str | None = None

        while True:
            kwargs: dict[str, Any] = {
                "Bucket": self.bucket,
                "Prefix": prefix,
            }
            if token is not None:
                kwargs["ContinuationToken"] = token

            response = self.client.list_objects_v2(**kwargs)

            for item in response.get("Contents", []):
                key = item.get("Key")
                if not isinstance(key, str) or key.endswith("/"):
                    continue

                ref_kind = kind.strip("/") if kind is not None else self._kind_from_key(key)
                if not ref_kind:
                    continue

                out.append(
                    ArtifactRef(
                        id=ArtifactId(""),
                        kind=ref_kind,
                        name=self._name_from_key(key),
                        uri=f"s3://{self.bucket}/{key}",
                        sha256=None,
                        size_bytes=item.get("Size"),
                        meta={},
                    )
                )

            token = response.get("NextContinuationToken")
            if token is None:
                break

        return out

    def _base_prefix(self) -> str:
        if not self.prefix:
            return ""
        return f"{self.prefix}/"

    def _list_prefix(self, kind: str | None) -> str:
        base = self._base_prefix()
        if kind is None:
            return base
        return f"{base}{kind.strip('/')}/"

    def _key_for(self, *, kind: str, name: str, artifact_id: ArtifactId, ext: str) -> str:
        return f"{self._list_prefix(kind)}{name.strip('/')}__{artifact_id}{ext}"

    def _kind_from_key(self, key: str) -> str:
        base = self._base_prefix()
        rel = key[len(base):] if base and key.startswith(base) else key
        if "/" not in rel:
            return ""
        return rel.split("/", 1)[0]

    @staticmethod
    def _name_from_key(key: str) -> str:
        filename = key.rsplit("/", 1)[-1]
        return filename.split("__", 1)[0]
