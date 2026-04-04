from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any
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


def _normalize_prefix(prefix: str) -> str:
    return prefix.strip("/")


def _join_key(*parts: str) -> str:
    return "/".join(str(p).strip("/") for p in parts if str(p).strip("/"))


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ArtifactError(
            EC.ARTIFACT_READ_FAILED,
            "Invalid S3 artifact uri",
            hint="Expected uri format: s3://bucket/key",
            context={"uri": uri},
        )

    return parsed.netloc, parsed.path.lstrip("/")


class S3ArtifactStore(object):
    def __init__(
            self,
            *,
            bucket: str,
            prefix: str = "",
            client: Any | None = None,
            endpoint_url: str | None = None,
            region_name: str | None = None,
            client_kwargs: dict[str, Any] | None = None,
    ):
        self.bucket = str(bucket)
        self.prefix = _normalize_prefix(prefix)

        if client is None:
            make_client = _load_boto3_client()
            kwargs = dict(client_kwargs or {})
            if endpoint_url is not None:
                kwargs["endpoint_url"] = endpoint_url
            if region_name is not None:
                kwargs["region_name"] = region_name
            client = make_client("s3", **kwargs)

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
        key = _join_key(self.prefix, kind, f"{name}__{aid}{ext}")

        try:
            self.client.upload_file(local_path, self.bucket, key)
            size = os.path.getsize(local_path)
            sha = _sha256_file(local_path)
        except BaseException as e:
            raise ArtifactError(
                EC.ARTIFACT_WRITE_FAILED,
                "Failed to upload artifact to S3",
                hint="Check S3 credentials, bucket, endpoint and network access",
                context={
                    "path": local_path,
                    "bucket": self.bucket,
                    "key": key,
                    "kind": kind,
                    "name": name,
                },
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
        bucket, key = _parse_s3_uri(ref.uri)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(key))

        try:
            self.client.download_file(bucket, key, dst)
        except BaseException as e:
            raise ArtifactError(
                EC.ARTIFACT_READ_FAILED,
                "Failed to download artifact from S3",
                hint="Check S3 credentials, artifact uri and network access",
                context={"bucket": bucket, "key": key, "dst": dst},
                cause=e,
            ) from e

        return dst

    def exists(self, ref: ArtifactRef) -> bool:
        bucket, key = _parse_s3_uri(ref.uri)

        try:
            self.client.head_object(Bucket=bucket, Key=key)
            return True
        except BaseException:
            return False

    def list(self, *, kind: str | None = None):
        prefix = _join_key(self.prefix, kind or "")

        kwargs = {
            "Bucket": self.bucket,
            "Prefix": prefix,
        }

        out = []
        while True:
            response = self.client.list_objects_v2(**kwargs)

            for obj in response.get("Contents", []):
                key = obj["Key"]
                parts = key.split("/")
                if self.prefix:
                    prefix_parts = self.prefix.split("/")
                    parts = parts[len(prefix_parts):]

                if len(parts) < 2:
                    continue

                artifact_kind = parts[0]
                filename = parts[-1]
                name = filename.split("__", 1)[0]

                out.append(
                    ArtifactRef(
                        id=ArtifactId(""),
                        kind=artifact_kind,
                        name=name,
                        uri=f"s3://{self.bucket}/{key}",
                        sha256=None,
                        size_bytes=int(obj.get("Size", 0)),
                        meta={},
                    )
                )

            token = response.get("NextContinuationToken")
            if token is None:
                break
            kwargs["ContinuationToken"] = token

        return tuple(out)
