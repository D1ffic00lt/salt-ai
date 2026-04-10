from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any

from saltai.artifacts.refs import make_artifact_ref, validate_artifact_key, validate_artifact_ref
from saltai.utils.errors.base import ArtifactError
from saltai.utils.errors.codes import EC
from saltai.utils.typing.core import ArtifactId, ArtifactRef
from saltai.utils.typing.json_types import JSONObject


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _strip_slashes(value: str) -> str:
    return str(value).strip().strip("/")


def _join_s3_key(*parts: str) -> str:
    clean = [_strip_slashes(p) for p in parts if _strip_slashes(p)]
    return "/".join(clean)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    uri_s = str(uri).strip()
    if not uri_s.startswith("s3://"):
        raise ArtifactError(
            EC.ARTIFACT_INVALID_REF,
            "Artifact uri must be an s3:// uri",
            hint="Use refs produced by S3ArtifactStore or pass a valid s3://bucket/key uri",
            context={"uri": uri},
        )

    rest = uri_s[5:]
    if "/" not in rest:
        raise ArtifactError(
            EC.ARTIFACT_INVALID_REF,
            "Artifact s3 uri must include bucket and key",
            hint="Expected format: s3://bucket/key",
            context={"uri": uri},
        )

    bucket, key = rest.split("/", 1)
    bucket = bucket.strip()
    key = key.strip().lstrip("/")

    if not bucket or not key:
        raise ArtifactError(
            EC.ARTIFACT_INVALID_REF,
            "Artifact s3 uri must include non-empty bucket and key",
            hint="Expected format: s3://bucket/key",
            context={"uri": uri},
        )

    return bucket, key


def _optional_boto3_client(
        *,
        endpoint_url: str | None,
        region_name: str | None,
        profile_name: str | None,
        client_kwargs: dict[str, Any],
):
    try:
        import boto3
    except ImportError as e:
        raise ArtifactError(
            EC.ARTIFACT_WRITE_FAILED,
            "boto3 is not installed",
            hint="Install salt-ai with the s3 extra",
            context={"extra": "s3"},
            cause=e,
        ) from e

    if profile_name is not None:
        session = boto3.Session(profile_name=profile_name)
        return session.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            **client_kwargs,
        )

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region_name,
        **client_kwargs,
    )


class S3ArtifactStore(object):
    def __init__(
            self,
            *,
            bucket: str,
            prefix: str = "",
            client: Any | None = None,
            endpoint_url: str | None = None,
            region_name: str | None = None,
            profile_name: str | None = None,
            client_kwargs: dict[str, Any] | None = None,
    ):
        self.bucket = str(bucket).strip()
        self.prefix = _strip_slashes(prefix)

        if not self.bucket:
            raise ArtifactError(
                EC.ARTIFACT_INVALID_REF,
                "S3 bucket must be non-empty",
                hint="Pass bucket='your-bucket-name'",
                context={"bucket": bucket},
            )

        if client is None:
            self.client = _optional_boto3_client(
                endpoint_url=endpoint_url,
                region_name=region_name,
                profile_name=profile_name,
                client_kwargs=client_kwargs or {},
            )
        else:
            self.client = client

    def put(self, local_path: str, *, kind: str, name: str, meta: JSONObject | None = None) -> ArtifactRef:
        kind, name = validate_artifact_key(kind=kind, name=name)
        src = str(local_path)

        if not os.path.isfile(src):
            raise ArtifactError(
                EC.ARTIFACT_NOT_FOUND,
                "Local artifact file not found",
                hint="Check the path you pass to put()",
                context={"path": src, "kind": kind, "name": name},
            )

        aid = ArtifactId(uuid.uuid4().hex)
        ext = Path(src).suffix
        key = _join_s3_key(self.prefix, kind, f"{name}__{aid}{ext}")

        try:
            size = os.path.getsize(src)
            sha = _sha256_file(src)
            self.client.upload_file(src, self.bucket, key)
        except BaseException as e:
            raise ArtifactError(
                EC.ARTIFACT_WRITE_FAILED,
                "Failed to store artifact in S3",
                hint="Check S3 credentials, bucket permissions and network access",
                context={"path": src, "bucket": self.bucket, "key": key, "kind": kind, "name": name},
                cause=e,
            ) from e

        return make_artifact_ref(
            id=aid,
            kind=kind,
            name=name,
            uri=f"s3://{self.bucket}/{key}",
            sha256=sha,
            size_bytes=size,
            meta=meta or {},
        )

    def get(self, ref: ArtifactRef, *, dst_dir: str) -> str:
        validate_artifact_ref(ref, allowed_schemes=("s3",))
        bucket, key = _parse_s3_uri(ref.uri)

        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(key))

        try:
            self.client.download_file(bucket, key, dst)
        except BaseException as e:
            raise ArtifactError(
                EC.ARTIFACT_READ_FAILED,
                "Failed to retrieve artifact from S3",
                hint="Check S3 credentials, bucket permissions and artifact uri",
                context={"uri": ref.uri, "bucket": bucket, "key": key, "dst": dst},
                cause=e,
            ) from e

        return dst

    def exists(self, ref: ArtifactRef) -> bool:
        try:
            validate_artifact_ref(ref, allowed_schemes=("s3",))
            bucket, key = _parse_s3_uri(ref.uri)
            self.client.head_object(Bucket=bucket, Key=key)
            return True
        except ArtifactError:
            return False
        except BaseException:
            return False

    def list(self, *, kind: str | None = None):
        if kind is not None:
            kind, _ = validate_artifact_key(kind=kind, name="_")

        prefix = _join_s3_key(self.prefix, kind or "")
        if prefix:
            prefix += "/"

        out = []
        token = None

        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": prefix}
            if token is not None:
                kwargs["ContinuationToken"] = token

            try:
                response = self.client.list_objects_v2(**kwargs)
            except BaseException as e:
                raise ArtifactError(
                    EC.ARTIFACT_READ_FAILED,
                    "Failed to list S3 artifacts",
                    hint="Check S3 credentials, bucket permissions and prefix",
                    context={"bucket": self.bucket, "prefix": prefix},
                    cause=e,
                ) from e

            for item in response.get("Contents", []):
                key = item.get("Key")
                if not isinstance(key, str) or key.endswith("/"):
                    continue

                parsed_kind, name, aid = self._parse_key(key, kind=kind)
                if parsed_kind is None:
                    continue

                out.append(
                    ArtifactRef(
                        id=aid,
                        kind=parsed_kind,
                        name=name,
                        uri=f"s3://{self.bucket}/{key}",
                        sha256=None,
                        size_bytes=item.get("Size"),
                        meta={},
                    )
                )

            if not response.get("IsTruncated"):
                break

            token = response.get("NextContinuationToken")
            if token is None:
                break

        return out

    def _parse_key(self, key: str, *, kind: str | None) -> tuple[str | None, str, ArtifactId]:
        rel = key
        if self.prefix:
            prefix = self.prefix + "/"
            if not rel.startswith(prefix):
                return None, "", ArtifactId("")
            rel = rel[len(prefix):]

        parts = rel.split("/", 1)
        if kind is None:
            if len(parts) != 2:
                return None, "", ArtifactId("")
            parsed_kind, filename = parts
        else:
            parsed_kind = kind
            filename = parts[-1]

        name, aid = self._parse_stored_artifact_filename(filename)
        return parsed_kind, name, aid

    @staticmethod
    def _parse_stored_artifact_filename(filename: str) -> tuple[str, ArtifactId]:
        if "__" not in filename:
            return Path(filename).stem, ArtifactId("")

        name, artifact_part = filename.rsplit("__", 1)
        artifact_id = Path(artifact_part).stem
        return name, ArtifactId(artifact_id)
