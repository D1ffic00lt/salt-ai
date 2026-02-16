from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from pathlib import Path

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


def _strip_file_uri(uri: str) -> str:
    return uri[7:] if uri.startswith("file://") else uri


class LocalArtifactStore(object):
    def __init__(self, root: str):
        self.root = str(root)
        Path(self.root).mkdir(parents=True, exist_ok=True)

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
        dst_dir = os.path.join(self.root, kind)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, f"{name}__{aid}{ext}")

        try:
            shutil.copy2(local_path, dst)
            size = os.path.getsize(dst)
            sha = _sha256_file(dst)
        except BaseException as e:
            raise ArtifactError(
                EC.ARTIFACT_WRITE_FAILED,
                "Failed to store artifact",
                hint="Check filesystem permissions and free space",
                context={"src": local_path, "dst": dst, "kind": kind, "name": name},
                cause=e,
            ) from e

        return ArtifactRef(
            id=aid,
            kind=kind,
            name=name,
            uri=f"file://{dst}",
            sha256=sha,
            size_bytes=size,
            meta=meta or {},
        )

    def exists(self, ref: ArtifactRef) -> bool:
        return os.path.exists(_strip_file_uri(ref.uri))

    @staticmethod
    def get(ref: ArtifactRef, *, dst_dir: str) -> str:
        src = _strip_file_uri(ref.uri)
        if not os.path.exists(src):
            raise ArtifactError(
                EC.ARTIFACT_NOT_FOUND,
                "Artifact not found in store",
                hint="Check artifact uri and store root",
                context={"uri": ref.uri, "kind": ref.kind, "name": ref.name},
            )

        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(src))
        try:
            shutil.copy2(src, dst)
        except BaseException as e:
            raise ArtifactError(
                EC.ARTIFACT_READ_FAILED,
                "Failed to retrieve artifact",
                hint="Check filesystem permissions",
                context={"src": src, "dst": dst},
                cause=e,
            ) from e
        return dst

    def list(self, *, kind: str | None = None):
        base = os.path.join(self.root, kind) if kind else self.root
        if not os.path.exists(base):
            return []

        out = []
        for root, _, files in os.walk(base):
            for fn in files:
                if fn.startswith("."):
                    continue
                path = os.path.join(root, fn)

                if kind is None:
                    rel = os.path.relpath(path, self.root)
                    k = rel.split(os.sep, 1)[0]
                else:
                    k = kind

                name = fn.split("__", 1)[0]
                out.append(
                    ArtifactRef(
                        id=ArtifactId(""),
                        kind=k,
                        name=name,
                        uri=f"file://{path}",
                        sha256=None,
                        size_bytes=os.path.getsize(path),
                        meta={},
                    )
                )
        return out
