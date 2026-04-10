from __future__ import annotations

from typing import Sequence

from saltai.utils.typing.core import ArtifactRef, ArtifactStore
from saltai.utils.typing.json_types import JSONObject, PathLike


class BaseArtifactStore(ArtifactStore):
    def put(self, local_path: PathLike, *, kind: str, name: str, meta: JSONObject | None = None) -> ArtifactRef:
        # ignore: Type of 'put' is incompatible with 'ArtifactStore'
        pass

    def get(self, ref: ArtifactRef, *, dst_dir: PathLike) -> PathLike:
        # ignore: Type of 'put' is incompatible with 'ArtifactStore'
        pass

    def exists(self, ref: ArtifactRef) -> bool:
        raise NotImplementedError

    def list(self, *, kind: str | None = None) -> Sequence[ArtifactRef]:
        # ignore: Type of 'put' is incompatible with 'ArtifactStore'
        pass