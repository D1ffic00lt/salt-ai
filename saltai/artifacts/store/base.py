from __future__ import annotations

from typing import Sequence

from saltai.utils.typing.core import ArtifactRef, ArtifactStore
from saltai.utils.typing.json_types import JSONObject, PathLike


class BaseArtifactStore(ArtifactStore):
    def put(  # type: ignore
            self,
            local_path: PathLike,
            *,
            kind: str,
            name: str,
            meta: JSONObject | None = None
    ) -> ArtifactRef:
        pass

    def get(self, ref: ArtifactRef, *, dst_dir: PathLike) -> PathLike:  # type: ignore
        pass

    def exists(self, ref: ArtifactRef) -> bool:
        raise NotImplementedError

    def list(self, *, kind: str | None = None) -> Sequence[ArtifactRef]:  # type: ignore
        pass
