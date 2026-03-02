from __future__ import annotations

import json
import os
import time

from pathlib import Path
from typing import Any

from saltai.artifacts.store.local import LocalArtifactStore
from saltai.utils.errors.base import CheckpointError
from saltai.utils.errors.codes import EC
from saltai.utils.typing.core import ArtifactRef, Checkpointable


class CheckpointManager(object):
    def __init__(self, root: str, *, keep_last: int = 3):
        self.root = str(root)
        self.keep_last = int(keep_last)
        Path(self.root).mkdir(parents=True, exist_ok=True)
        self._store = LocalArtifactStore(root=os.path.join(self.root, "_store"))

        self._latest: list[ArtifactRef] = []
        self._best: ArtifactRef | None = None
        self._best_metric: float | None = None

    @staticmethod
    def _dump_state(obj: Checkpointable, path: str, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload["state"] = obj.state_dict()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)

    @staticmethod
    def path_of(ref: ArtifactRef) -> str:
        return ref.uri.replace("file://", "", 1)

    def save_latest(self, obj: Checkpointable, *, step: int) -> ArtifactRef:
        p = os.path.join(self.root, f"latest_step_{int(step)}.ckpt.json")
        try:
            self._dump_state(obj, p, {"step": int(step), "ts": time.time(), "tag": "latest"})
            ref = self._store.put(
                p, kind="ckpt", name=f"latest_step_{int(step)}", meta={"step": int(step), "tag": "latest"}
            )
        except BaseException as e:
            raise CheckpointError(
                EC.CKPT_SAVE_FAILED,
                "Failed to save latest checkpoint",
                hint="Check filesystem permissions and free space",
                context={"path": p, "step": int(step)},
                cause=e,
            ) from e

        self._latest.append(ref)
        while len(self._latest) > self.keep_last:
            old = self._latest.pop(0)
            try:
                os.remove(self.path_of(old))
            except Exception as ex:
                pass
        return ref

    def save_best(self, obj: Checkpointable, *, metric: float, step: int) -> ArtifactRef:
        metric = float(metric)
        if self._best_metric is not None and metric <= self._best_metric:
            return self._best  # type: ignore[return-value]

        p = os.path.join(self.root, f"best_step_{int(step)}.ckpt.json")
        try:
            self._dump_state(obj, p, {"step": int(step), "ts": time.time(), "tag": "best", "metric": metric})
            ref = self._store.put(
                p, kind="ckpt", name=f"best_step_{int(step)}", meta={"step": int(step), "tag": "best", "metric": metric}
            )
        except BaseException as e:
            raise CheckpointError(
                EC.CKPT_SAVE_FAILED,
                "Failed to save best checkpoint",
                hint="Check filesystem permissions and free space",
                context={"path": p, "step": int(step), "metric": metric},
                cause=e,
            ) from e

        if self._best is not None:
            try:
                os.remove(self.path_of(self._best))
            except Exception as ex:
                pass

        self._best = ref
        self._best_metric = metric
        return ref

    def load(self, obj: Checkpointable, ref: ArtifactRef) -> dict[str, Any]:
        p = self.path_of(ref)
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
            state = payload.get("state")
            if not isinstance(state, dict):
                raise ValueError("Invalid checkpoint payload: missing 'state'")
            obj.load_state_dict(state)
            return payload
        except BaseException as e:
            raise CheckpointError(
                EC.CKPT_LOAD_FAILED,
                "Failed to load checkpoint",
                hint="Check checkpoint file and compatibility of state_dict",
                context={"path": p, "ref_name": ref.name, "ref_kind": ref.kind},
                cause=e,
            ) from e
