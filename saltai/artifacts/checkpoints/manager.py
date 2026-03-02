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
        Path(os.path.join(self.root, "_tmp")).mkdir(parents=True, exist_ok=True)

        self._store = LocalArtifactStore(root=os.path.join(self.root, "_store"))

        self._latest: list[ArtifactRef] = []
        self._best: ArtifactRef | None = None
        self._best_metric: float | None = None

    @staticmethod
    def path_of(ref: ArtifactRef) -> str:
        return ref.uri.replace("file://", "", 1)

    def _tmp_path(self, filename: str) -> str:
        return os.path.join(self.root, "_tmp", filename)

    @staticmethod
    def _atomic_json_dump(path: str, payload: dict[str, Any]) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)

    def _build_payload(self, obj: Checkpointable, payload: dict[str, Any]) -> dict[str, Any]:
        out = dict(payload)
        out["state"] = dict(obj.state_dict())
        return out

    def _read_header(self, ref: ArtifactRef) -> tuple[str | None, int | None, float | None]:
        p = self.path_of(ref)
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
            tag = payload.get("tag")
            step = payload.get("step")
            metric = payload.get("metric")

            tag_s = tag if isinstance(tag, str) else None
            step_i = int(step) if isinstance(step, int) else None
            metric_f = float(metric) if isinstance(metric, (int, float)) else None

            return tag_s, step_i, metric_f
        except Exception:
            return None, None, None

    def save_latest(self, obj: Checkpointable, *, step: int) -> ArtifactRef:
        step_i = int(step)
        tmp_json = self._tmp_path(f"latest_step_{step_i}.ckpt.json")
        ref: ArtifactRef | None = None

        try:
            payload = self._build_payload(obj, {"step": step_i, "ts": time.time(), "tag": "latest"})
            self._atomic_json_dump(tmp_json, payload)
            ref = self._store.put(
                tmp_json,
                kind="ckpt",
                name=f"latest_step_{step_i}",
                meta={"step": step_i, "tag": "latest"},
            )
        except BaseException as e:
            raise CheckpointError(
                EC.CKPT_SAVE_FAILED,
                "Failed to save latest checkpoint",
                hint="Check filesystem permissions and free space",
                context={"tmp_path": tmp_json, "step": step_i},
                cause=e,
            ) from e
        finally:
            try:
                os.remove(tmp_json)
            except Exception:
                pass

        assert ref is not None
        self._latest.append(ref)
        while len(self._latest) > self.keep_last:
            old = self._latest.pop(0)
            try:
                os.remove(self.path_of(old))
            except Exception:
                pass

        return ref

    def save_best(self, obj: Checkpointable, *, metric: float, step: int) -> ArtifactRef:
        step_i = int(step)
        metric_f = float(metric)

        if self._best_metric is not None and metric_f <= self._best_metric:
            return self._best  # type: ignore[return-value]

        tmp_json = self._tmp_path(f"best_step_{step_i}.ckpt.json")
        ref: ArtifactRef | None = None

        try:
            payload = self._build_payload(obj, {"step": step_i, "ts": time.time(), "tag": "best", "metric": metric_f})
            self._atomic_json_dump(tmp_json, payload)
            ref = self._store.put(
                tmp_json,
                kind="ckpt",
                name=f"best_step_{step_i}",
                meta={"step": step_i, "tag": "best", "metric": metric_f},
            )
        except BaseException as e:
            raise CheckpointError(
                EC.CKPT_SAVE_FAILED,
                "Failed to save best checkpoint",
                hint="Check filesystem permissions and free space",
                context={"tmp_path": tmp_json, "step": step_i, "metric": metric_f},
                cause=e,
            ) from e
        finally:
            try:
                os.remove(tmp_json)
            except Exception:
                pass

        assert ref is not None
        if self._best is not None:
            try:
                os.remove(self.path_of(self._best))
            except Exception:
                pass

        self._best = ref
        self._best_metric = metric_f
        return ref

    def read(self, ref: ArtifactRef) -> dict[str, Any]:
        p = self.path_of(ref)
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
            state = payload.get("state")
            if not isinstance(state, dict):
                raise ValueError("Invalid checkpoint payload: missing 'state'")
            return payload
        except BaseException as e:
            raise CheckpointError(
                EC.CKPT_LOAD_FAILED,
                "Failed to read checkpoint",
                hint="Check checkpoint file integrity",
                context={"path": p, "ref_name": ref.name, "ref_kind": ref.kind},
                cause=e,
            ) from e

    def load(self, obj: Checkpointable, ref: ArtifactRef) -> dict[str, Any]:
        payload = self.read(ref)
        try:
            obj.load_state_dict(payload["state"])
            return payload
        except BaseException as e:
            raise CheckpointError(
                EC.CKPT_LOAD_FAILED,
                "Failed to load checkpoint into object",
                hint="Check compatibility of state_dict",
                context={"ref_name": ref.name, "ref_kind": ref.kind},
                cause=e,
            ) from e

    def find_latest(self) -> ArtifactRef | None:
        items = self._store.list(kind="ckpt")
        best_ref: ArtifactRef | None = None
        best_step: int | None = None

        for r in items:
            tag, step, _ = self._read_header(r)
            if tag != "latest":
                continue
            if step is None:
                continue
            if best_step is None or step > best_step:
                best_step = step
                best_ref = r

        return best_ref

    def find_best(self) -> ArtifactRef | None:
        items = self._store.list(kind="ckpt")
        best_ref: ArtifactRef | None = None
        best_metric: float | None = None
        best_step: int | None = None

        for r in items:
            tag, step, metric = self._read_header(r)
            if tag != "best":
                continue
            if metric is None:
                continue
            if step is None:
                continue

            if (
                best_metric is None
                or metric > best_metric
                or (metric == best_metric and (best_step is None or step > best_step))
            ):
                best_metric = metric
                best_step = step
                best_ref = r

        return best_ref

    def resolve(self, which: str) -> ArtifactRef | None:
        w = str(which).strip().lower()
        if w == "latest":
            return self.find_latest()
        if w == "best":
            return self.find_best()
        return None
