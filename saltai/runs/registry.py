from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _optional_float(value: Any) -> float | None:
    if not _is_number(value):
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value


def _dict_or_empty(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return value


def _flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}

    for key, value in metrics.items():
        name = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            out.update(_flatten_metrics(value, name))
        else:
            out[name] = value

    return out


@dataclass(frozen=True, slots=True)
class RunRecord(object):
    run_id: str
    run_dir: str
    manifest_path: str
    status: str
    started_ts: float | None
    finished_ts: float | None
    config_hash: str | None
    metrics: dict[str, Any]
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    error: dict[str, Any] | None
    extra: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def duration_s(self) -> float | None:
        if self.started_ts is None or self.finished_ts is None:
            return None
        return self.finished_ts - self.started_ts

    def metric(self, path: str, default: Any = None) -> Any:
        if not path:
            return default

        value: Any = self.metrics
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]

        return value

    def artifacts(self, *, kind: str | None = None) -> list[dict[str, Any]]:
        value = self.outputs.get("artifacts")
        if not isinstance(value, list):
            return []

        refs: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            if kind is not None and item.get("kind") != kind:
                continue
            refs.append(item)

        return refs

    def artifact(
            self,
            name: str,
            *,
            kind: str | None = None,
            default: Any = None,
    ) -> dict[str, Any] | Any:
        for ref in self.artifacts(kind=kind):
            if ref.get("name") == name:
                return ref

        return default

    def checkpoint(self, tag: str, default: Any = None) -> dict[str, Any] | Any:
        checkpoints = self.outputs.get("checkpoints")
        if not isinstance(checkpoints, dict):
            return default

        value = checkpoints.get(tag)
        if not isinstance(value, dict):
            return default

        return value

    @property
    def latest_checkpoint(self) -> dict[str, Any] | None:
        return self.checkpoint("latest", default=None)

    @property
    def best_checkpoint(self) -> dict[str, Any] | None:
        return self.checkpoint("best", default=None)

    @property
    def resume_checkpoint(self) -> dict[str, Any] | None:
        return self.checkpoint("resume_from", default=None)

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "manifest_path": self.manifest_path,
            "status": self.status,
            "started_ts": self.started_ts,
            "finished_ts": self.finished_ts,
            "duration_s": self.duration_s,
            "config_hash": self.config_hash,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error": self.error,
            "extra": self.extra,
        }

        for key, value in _flatten_metrics(self.metrics).items():
            row[f"metric.{key}"] = value

        return row


class RunRegistry(object):
    def __init__(self, root: str):
        self.root = os.fspath(root)

    def list(
            self,
            *,
            status: str | None = None,
            config_hash: str | None = None,
    ) -> list[RunRecord]:
        if not os.path.isdir(self.root):
            return []

        records: list[RunRecord] = []

        for dirpath, _, filenames in os.walk(self.root):
            if "manifest.json" not in filenames:
                continue

            manifest_path = os.path.join(dirpath, "manifest.json")
            record = self._load_record(manifest_path)
            if record is None:
                continue

            if status is not None and record.status != status:
                continue

            if config_hash is not None and record.config_hash != config_hash:
                continue

            records.append(record)

        records.sort(key=lambda r: (r.started_ts is None, -(r.started_ts or 0.0)))
        return records

    def get(self, run_id: str) -> RunRecord | None:
        for record in self.list():
            if record.run_id == run_id:
                return record
        return None

    def best(
            self,
            metric: str,
            *,
            mode: str = "max",
            status: str | None = "finished",
    ) -> RunRecord | None:
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")

        candidates: list[tuple[float, RunRecord]] = []

        for record in self.list(status=status):
            value = record.metric(metric)
            if not _is_number(value):
                continue
            candidates.append((float(value), record))

        if not candidates:
            return None

        if mode == "max":
            return max(candidates, key=lambda item: item[0])[1]

        return min(candidates, key=lambda item: item[0])[1]

    def latest(
            self,
            *,
            status: str | None = None,
            config_hash: str | None = None,
    ) -> RunRecord | None:
        records = self.list(status=status, config_hash=config_hash)
        if not records:
            return None
        return records[0]

    def count(
            self,
            *,
            status: str | None = None,
            config_hash: str | None = None,
    ) -> int:
        return len(self.list(status=status, config_hash=config_hash))

    def status_counts(
            self,
            *,
            config_hash: str | None = None,
    ) -> dict[str, int]:
        counts: dict[str, int] = {}

        for record in self.list(config_hash=config_hash):
            counts[record.status] = counts.get(record.status, 0) + 1

        return counts

    def metric_paths(
            self,
            *,
            status: str | None = None,
            config_hash: str | None = None,
    ) -> list[str]:
        paths: set[str] = set()

        for record in self.list(status=status, config_hash=config_hash):
            paths.update(_flatten_metrics(record.metrics).keys())

        return sorted(paths)

    def to_rows(
            self,
            *,
            status: str | None = None,
            config_hash: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            record.to_row()
            for record in self.list(status=status, config_hash=config_hash)
        ]

    def to_dataframe(
            self,
            *,
            status: str | None = None,
            config_hash: str | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            self.to_rows(status=status, config_hash=config_hash)
        )

    def to_csv(
            self,
            path: str,
            *,
            status: str | None = None,
            config_hash: str | None = None,
    ) -> None:
        self.to_dataframe(status=status, config_hash=config_hash).to_csv(
            os.fspath(path),
            index=False,
        )

    def to_jsonl(
            self,
            path: str,
            *,
            status: str | None = None,
            config_hash: str | None = None,
    ) -> None:
        self.to_dataframe(status=status, config_hash=config_hash).to_json(
            os.fspath(path),
            orient="records",
            lines=True,
            force_ascii=False,
        )

    def _load_record(self, manifest_path: str) -> RunRecord | None:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None

        if not isinstance(manifest, dict):
            return None

        run_id = _optional_str(manifest.get("run_id"))
        if run_id is None:
            return None

        error = manifest.get("error")
        if error is not None and not isinstance(error, dict):
            error = None

        return RunRecord(
            run_id=run_id,
            run_dir=os.path.dirname(manifest_path),
            manifest_path=manifest_path,
            status=_optional_str(manifest.get("status")) or "",
            started_ts=_optional_float(manifest.get("started_ts")),
            finished_ts=_optional_float(manifest.get("finished_ts")),
            config_hash=_optional_str(manifest.get("config_hash")),
            metrics=_dict_or_empty(manifest.get("metrics")),
            inputs=_dict_or_empty(manifest.get("inputs")),
            outputs=_dict_or_empty(manifest.get("outputs")),
            error=error,
            extra=_dict_or_empty(manifest.get("extra")),
            manifest=manifest,
        )
