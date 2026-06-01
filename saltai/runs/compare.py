from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from saltai.runs.registry import RunRecord, RunRegistry

DEFAULT_COMPARE_FIELDS = (
    "run_id",
    "status",
    "started_ts",
    "finished_ts",
    "duration_s",
    "config_hash",
)


def _normalize_metrics(metrics: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(metrics, str):
        return (metrics,)
    return tuple(metrics)


def _record_field(record: RunRecord, field: str, default: Any) -> Any:
    if not field:
        return default
    return getattr(record, field, default)


def compare_records(
        records: Iterable[RunRecord],
        metrics: str | Sequence[str],
        *,
        fields: Sequence[str] | None = None,
        default: Any = None,
) -> list[dict[str, Any]]:
    metric_paths = _normalize_metrics(metrics)
    selected_fields = DEFAULT_COMPARE_FIELDS if fields is None else tuple(fields)

    rows: list[dict[str, Any]] = []

    for record in records:
        row = {
            field: _record_field(record, field, default)
            for field in selected_fields
        }

        for metric in metric_paths:
            row[f"metric.{metric}"] = record.metric(metric, default=default)

        rows.append(row)

    return rows


def compare_runs(
        registry: RunRegistry,
        metrics: str | Sequence[str],
        *,
        status: str | None = None,
        config_hash: str | None = None,
        fields: Sequence[str] | None = None,
        default: Any = None,
) -> list[dict[str, Any]]:
    return compare_records(
        registry.list(status=status, config_hash=config_hash),
        metrics,
        fields=fields,
        default=default,
    )
