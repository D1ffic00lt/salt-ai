from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any, TextIO

from saltai.runs import RunRecord, RunRegistry, compare_runs

DEFAULT_COMPARE_FIELDS = (
    "run_id",
    "status",
    "started_ts",
    "finished_ts",
    "duration_s",
    "config_hash",
)


def _json_default(value: Any) -> str:
    return str(value)


def _write_json(value: Any, out: TextIO) -> None:
    json.dump(value, out, ensure_ascii=False, indent=2, default=_json_default)
    out.write("\n")


def _write_jsonl(rows: list[dict[str, Any]], out: TextIO) -> None:
    for row in rows:
        out.write(json.dumps(row, ensure_ascii=False, default=_json_default))
        out.write("\n")


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=_json_default)
    return str(value)


def _write_table(rows: list[dict[str, Any]], out: TextIO) -> None:
    if not rows:
        out.write("No runs found.\n")
        return

    columns = list(rows[0].keys())
    widths = {
        column: max(
            len(column),
            *(len(_format_cell(row.get(column))) for row in rows),
        )
        for column in columns
    }

    out.write("  ".join(column.ljust(widths[column]) for column in columns))
    out.write("\n")
    out.write("  ".join("-" * widths[column] for column in columns))
    out.write("\n")

    for row in rows:
        out.write(
            "  ".join(
                _format_cell(row.get(column)).ljust(widths[column])
                for column in columns
            )
        )
        out.write("\n")


def _write_rows(rows: list[dict[str, Any]], fmt: str, out: TextIO) -> None:
    if fmt == "json":
        _write_json(rows, out)
        return

    if fmt == "jsonl":
        _write_jsonl(rows, out)
        return

    _write_table(rows, out)


def _record_summary(record: RunRecord) -> dict[str, Any]:
    return {
        "run_id": record.run_id,
        "status": record.status,
        "started_ts": record.started_ts,
        "duration_s": record.duration_s,
        "config_hash": record.config_hash,
        "run_dir": record.run_dir,
    }


def _record_with_metric(record: RunRecord, metric: str) -> dict[str, Any]:
    row = _record_summary(record)
    row[f"metric.{metric}"] = record.metric(metric)
    return row


def _run_list(args: argparse.Namespace, out: TextIO) -> int:
    registry = RunRegistry(args.root)
    records = registry.list(status=args.status, config_hash=args.config_hash)

    if args.limit is not None:
        records = records[:args.limit]

    rows = [_record_summary(record) for record in records]
    _write_rows(rows, args.format, out)
    return 0


def _run_latest(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    registry = RunRegistry(args.root)
    record = registry.latest(status=args.status, config_hash=args.config_hash)

    if record is None:
        err.write("No runs found.\n")
        return 1

    row = _record_summary(record)

    if args.format == "json":
        _write_json(row, out)
    else:
        _write_table([row], out)

    return 0


def _run_best(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    registry = RunRegistry(args.root)
    status = None if args.all_statuses else args.status

    record = registry.best(args.metric, mode=args.mode, status=status)

    if record is None:
        err.write("No run found for the requested metric.\n")
        return 1

    row = _record_with_metric(record, args.metric)

    if args.format == "json":
        _write_json(row, out)
    else:
        _write_table([row], out)

    return 0


def _run_compare(args: argparse.Namespace, out: TextIO) -> int:
    registry = RunRegistry(args.root)
    fields = tuple(args.fields) if args.fields else None

    rows = compare_runs(
        registry,
        args.metrics,
        status=args.status,
        config_hash=args.config_hash,
        fields=fields,
        default=args.default,
    )

    _write_rows(rows, args.format, out)
    return 0


def _add_common_run_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=".", help="Runs root directory")
    parser.add_argument("--status", default=None, help="Filter by run status")
    parser.add_argument("--config-hash", default=None, help="Filter by config hash")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="saltai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    runs = subparsers.add_parser("runs", help="Inspect local SaltAI runs")
    runs_subparsers = runs.add_subparsers(dest="runs_command", required=True)

    list_parser = runs_subparsers.add_parser("list", help="List runs")
    _add_common_run_filters(list_parser)
    list_parser.add_argument("--limit", type=int, default=None)
    list_parser.add_argument("--format", choices=("table", "json", "jsonl"), default="table")
    list_parser.set_defaults(func=_run_list)

    latest_parser = runs_subparsers.add_parser("latest", help="Show latest run")
    _add_common_run_filters(latest_parser)
    latest_parser.add_argument("--format", choices=("table", "json"), default="table")
    latest_parser.set_defaults(func=_run_latest)

    best_parser = runs_subparsers.add_parser("best", help="Show best run by metric")
    best_parser.add_argument("--root", default=".", help="Runs root directory")
    best_parser.add_argument("--metric", required=True, help="Metric path, e.g. val.accuracy")
    best_parser.add_argument("--mode", choices=("max", "min"), default="max")
    best_parser.add_argument("--status", default="success", help="Filter by run status")
    best_parser.add_argument("--all-statuses", action="store_true")
    best_parser.add_argument("--format", choices=("table", "json"), default="table")
    best_parser.set_defaults(func=_run_best)

    compare_parser = runs_subparsers.add_parser("compare", help="Compare runs by metrics")
    _add_common_run_filters(compare_parser)
    compare_parser.add_argument("--metrics", nargs="+", required=True)
    compare_parser.add_argument("--fields", nargs="*", default=None)
    compare_parser.add_argument("--default", default=None)
    compare_parser.add_argument("--format", choices=("table", "json", "jsonl"), default="table")
    compare_parser.set_defaults(func=_run_compare)

    return parser


def main(
        argv: Sequence[str] | None = None,
        *,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
) -> int:
    out = stdout or sys.stdout
    err = stderr or sys.stderr

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.func is _run_latest or args.func is _run_best:
        return args.func(args, out, err)

    return args.func(args, out)


if __name__ == "__main__":
    raise SystemExit(main())
