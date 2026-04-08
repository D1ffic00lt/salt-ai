import pandas as pd
import json
import os
import tempfile
import unittest

from saltai import RunRegistry


def _write_manifest(root, rel_dir, **overrides):
    run_dir = os.path.join(root, rel_dir)
    os.makedirs(run_dir, exist_ok=True)

    manifest = {
        "run_id": rel_dir.replace(os.sep, "_"),
        "status": "finished",
        "started_ts": 1.0,
        "finished_ts": 2.0,
        "config_hash": "cfg-default",
        "metrics": {},
        "inputs": {},
        "outputs": {},
        "error": None,
        "extra": {},
    }
    manifest.update(overrides)

    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    return manifest_path


class TestRunRegistry(unittest.TestCase):
    def test_list_returns_empty_if_root_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "missing")

            registry = RunRegistry(root)

            self.assertEqual(registry.list(), [])

    def test_list_discovers_manifest_json_recursively(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "runs/r1",
                run_id="r1",
                started_ts=10.0,
            )
            _write_manifest(
                d,
                "nested/runs/r2",
                run_id="r2",
                started_ts=20.0,
            )

            registry = RunRegistry(d)
            records = registry.list()

            self.assertEqual([record.run_id for record in records], ["r2", "r1"])
            self.assertTrue(records[0].manifest_path.endswith("manifest.json"))
            self.assertTrue(records[0].run_dir.endswith(os.path.join("nested", "runs", "r2")))

    def test_list_filters_by_status(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                status="finished",
            )
            _write_manifest(
                d,
                "r2",
                run_id="r2",
                status="failed",
            )

            registry = RunRegistry(d)
            records = registry.list(status="finished")

            self.assertEqual([record.run_id for record in records], ["r1"])

    def test_list_filters_by_config_hash(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                config_hash="cfg-a",
            )
            _write_manifest(
                d,
                "r2",
                run_id="r2",
                config_hash="cfg-b",
            )

            registry = RunRegistry(d)
            records = registry.list(config_hash="cfg-b")

            self.assertEqual([record.run_id for record in records], ["r2"])

    def test_get_returns_record_by_run_id(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
            )
            _write_manifest(
                d,
                "r2",
                run_id="r2",
                metrics={"val": {"accuracy": 0.91}},
            )

            registry = RunRegistry(d)
            record = registry.get("r2")

            self.assertIsNotNone(record)
            self.assertEqual(record.run_id, "r2")
            self.assertEqual(record.metric("val.accuracy"), 0.91)
            self.assertIsNone(registry.get("missing"))

    def test_record_duration_s(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                started_ts=10.0,
                finished_ts=25.5,
            )

            registry = RunRegistry(d)
            record = registry.get("r1")

            self.assertIsNotNone(record)
            self.assertEqual(record.duration_s, 15.5)

    def test_record_duration_s_returns_none_if_timestamp_missing(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                started_ts=10.0,
                finished_ts=None,
            )

            registry = RunRegistry(d)
            record = registry.get("r1")

            self.assertIsNotNone(record)
            self.assertIsNone(record.duration_s)

    def test_metric_returns_nested_value_by_dot_path(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                metrics={
                    "train": {"loss": 0.25},
                    "val": {"accuracy": 0.92},
                },
            )

            registry = RunRegistry(d)
            record = registry.get("r1")

            self.assertIsNotNone(record)
            self.assertEqual(record.metric("train.loss"), 0.25)
            self.assertEqual(record.metric("val.accuracy"), 0.92)
            self.assertEqual(record.metric("missing.metric", default="x"), "x")
            self.assertEqual(record.metric("", default="x"), "x")

    def test_best_max_by_nested_metric_path(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                metrics={"val": {"accuracy": 0.80}},
            )
            _write_manifest(
                d,
                "r2",
                run_id="r2",
                metrics={"val": {"accuracy": 0.95}},
            )
            _write_manifest(
                d,
                "r3",
                run_id="r3",
                metrics={"val": {"accuracy": 0.90}},
            )

            registry = RunRegistry(d)
            best = registry.best("val.accuracy", mode="max")

            self.assertIsNotNone(best)
            self.assertEqual(best.run_id, "r2")

    def test_best_min_by_nested_metric_path(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                metrics={"val": {"loss": 0.30}},
            )
            _write_manifest(
                d,
                "r2",
                run_id="r2",
                metrics={"val": {"loss": 0.10}},
            )
            _write_manifest(
                d,
                "r3",
                run_id="r3",
                metrics={"val": {"loss": 0.20}},
            )

            registry = RunRegistry(d)
            best = registry.best("val.loss", mode="min")

            self.assertIsNotNone(best)
            self.assertEqual(best.run_id, "r2")

    def test_best_filters_by_status(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                status="finished",
                metrics={"val": {"accuracy": 0.90}},
            )
            _write_manifest(
                d,
                "r2",
                run_id="r2",
                status="failed",
                metrics={"val": {"accuracy": 0.99}},
            )

            registry = RunRegistry(d)
            best = registry.best("val.accuracy", mode="max")

            self.assertIsNotNone(best)
            self.assertEqual(best.run_id, "r1")

    def test_best_can_use_success_status_for_runner_manifests(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                status="success",
                metrics={"val": {"accuracy": 0.90}},
            )
            _write_manifest(
                d,
                "r2",
                run_id="r2",
                status="success",
                metrics={"val": {"accuracy": 0.95}},
            )

            registry = RunRegistry(d)
            best = registry.best("val.accuracy", mode="max", status="success")

            self.assertIsNotNone(best)
            self.assertEqual(best.run_id, "r2")

    def test_best_ignores_missing_and_non_numeric_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                metrics={"val": {"accuracy": "0.99"}},
            )
            _write_manifest(
                d,
                "r2",
                run_id="r2",
                metrics={"train": {"loss": 0.1}},
            )
            _write_manifest(
                d,
                "r3",
                run_id="r3",
                metrics={"val": {"accuracy": True}},
            )
            _write_manifest(
                d,
                "r4",
                run_id="r4",
                metrics={"val": {"accuracy": 0.88}},
            )

            registry = RunRegistry(d)
            best = registry.best("val.accuracy", mode="max")

            self.assertIsNotNone(best)
            self.assertEqual(best.run_id, "r4")

    def test_best_returns_none_if_no_numeric_metric(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                metrics={"val": {"accuracy": "bad"}},
            )
            _write_manifest(
                d,
                "r2",
                run_id="r2",
                metrics={},
            )

            registry = RunRegistry(d)

            self.assertIsNone(registry.best("val.accuracy", mode="max"))

    def test_best_rejects_invalid_mode(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                metrics={"val": {"accuracy": 0.9}},
            )

            registry = RunRegistry(d)

            with self.assertRaises(ValueError):
                registry.best("val.accuracy", mode="median")

    def test_to_rows_flattens_metrics_with_metric_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                status="finished",
                started_ts=10.0,
                finished_ts=13.5,
                config_hash="cfg-a",
                metrics={
                    "train": {"loss": 0.25},
                    "val": {"accuracy": 0.92},
                },
                inputs={"config": "config.yaml"},
                outputs={"artifacts": []},
                extra={"note": "ok"},
            )

            registry = RunRegistry(d)
            rows = registry.to_rows()

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "r1")
            self.assertEqual(rows[0]["status"], "finished")
            self.assertEqual(rows[0]["config_hash"], "cfg-a")
            self.assertEqual(rows[0]["duration_s"], 3.5)
            self.assertEqual(rows[0]["inputs"], {"config": "config.yaml"})
            self.assertEqual(rows[0]["outputs"], {"artifacts": []})
            self.assertEqual(rows[0]["extra"], {"note": "ok"})
            self.assertEqual(rows[0]["metric.train.loss"], 0.25)
            self.assertEqual(rows[0]["metric.val.accuracy"], 0.92)

    def test_to_rows_applies_filters(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                status="finished",
                config_hash="cfg-a",
            )
            _write_manifest(
                d,
                "r2",
                run_id="r2",
                status="failed",
                config_hash="cfg-a",
            )
            _write_manifest(
                d,
                "r3",
                run_id="r3",
                status="finished",
                config_hash="cfg-b",
            )

            registry = RunRegistry(d)
            rows = registry.to_rows(status="finished", config_hash="cfg-a")

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "r1")

    def test_broken_manifest_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "valid",
                run_id="valid",
            )

            broken_dir = os.path.join(d, "broken")
            os.makedirs(broken_dir, exist_ok=True)
            with open(os.path.join(broken_dir, "manifest.json"), "w", encoding="utf-8") as f:
                f.write("{bad json")

            registry = RunRegistry(d)
            records = registry.list()

            self.assertEqual([record.run_id for record in records], ["valid"])

    def test_manifest_without_run_id_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "valid",
                run_id="valid",
            )
            _write_manifest(
                d,
                "missing_run_id",
                run_id=None,
            )

            registry = RunRegistry(d)
            records = registry.list()

            self.assertEqual([record.run_id for record in records], ["valid"])

    def test_unknown_started_ts_goes_last(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                started_ts=None,
            )
            _write_manifest(
                d,
                "r2",
                run_id="r2",
                started_ts=20.0,
            )
            _write_manifest(
                d,
                "r3",
                run_id="r3",
                started_ts=10.0,
            )

            registry = RunRegistry(d)
            records = registry.list()

            self.assertEqual([record.run_id for record in records], ["r2", "r3", "r1"])

    def test_latest_returns_most_recent_run(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                started_ts=10.0,
            )
            _write_manifest(
                d,
                "r2",
                run_id="r2",
                started_ts=30.0,
            )
            _write_manifest(
                d,
                "r3",
                run_id="r3",
                started_ts=20.0,
            )

            registry = RunRegistry(d)
            latest = registry.latest()

            self.assertIsNotNone(latest)
            self.assertEqual(latest.run_id, "r2")

    def test_latest_applies_filters(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                status="failed",
                config_hash="cfg-a",
                started_ts=40.0,
            )
            _write_manifest(
                d,
                "r2",
                run_id="r2",
                status="success",
                config_hash="cfg-a",
                started_ts=30.0,
            )
            _write_manifest(
                d,
                "r3",
                run_id="r3",
                status="success",
                config_hash="cfg-b",
                started_ts=20.0,
            )
            _write_manifest(
                d,
                "r4",
                run_id="r4",
                status="success",
                config_hash="cfg-a",
                started_ts=10.0,
            )

            registry = RunRegistry(d)
            latest = registry.latest(status="success", config_hash="cfg-a")

            self.assertIsNotNone(latest)
            self.assertEqual(latest.run_id, "r2")

    def test_latest_returns_none_if_no_runs_match(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                status="failed",
                started_ts=10.0,
            )

            registry = RunRegistry(d)

            self.assertIsNone(registry.latest(status="success"))

    def test_latest_returns_none_if_root_missing(self):
        with tempfile.TemporaryDirectory() as d:
            registry = RunRegistry(os.path.join(d, "missing"))

            self.assertIsNone(registry.latest())

    def test_count_returns_number_of_runs(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(d, "r1", run_id="r1")
            _write_manifest(d, "r2", run_id="r2")
            _write_manifest(d, "r3", run_id="r3")

            registry = RunRegistry(d)

            self.assertEqual(registry.count(), 3)

    def test_count_applies_filters(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                status="success",
                config_hash="cfg-a",
            )
            _write_manifest(
                d,
                "r2",
                run_id="r2",
                status="failed",
                config_hash="cfg-a",
            )
            _write_manifest(
                d,
                "r3",
                run_id="r3",
                status="success",
                config_hash="cfg-b",
            )

            registry = RunRegistry(d)

            self.assertEqual(registry.count(status="success"), 2)
            self.assertEqual(registry.count(config_hash="cfg-a"), 2)
            self.assertEqual(registry.count(status="success", config_hash="cfg-a"), 1)
            self.assertEqual(registry.count(status="missing"), 0)

    def test_count_returns_zero_if_root_missing(self):
        with tempfile.TemporaryDirectory() as d:
            registry = RunRegistry(os.path.join(d, "missing"))

            self.assertEqual(registry.count(), 0)

    def test_status_counts_returns_counts_by_status(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                status="success",
            )
            _write_manifest(
                d,
                "r2",
                run_id="r2",
                status="failed",
            )
            _write_manifest(
                d,
                "r3",
                run_id="r3",
                status="success",
            )

            registry = RunRegistry(d)

            self.assertEqual(registry.status_counts(), {
                "success": 2,
                "failed": 1,
            })

    def test_status_counts_applies_config_hash_filter(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                status="success",
                config_hash="cfg-a",
            )
            _write_manifest(
                d,
                "r2",
                run_id="r2",
                status="failed",
                config_hash="cfg-a",
            )
            _write_manifest(
                d,
                "r3",
                run_id="r3",
                status="success",
                config_hash="cfg-b",
            )

            registry = RunRegistry(d)

            self.assertEqual(registry.status_counts(config_hash="cfg-a"), {
                "success": 1,
                "failed": 1,
            })

    def test_status_counts_returns_empty_dict_if_root_missing(self):
        with tempfile.TemporaryDirectory() as d:
            registry = RunRegistry(os.path.join(d, "missing"))

            self.assertEqual(registry.status_counts(), {})

    def test_metric_paths_returns_sorted_unique_metric_paths(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                metrics={
                    "train": {"loss": 0.25},
                    "val": {"accuracy": 0.91},
                },
            )
            _write_manifest(
                d,
                "r2",
                run_id="r2",
                metrics={
                    "train": {"loss": 0.20},
                    "val": {"loss": 0.30},
                    "score": 1.0,
                },
            )

            registry = RunRegistry(d)

            self.assertEqual(registry.metric_paths(), [
                "score",
                "train.loss",
                "val.accuracy",
                "val.loss",
            ])

    def test_metric_paths_applies_filters(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                status="success",
                config_hash="cfg-a",
                metrics={"val": {"accuracy": 0.91}},
            )
            _write_manifest(
                d,
                "r2",
                run_id="r2",
                status="failed",
                config_hash="cfg-a",
                metrics={"error_score": 1.0},
            )
            _write_manifest(
                d,
                "r3",
                run_id="r3",
                status="success",
                config_hash="cfg-b",
                metrics={"train": {"loss": 0.25}},
            )

            registry = RunRegistry(d)

            self.assertEqual(
                registry.metric_paths(status="success", config_hash="cfg-a"),
                ["val.accuracy"],
            )

    def test_metric_paths_returns_empty_list_if_root_missing(self):
        with tempfile.TemporaryDirectory() as d:
            registry = RunRegistry(os.path.join(d, "missing"))

            self.assertEqual(registry.metric_paths(), [])

    def test_record_artifacts_returns_manifest_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            model_ref = {
                "id": "a1",
                "kind": "model",
                "name": "weights",
                "uri": "file:///tmp/weights.bin",
                "sha256": "abc",
                "size_bytes": 123,
                "meta": {"format": "bin"},
            }
            log_ref = {
                "id": "a2",
                "kind": "log",
                "name": "events",
                "uri": "file:///tmp/events.jsonl",
                "sha256": "def",
                "size_bytes": 456,
                "meta": {},
            }

            _write_manifest(
                d,
                "r1",
                run_id="r1",
                outputs={
                    "artifacts": [model_ref, log_ref],
                    "checkpoints": {},
                },
            )

            record = RunRegistry(d).get("r1")

            self.assertIsNotNone(record)
            self.assertEqual(record.artifacts(), [model_ref, log_ref])
            self.assertEqual(record.artifacts(kind="model"), [model_ref])
            self.assertEqual(record.artifacts(kind="missing"), [])

    def test_record_artifact_finds_by_name_and_kind(self):
        with tempfile.TemporaryDirectory() as d:
            model_ref = {
                "id": "a1",
                "kind": "model",
                "name": "weights",
                "uri": "file:///tmp/weights.bin",
                "sha256": "abc",
                "size_bytes": 123,
                "meta": {},
            }
            log_ref = {
                "id": "a2",
                "kind": "log",
                "name": "weights",
                "uri": "file:///tmp/weights.log",
                "sha256": "def",
                "size_bytes": 456,
                "meta": {},
            }

            _write_manifest(
                d,
                "r1",
                run_id="r1",
                outputs={
                    "artifacts": [model_ref, log_ref],
                    "checkpoints": {},
                },
            )

            record = RunRegistry(d).get("r1")

            self.assertIsNotNone(record)
            self.assertEqual(record.artifact("weights", kind="model"), model_ref)
            self.assertEqual(record.artifact("weights", kind="log"), log_ref)
            self.assertEqual(record.artifact("missing", default="x"), "x")

    def test_record_checkpoint_accessors_return_manifest_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            latest_ref = {
                "id": "c1",
                "kind": "ckpt",
                "name": "latest_step_1",
                "uri": "file:///tmp/latest.ckpt.json",
                "sha256": "abc",
                "size_bytes": 123,
                "meta": {"step": 1, "tag": "latest"},
            }
            best_ref = {
                "id": "c2",
                "kind": "ckpt",
                "name": "best_step_2",
                "uri": "file:///tmp/best.ckpt.json",
                "sha256": "def",
                "size_bytes": 456,
                "meta": {"step": 2, "tag": "best", "metric": 0.95},
            }
            resume_ref = {
                "id": "c3",
                "kind": "ckpt",
                "name": "latest_step_0",
                "uri": "file:///tmp/resume.ckpt.json",
                "sha256": "ghi",
                "size_bytes": 111,
                "meta": {"step": 0, "tag": "latest"},
            }

            _write_manifest(
                d,
                "r1",
                run_id="r1",
                outputs={
                    "artifacts": [],
                    "checkpoints": {
                        "latest": latest_ref,
                        "best": best_ref,
                        "resume_from": resume_ref,
                    },
                },
            )

            record = RunRegistry(d).get("r1")

            self.assertIsNotNone(record)
            self.assertEqual(record.checkpoint("latest"), latest_ref)
            self.assertEqual(record.checkpoint("best"), best_ref)
            self.assertEqual(record.checkpoint("resume_from"), resume_ref)
            self.assertEqual(record.latest_checkpoint, latest_ref)
            self.assertEqual(record.best_checkpoint, best_ref)
            self.assertEqual(record.resume_checkpoint, resume_ref)
            self.assertEqual(record.checkpoint("missing", default="x"), "x")

    def test_record_artifact_accessors_ignore_malformed_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                outputs={
                    "artifacts": "bad",
                    "checkpoints": "bad",
                },
            )

            record = RunRegistry(d).get("r1")

            self.assertIsNotNone(record)
            self.assertEqual(record.artifacts(), [])
            self.assertEqual(record.artifact("missing", default="x"), "x")
            self.assertEqual(record.checkpoint("latest", default="x"), "x")
            self.assertIsNone(record.latest_checkpoint)
            self.assertIsNone(record.best_checkpoint)
            self.assertIsNone(record.resume_checkpoint)

    def test_record_artifact_accessors_skip_malformed_artifact_items(self):
        with tempfile.TemporaryDirectory() as d:
            ref = {
                "id": "a1",
                "kind": "model",
                "name": "weights",
                "uri": "file:///tmp/weights.bin",
                "sha256": "abc",
                "size_bytes": 123,
                "meta": {},
            }

            _write_manifest(
                d,
                "r1",
                run_id="r1",
                outputs={
                    "artifacts": [ref, "bad", None, 123],
                    "checkpoints": {},
                },
            )

            record = RunRegistry(d).get("r1")

            self.assertIsNotNone(record)
            self.assertEqual(record.artifacts(), [ref])

    def test_to_dataframe_returns_pandas_dataframe(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                status="success",
                metrics={"val": {"accuracy": 0.91}},
                inputs={"config": "config.yaml"},
            )

            df = RunRegistry(d).to_dataframe()

            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["run_id"], "r1")
            self.assertEqual(df.iloc[0]["status"], "success")
            self.assertEqual(df.iloc[0]["metric.val.accuracy"], 0.91)
            self.assertEqual(df.iloc[0]["inputs"], {"config": "config.yaml"})

    def test_to_dataframe_applies_filters(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                status="success",
                config_hash="cfg-a",
                metrics={"val": {"accuracy": 0.91}},
            )
            _write_manifest(
                d,
                "r2",
                run_id="r2",
                status="failed",
                config_hash="cfg-a",
                metrics={"val": {"accuracy": 0.50}},
            )
            _write_manifest(
                d,
                "r3",
                run_id="r3",
                status="success",
                config_hash="cfg-b",
                metrics={"val": {"accuracy": 0.80}},
            )

            df = RunRegistry(d).to_dataframe(status="success", config_hash="cfg-a")

            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["run_id"], "r1")
            self.assertEqual(df.iloc[0]["metric.val.accuracy"], 0.91)

    def test_to_csv_writes_dataframe_export(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                status="success",
                metrics={"val": {"accuracy": 0.91}},
            )

            out_path = os.path.join(d, "runs.csv")

            RunRegistry(d).to_csv(out_path)

            df = pd.read_csv(out_path)

            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["run_id"], "r1")
            self.assertEqual(df.iloc[0]["status"], "success")
            self.assertEqual(df.iloc[0]["metric.val.accuracy"], 0.91)

    def test_to_jsonl_writes_dataframe_export(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                status="success",
                metrics={"val": {"accuracy": 0.91}},
                inputs={"config": "config.yaml"},
            )

            out_path = os.path.join(d, "runs.jsonl")

            RunRegistry(d).to_jsonl(out_path)

            with open(out_path, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "r1")
            self.assertEqual(rows[0]["status"], "success")
            self.assertEqual(rows[0]["metric.val.accuracy"], 0.91)
            self.assertEqual(rows[0]["inputs"], {"config": "config.yaml"})


if __name__ == "__main__":
    unittest.main()
