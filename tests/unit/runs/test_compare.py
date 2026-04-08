import json
import os
import tempfile
import unittest

from saltai.runs import RunRegistry, compare_records, compare_runs


def _write_manifest(root, run_id, manifest):
    run_dir = os.path.join(root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    path = os.path.join(run_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    return path


class TestRunCompare(unittest.TestCase):
    def test_compare_runs_returns_default_fields_and_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                {
                    "run_id": "r1",
                    "status": "success",
                    "started_ts": 10.0,
                    "finished_ts": 15.0,
                    "config_hash": "cfg-1",
                    "metrics": {"val": {"accuracy": 0.8, "loss": 0.4}},
                },
            )
            _write_manifest(
                d,
                "r2",
                {
                    "run_id": "r2",
                    "status": "success",
                    "started_ts": 20.0,
                    "finished_ts": 28.0,
                    "config_hash": "cfg-2",
                    "metrics": {"val": {"accuracy": 0.9, "loss": 0.3}},
                },
            )

            rows = compare_runs(
                RunRegistry(d),
                ["val.accuracy", "val.loss"],
                status="success",
            )

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["run_id"], "r2")
            self.assertEqual(rows[0]["status"], "success")
            self.assertEqual(rows[0]["started_ts"], 20.0)
            self.assertEqual(rows[0]["finished_ts"], 28.0)
            self.assertEqual(rows[0]["duration_s"], 8.0)
            self.assertEqual(rows[0]["config_hash"], "cfg-2")
            self.assertEqual(rows[0]["metric.val.accuracy"], 0.9)
            self.assertEqual(rows[0]["metric.val.loss"], 0.3)

    def test_compare_runs_accepts_single_metric_string(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                {
                    "run_id": "r1",
                    "status": "success",
                    "metrics": {"val": {"accuracy": 0.8}},
                },
            )

            rows = compare_runs(RunRegistry(d), "val.accuracy", status="success")

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["metric.val.accuracy"], 0.8)

    def test_compare_runs_respects_filters(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                {
                    "run_id": "r1",
                    "status": "success",
                    "config_hash": "cfg-1",
                    "metrics": {"score": 1.0},
                },
            )
            _write_manifest(
                d,
                "r2",
                {
                    "run_id": "r2",
                    "status": "failed",
                    "config_hash": "cfg-1",
                    "metrics": {"score": 2.0},
                },
            )
            _write_manifest(
                d,
                "r3",
                {
                    "run_id": "r3",
                    "status": "success",
                    "config_hash": "cfg-2",
                    "metrics": {"score": 3.0},
                },
            )

            rows = compare_runs(
                RunRegistry(d),
                "score",
                status="success",
                config_hash="cfg-1",
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "r1")
            self.assertEqual(rows[0]["metric.score"], 1.0)

    def test_compare_records_uses_default_for_missing_metric(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                {
                    "run_id": "r1",
                    "status": "success",
                    "metrics": {"train": {"loss": 0.5}},
                },
            )

            record = RunRegistry(d).get("r1")
            rows = compare_records([record], "val.accuracy", default="-")

            self.assertEqual(rows[0]["metric.val.accuracy"], "-")

    def test_compare_records_supports_custom_fields(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                {
                    "run_id": "r1",
                    "status": "success",
                    "started_ts": 10.0,
                    "finished_ts": 13.5,
                    "metrics": {"score": 0.7},
                },
            )

            record = RunRegistry(d).get("r1")
            rows = compare_records(
                [record],
                "score",
                fields=("run_id", "duration_s"),
            )

            self.assertEqual(rows, [
                {
                    "run_id": "r1",
                    "duration_s": 3.5,
                    "metric.score": 0.7,
                }
            ])

    def test_compare_records_unknown_field_uses_default(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                {
                    "run_id": "r1",
                    "status": "success",
                    "metrics": {"score": 0.7},
                },
            )

            record = RunRegistry(d).get("r1")
            rows = compare_records(
                [record],
                "score",
                fields=("run_id", "unknown"),
                default=None,
            )

            self.assertEqual(rows[0]["run_id"], "r1")
            self.assertIsNone(rows[0]["unknown"])
            self.assertEqual(rows[0]["metric.score"], 0.7)


if __name__ == "__main__":
    unittest.main()
