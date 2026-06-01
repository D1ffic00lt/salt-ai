import json
import os
import pathlib
import tempfile
import unittest

from saltai.runs import RunRegistry


def _write_manifest(root, run_id, manifest):
    run_dir = os.path.join(root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    path = os.path.join(run_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    return path


class TestRunRegistryHardening(unittest.TestCase):
    def test_accepts_pathlike_root(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)

            _write_manifest(
                root,
                "r1",
                {
                    "run_id": "r1",
                    "status": "success",
                    "started_ts": 10.0,
                    "finished_ts": 15.0,
                    "metrics": {"val": {"accuracy": 0.91}},
                },
            )

            records = RunRegistry(root).list()

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].run_id, "r1")
            self.assertEqual(records[0].metric("val.accuracy"), 0.91)

    def test_malformed_optional_dict_fields_become_empty_dicts(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                {
                    "run_id": "r1",
                    "status": "success",
                    "metrics": None,
                    "inputs": [],
                    "outputs": "bad",
                    "extra": 123,
                },
            )

            record = RunRegistry(d).get("r1")

            self.assertIsNotNone(record)
            self.assertEqual(record.metrics, {})
            self.assertEqual(record.inputs, {})
            self.assertEqual(record.outputs, {})
            self.assertEqual(record.extra, {})
            self.assertEqual(record.metric("val.accuracy", default="missing"), "missing")

    def test_non_dict_error_becomes_none(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                {
                    "run_id": "r1",
                    "status": "failed",
                    "error": "boom",
                },
            )

            record = RunRegistry(d).get("r1")

            self.assertIsNotNone(record)
            self.assertIsNone(record.error)

    def test_malformed_timestamps_do_not_break_duration_or_sorting(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "bad_ts",
                {
                    "run_id": "bad_ts",
                    "status": "success",
                    "started_ts": "not-a-date",
                    "finished_ts": "also-bad",
                },
            )
            _write_manifest(
                d,
                "good_ts",
                {
                    "run_id": "good_ts",
                    "status": "success",
                    "started_ts": 20.0,
                    "finished_ts": 25.0,
                },
            )

            records = RunRegistry(d).list()

            self.assertEqual([r.run_id for r in records], ["good_ts", "bad_ts"])
            self.assertIsNone(records[1].started_ts)
            self.assertIsNone(records[1].finished_ts)
            self.assertIsNone(records[1].duration_s)

    def test_best_ignores_string_and_bool_metric_values(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "numeric",
                {
                    "run_id": "numeric",
                    "status": "success",
                    "metrics": {"val": {"accuracy": 0.9}},
                },
            )
            _write_manifest(
                d,
                "string",
                {
                    "run_id": "string",
                    "status": "success",
                    "metrics": {"val": {"accuracy": "0.99"}},
                },
            )
            _write_manifest(
                d,
                "bool",
                {
                    "run_id": "bool",
                    "status": "success",
                    "metrics": {"val": {"accuracy": True}},
                },
            )

            best = RunRegistry(d).best("val.accuracy", status="success")

            self.assertIsNotNone(best)
            self.assertEqual(best.run_id, "numeric")

    def test_to_rows_handles_malformed_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                {
                    "run_id": "r1",
                    "status": "success",
                    "metrics": ["bad"],
                },
            )

            rows = RunRegistry(d).to_rows(status="success")

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "r1")
            self.assertFalse(any(key.startswith("metric.") for key in rows[0]))

    def test_manifest_root_value_must_be_dict(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = os.path.join(d, "r1")
            os.makedirs(run_dir, exist_ok=True)

            with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(["bad"], f)

            self.assertEqual(RunRegistry(d).list(), [])


if __name__ == "__main__":
    unittest.main()
