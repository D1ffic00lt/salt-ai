import io
import json
import os
import tempfile
import unittest

from saltai.cli.main import main


def _write_manifest(root, rel_dir, **overrides):
    run_dir = os.path.join(root, rel_dir)
    os.makedirs(run_dir, exist_ok=True)

    manifest = {
        "run_id": rel_dir.replace(os.sep, "_"),
        "status": "success",
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

    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    return run_dir


class TestRunsCLI(unittest.TestCase):
    def test_runs_list_outputs_table(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                started_ts=10.0,
                metrics={"val": {"accuracy": 0.8}},
            )
            _write_manifest(
                d,
                "r2",
                run_id="r2",
                started_ts=20.0,
                metrics={"val": {"accuracy": 0.9}},
            )

            stdout = io.StringIO()
            stderr = io.StringIO()

            code = main(
                ["runs", "list", "--root", d],
                stdout=stdout,
                stderr=stderr,
            )

            self.assertEqual(code, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertIn("run_id", stdout.getvalue())
            self.assertIn("r2", stdout.getvalue())
            self.assertIn("r1", stdout.getvalue())

    def test_runs_list_outputs_json(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                status="success",
                started_ts=10.0,
            )

            stdout = io.StringIO()
            stderr = io.StringIO()

            code = main(
                ["runs", "list", "--root", d, "--format", "json"],
                stdout=stdout,
                stderr=stderr,
            )

            self.assertEqual(code, 0)
            rows = json.loads(stdout.getvalue())

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "r1")
            self.assertEqual(rows[0]["status"], "success")

    def test_runs_list_applies_status_filter(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(d, "r1", run_id="r1", status="success")
            _write_manifest(d, "r2", run_id="r2", status="failed")

            stdout = io.StringIO()
            stderr = io.StringIO()

            code = main(
                ["runs", "list", "--root", d, "--status", "failed", "--format", "json"],
                stdout=stdout,
                stderr=stderr,
            )

            self.assertEqual(code, 0)
            rows = json.loads(stdout.getvalue())

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "r2")

    def test_runs_latest_outputs_latest_run(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(d, "r1", run_id="r1", started_ts=10.0)
            _write_manifest(d, "r2", run_id="r2", started_ts=30.0)
            _write_manifest(d, "r3", run_id="r3", started_ts=20.0)

            stdout = io.StringIO()
            stderr = io.StringIO()

            code = main(
                ["runs", "latest", "--root", d, "--format", "json"],
                stdout=stdout,
                stderr=stderr,
            )

            self.assertEqual(code, 0)
            row = json.loads(stdout.getvalue())

            self.assertEqual(row["run_id"], "r2")

    def test_runs_latest_returns_error_if_no_runs(self):
        with tempfile.TemporaryDirectory() as d:
            stdout = io.StringIO()
            stderr = io.StringIO()

            code = main(
                ["runs", "latest", "--root", d],
                stdout=stdout,
                stderr=stderr,
            )

            self.assertEqual(code, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("No runs found", stderr.getvalue())

    def test_runs_best_uses_success_status_by_default(self):
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
            _write_manifest(
                d,
                "r3",
                run_id="r3",
                status="failed",
                metrics={"val": {"accuracy": 0.99}},
            )

            stdout = io.StringIO()
            stderr = io.StringIO()

            code = main(
                [
                    "runs",
                    "best",
                    "--root",
                    d,
                    "--metric",
                    "val.accuracy",
                    "--mode",
                    "max",
                    "--format",
                    "json",
                ],
                stdout=stdout,
                stderr=stderr,
            )

            self.assertEqual(code, 0)
            row = json.loads(stdout.getvalue())

            self.assertEqual(row["run_id"], "r2")
            self.assertEqual(row["metric.val.accuracy"], 0.95)

    def test_runs_best_can_include_all_statuses(self):
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
                status="failed",
                metrics={"val": {"accuracy": 0.99}},
            )

            stdout = io.StringIO()
            stderr = io.StringIO()

            code = main(
                [
                    "runs",
                    "best",
                    "--root",
                    d,
                    "--metric",
                    "val.accuracy",
                    "--all-statuses",
                    "--format",
                    "json",
                ],
                stdout=stdout,
                stderr=stderr,
            )

            self.assertEqual(code, 0)
            row = json.loads(stdout.getvalue())

            self.assertEqual(row["run_id"], "r2")

    def test_runs_best_min_mode(self):
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

            stdout = io.StringIO()
            stderr = io.StringIO()

            code = main(
                [
                    "runs",
                    "best",
                    "--root",
                    d,
                    "--metric",
                    "val.loss",
                    "--mode",
                    "min",
                    "--format",
                    "json",
                ],
                stdout=stdout,
                stderr=stderr,
            )

            self.assertEqual(code, 0)
            row = json.loads(stdout.getvalue())

            self.assertEqual(row["run_id"], "r2")
            self.assertEqual(row["metric.val.loss"], 0.10)

    def test_runs_best_returns_error_if_metric_missing(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                metrics={"val": {"accuracy": 0.90}},
            )

            stdout = io.StringIO()
            stderr = io.StringIO()

            code = main(
                [
                    "runs",
                    "best",
                    "--root",
                    d,
                    "--metric",
                    "val.loss",
                ],
                stdout=stdout,
                stderr=stderr,
            )

            self.assertEqual(code, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("No run found", stderr.getvalue())

    def test_runs_compare_outputs_selected_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            _write_manifest(
                d,
                "r1",
                run_id="r1",
                started_ts=10.0,
                metrics={
                    "train": {"loss": 0.30},
                    "val": {"accuracy": 0.90},
                },
            )
            _write_manifest(
                d,
                "r2",
                run_id="r2",
                started_ts=20.0,
                metrics={
                    "train": {"loss": 0.20},
                    "val": {"accuracy": 0.95},
                },
            )

            stdout = io.StringIO()
            stderr = io.StringIO()

            code = main(
                [
                    "runs",
                    "compare",
                    "--root",
                    d,
                    "--metrics",
                    "val.accuracy",
                    "train.loss",
                    "--format",
                    "json",
                ],
                stdout=stdout,
                stderr=stderr,
            )

            self.assertEqual(code, 0)
            rows = json.loads(stdout.getvalue())

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["run_id"], "r2")
            self.assertEqual(rows[0]["metric.val.accuracy"], 0.95)
            self.assertEqual(rows[0]["metric.train.loss"], 0.20)
            self.assertEqual(rows[1]["run_id"], "r1")
            self.assertEqual(rows[1]["metric.val.accuracy"], 0.90)
            self.assertEqual(rows[1]["metric.train.loss"], 0.30)

    def test_runs_compare_applies_status_filter(self):
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
                status="failed",
                metrics={"val": {"accuracy": 0.95}},
            )

            stdout = io.StringIO()
            stderr = io.StringIO()

            code = main(
                [
                    "runs",
                    "compare",
                    "--root",
                    d,
                    "--status",
                    "failed",
                    "--metrics",
                    "val.accuracy",
                    "--format",
                    "json",
                ],
                stdout=stdout,
                stderr=stderr,
            )

            self.assertEqual(code, 0)
            rows = json.loads(stdout.getvalue())

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "r2")
            self.assertEqual(rows[0]["metric.val.accuracy"], 0.95)


if __name__ == "__main__":
    unittest.main()
