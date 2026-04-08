import os
import tempfile
import unittest

from saltai import RunRegistry, Runner


class TestRunRegistryWithRunner(unittest.TestCase):
    def test_registry_reads_runner_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            runner = Runner()

            result = runner.run(
                {
                    "run": {"id": "registry_runner_run"},
                    "seed": 42,
                    "paths": {"root": d},
                },
                body=lambda _ctx: {
                    "train": {"loss": 0.25},
                    "val": {"accuracy": 0.91},
                },
            )

            registry = RunRegistry(d)
            record = registry.get("registry_runner_run")

            self.assertIsNotNone(record)
            self.assertEqual(record.run_id, "registry_runner_run")
            self.assertEqual(record.status, "success")
            self.assertEqual(record.config_hash, result.context["config_hash"])
            self.assertEqual(record.manifest_path, result.manifest_path)
            self.assertEqual(record.run_dir, result.context["run_dir"])
            self.assertTrue(os.path.exists(record.manifest_path))
            self.assertEqual(record.metric("train.loss"), 0.25)
            self.assertEqual(record.metric("val.accuracy"), 0.91)
            self.assertIsNone(record.error)

    def test_registry_best_works_with_runner_success_status(self):
        with tempfile.TemporaryDirectory() as d:
            runner = Runner()

            runner.run(
                {
                    "run": {"id": "run_low"},
                    "seed": 42,
                    "paths": {"root": d},
                },
                body=lambda _ctx: {"val": {"accuracy": 0.80}},
            )

            runner.run(
                {
                    "run": {"id": "run_high"},
                    "seed": 42,
                    "paths": {"root": d},
                },
                body=lambda _ctx: {"val": {"accuracy": 0.95}},
            )

            registry = RunRegistry(d)

            self.assertIsNone(registry.best("val.accuracy"))

            best = registry.best("val.accuracy", mode="max", status="success")

            self.assertIsNotNone(best)
            self.assertEqual(best.run_id, "run_high")
            self.assertEqual(best.metric("val.accuracy"), 0.95)

    def test_registry_to_rows_reads_runner_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            runner = Runner()

            runner.run(
                {
                    "run": {"id": "rows_run"},
                    "seed": 42,
                    "paths": {"root": d},
                },
                body=lambda _ctx: {
                    "train": {"loss": 0.10},
                    "val": {"accuracy": 0.93},
                },
            )

            registry = RunRegistry(d)
            rows = registry.to_rows(status="success")

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "rows_run")
            self.assertEqual(rows[0]["status"], "success")
            self.assertEqual(rows[0]["metric.train.loss"], 0.10)
            self.assertEqual(rows[0]["metric.val.accuracy"], 0.93)
            self.assertIsNotNone(rows[0]["duration_s"])

    def test_registry_list_filters_runner_runs_by_config_hash(self):
        with tempfile.TemporaryDirectory() as d:
            runner = Runner()

            result_a = runner.run(
                {
                    "run": {"id": "cfg_a"},
                    "seed": 42,
                    "paths": {"root": d},
                    "params": {"lr": 0.01},
                },
                body=lambda _ctx: {"val": {"accuracy": 0.90}},
            )

            runner.run(
                {
                    "run": {"id": "cfg_b"},
                    "seed": 42,
                    "paths": {"root": d},
                    "params": {"lr": 0.02},
                },
                body=lambda _ctx: {"val": {"accuracy": 0.91}},
            )

            registry = RunRegistry(d)
            records = registry.list(config_hash=result_a.context["config_hash"])

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].run_id, "cfg_a")

    def test_registry_reads_failed_runner_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            runner = Runner()

            def body(_ctx):
                raise RuntimeError("boom")

            result = runner.run(
                {
                    "run": {"id": "failed_run"},
                    "seed": 42,
                    "paths": {"root": d},
                },
                body=body,
            )

            registry = RunRegistry(d)
            record = registry.get("failed_run")

            self.assertIsNotNone(record)
            self.assertEqual(result.status, "failed")
            self.assertEqual(record.status, "failed")
            self.assertEqual(record.metrics, {})
            self.assertIsNotNone(record.error)


if __name__ == "__main__":
    unittest.main()
