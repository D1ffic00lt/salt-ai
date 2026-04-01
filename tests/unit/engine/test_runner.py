import json
import os
import tempfile
import unittest

from saltai.engine.runner.runner import Runner
from saltai.engine.event_bus.bus import EventBus
from saltai.utils.typing.core import MetricSummary


class _Sink(object):
    def __init__(self):
        self.events = []

    def log(self, event):
        self.events.append(event)

    def flush(self):
        return None

    def close(self):
        return None


class TestRunner(unittest.TestCase):
    def test_runner_success_writes_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            sink = _Sink()
            bus = EventBus([sink])

            r = Runner(event_bus=bus)
            res = r.run(
                {"run": {"id": "r1"}, "seed": 42, "paths": {"root": d}},
                body=lambda ctx: None,
            )

            self.assertEqual(res.status, "success")
            self.assertTrue(os.path.exists(res.manifest_path))

            with open(res.manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)

            self.assertEqual(m["run_id"], "r1")
            self.assertEqual(m["status"], "success")
            self.assertIsNone(m["error"])
            self.assertEqual(m["metrics"], {})

            self.assertTrue(any(type(e).__name__ == "RunStarted" for e in sink.events))
            self.assertTrue(any(type(e).__name__ == "RunFinished" for e in sink.events))

    def test_runner_stores_body_dict_metrics_in_manifest_and_result(self):
        with tempfile.TemporaryDirectory() as d:
            r = Runner()

            def body(_ctx):
                return {
                    "train": {"loss": 0.25},
                    "val": {"accuracy": 0.9},
                }

            res = r.run(
                {"run": {"id": "r_metrics_dict"}, "seed": 42, "paths": {"root": d}},
                body=body,
            )

            self.assertEqual(res.status, "success")
            self.assertEqual(res.metrics.values["train"]["loss"], 0.25)
            self.assertEqual(res.metrics.values["val"]["accuracy"], 0.9)

            with open(res.manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)

            self.assertEqual(m["metrics"]["train"]["loss"], 0.25)
            self.assertEqual(m["metrics"]["val"]["accuracy"], 0.9)

    def test_runner_stores_body_metric_summary_in_manifest_and_result(self):
        with tempfile.TemporaryDirectory() as d:
            r = Runner()

            def body(_ctx):
                return MetricSummary(values={"loss": 0.1}, extra={"source": "body"})

            res = r.run(
                {"run": {"id": "r_metrics_summary"}, "seed": 42, "paths": {"root": d}},
                body=body,
            )

            self.assertEqual(res.status, "success")
            self.assertEqual(res.metrics.values["loss"], 0.1)
            self.assertEqual(res.metrics.extra["source"], "body")

            with open(res.manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)

            self.assertEqual(m["metrics"]["loss"], 0.1)

    def test_runio_log_metric_publishes_metric_event(self):
        with tempfile.TemporaryDirectory() as d:
            sink = _Sink()
            bus = EventBus([sink])

            r = Runner(event_bus=bus)

            def body(ctx):
                ctx.io.log_metric(
                    "accuracy",
                    0.91,
                    step=7,
                    epoch=1,
                    split="val",
                    extra={"source": "body"},
                )

            res = r.run(
                {"run": {"id": "r_log_metric"}, "seed": 42, "paths": {"root": d}},
                body=body,
            )

            self.assertEqual(res.status, "success")

            metric_events = [e for e in sink.events if type(e).__name__ == "MetricLogged"]
            self.assertEqual(len(metric_events), 1)
            self.assertEqual(metric_events[0].point.name, "accuracy")
            self.assertEqual(metric_events[0].point.value, 0.91)
            self.assertEqual(metric_events[0].point.step, 7)
            self.assertEqual(metric_events[0].point.epoch, 1)
            self.assertEqual(metric_events[0].point.split, "val")
            self.assertEqual(metric_events[0].point.extra["source"], "body")

    def test_runio_save_artifact_stores_artifact_and_publishes_event(self):
        with tempfile.TemporaryDirectory() as d:
            sink = _Sink()
            bus = EventBus([sink])

            src = os.path.join(d, "model.txt")
            with open(src, "w", encoding="utf-8") as f:
                f.write("model")

            r = Runner(event_bus=bus)

            def body(ctx):
                ctx.io.save_artifact(
                    src,
                    kind="model",
                    name="tiny-model",
                    meta={"format": "txt"},
                )

            res = r.run(
                {"run": {"id": "r_save_artifact"}, "seed": 42, "paths": {"root": d}},
                body=body,
            )

            self.assertEqual(res.status, "success")
            self.assertEqual(len(res.artifacts), 1)
            self.assertEqual(res.artifacts[0].kind, "model")
            self.assertEqual(res.artifacts[0].name, "tiny-model")
            self.assertEqual(res.artifacts[0].meta["format"], "txt")

            artifact_events = [e for e in sink.events if type(e).__name__ == "ArtifactSaved"]
            self.assertEqual(len(artifact_events), 1)
            self.assertEqual(artifact_events[0].ref.name, "tiny-model")

            with open(res.manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)

            self.assertEqual(len(m["outputs"]["artifacts"]), 1)
            self.assertEqual(m["outputs"]["artifacts"][0]["name"], "tiny-model")

    def test_runner_failure_writes_failed_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            r = Runner()

            def boom(_ctx):
                raise ValueError("boom")

            res = r.run(
                {"run": {"id": "r2"}, "seed": 1, "paths": {"root": d}},
                body=boom,
            )

            self.assertEqual(res.status, "failed")
            with open(res.manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)

            self.assertEqual(m["run_id"], "r2")
            self.assertEqual(m["status"], "failed")
            self.assertIsNotNone(m["error"])
            self.assertIn("code", m["error"])
            self.assertIn("message", m["error"])
            self.assertEqual(m["metrics"], {})

    def test_runner_failure_publishes_run_failed_event(self):
        with tempfile.TemporaryDirectory() as d:
            sink = _Sink()
            bus = EventBus([sink])

            r = Runner(event_bus=bus)

            def boom(_ctx):
                raise ValueError("boom")

            res = r.run(
                {"run": {"id": "r_failed_event"}, "seed": 1, "paths": {"root": d}},
                body=boom,
            )

            self.assertEqual(res.status, "failed")

            failed_events = [e for e in sink.events if type(e).__name__ == "RunFailed"]
            self.assertEqual(len(failed_events), 1)
            self.assertIn("error", failed_events[0].data)
            self.assertIn("code", failed_events[0].data["error"])
            self.assertIn("message", failed_events[0].data["error"])

            finished_events = [e for e in sink.events if type(e).__name__ == "RunFinished"]
            self.assertEqual(len(finished_events), 1)
            self.assertEqual(finished_events[0].data["status"], "failed")


if __name__ == "__main__":
    unittest.main()