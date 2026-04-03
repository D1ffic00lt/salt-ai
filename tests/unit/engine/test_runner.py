import json
import os
import tempfile
import unittest

from saltai.engine.event_bus.bus import EventBus
from saltai.engine.runner.runner import Runner
from saltai.utils.typing.core import ArtifactId, ArtifactRef, MetricSummary


class _Sink(object):
    def __init__(self):
        self.events = []

    def log(self, event):
        self.events.append(event)

    def flush(self):
        return None

    def close(self):
        return None


class _FakeArtifactStore(object):
    def __init__(self):
        self.put_calls = []
        self.refs = []

    def put(self, local_path, *, kind, name, meta=None):
        ref = ArtifactRef(
            id=ArtifactId(f"fake-{len(self.refs) + 1}"),
            kind=kind,
            name=name,
            uri=f"memory://{kind}/{name}",
            sha256=None,
            size_bytes=None,
            meta=meta or {},
        )
        self.put_calls.append(
            {
                "local_path": local_path,
                "kind": kind,
                "name": name,
                "meta": meta or {},
            }
        )
        self.refs.append(ref)
        return ref

    def get(self, ref, *, dst_dir):
        return os.path.join(dst_dir, ref.name)

    def exists(self, ref):
        return ref in self.refs

    def list(self, *, kind=None):
        if kind is None:
            return tuple(self.refs)
        return tuple(ref for ref in self.refs if ref.kind == kind)


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

    def test_runner_uses_injected_artifact_store_factory_for_runio_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            created = []
            store = _FakeArtifactStore()

            def make_store(run_dir):
                created.append(run_dir)
                return store

            src = os.path.join(d, "model.txt")
            with open(src, "w", encoding="utf-8") as f:
                f.write("model")

            r = Runner(artifact_store_factory=make_store)

            def body(ctx):
                ref = ctx.io.save_artifact(
                    src,
                    kind="model",
                    name="remote-model",
                    meta={"format": "txt"},
                )
                return {"artifact_uri": ref.uri}

            res = r.run(
                {"run": {"id": "r_injected_store"}, "seed": 42, "paths": {"root": d}},
                body=body,
            )

            self.assertEqual(res.status, "success")
            self.assertEqual(created, [os.path.join(d, "r_injected_store")])

            self.assertEqual(len(store.put_calls), 1)
            self.assertEqual(store.put_calls[0]["local_path"], src)
            self.assertEqual(store.put_calls[0]["kind"], "model")
            self.assertEqual(store.put_calls[0]["name"], "remote-model")
            self.assertEqual(store.put_calls[0]["meta"], {"format": "txt"})

            self.assertEqual(len(res.artifacts), 1)
            self.assertEqual(res.artifacts[0].uri, "memory://model/remote-model")
            self.assertEqual(res.metrics.values["artifact_uri"], "memory://model/remote-model")

            with open(res.manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)

            self.assertEqual(m["outputs"]["artifacts"][0]["uri"], "memory://model/remote-model")
            self.assertEqual(m["outputs"]["artifacts"][0]["name"], "remote-model")

    def test_runner_uses_injected_artifact_store_factory_for_recorded_events_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            store = _FakeArtifactStore()

            r = Runner(
                record_events=True,
                store_artifacts=True,
                artifact_store_factory=lambda _run_dir: store,
            )

            res = r.run(
                {"run": {"id": "r_injected_events_store"}, "seed": 42, "paths": {"root": d}},
                body=lambda _ctx: None,
            )

            self.assertEqual(res.status, "success")
            self.assertEqual(len(store.put_calls), 1)
            self.assertEqual(store.put_calls[0]["kind"], "log")
            self.assertEqual(store.put_calls[0]["name"], "events")

            with open(res.manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)

            self.assertEqual(m["outputs"]["artifacts"][0]["uri"], "memory://log/events")
            self.assertEqual(m["outputs"]["artifacts"][0]["kind"], "log")
            self.assertEqual(m["outputs"]["artifacts"][0]["name"], "events")

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

    def test_runner_record_events_writes_run_failed_to_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            r = Runner(record_events=True)

            def boom(_ctx):
                raise ValueError("boom")

            res = r.run(
                {"run": {"id": "r_failed_jsonl"}, "seed": 1, "paths": {"root": d}},
                body=boom,
            )

            self.assertEqual(res.status, "failed")

            events_path = os.path.join(d, "r_failed_jsonl", "events.jsonl")
            self.assertTrue(os.path.exists(events_path))

            with open(events_path, "r", encoding="utf-8") as f:
                events = [json.loads(line) for line in f]

            event_types = [e["type"] for e in events]

            self.assertIn("run_started", event_types)
            self.assertIn("stage_started", event_types)
            self.assertIn("run_failed", event_types)
            self.assertIn("run_finished", event_types)

            failed = [e for e in events if e["type"] == "run_failed"]
            self.assertEqual(len(failed), 1)
            self.assertIn("error", failed[0]["data"])
            self.assertIn("code", failed[0]["data"]["error"])
            self.assertIn("message", failed[0]["data"]["error"])

            finished = [e for e in events if e["type"] == "run_finished"]
            self.assertEqual(len(finished), 1)
            self.assertEqual(finished[0]["data"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
