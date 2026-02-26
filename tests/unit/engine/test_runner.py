import json
import os
import tempfile
import unittest

from saltai.engine.runner.runner import Runner
from saltai.engine.event_bus.bus import EventBus


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

            self.assertTrue(any(type(e).__name__ == "RunStarted" for e in sink.events))
            self.assertTrue(any(type(e).__name__ == "RunFinished" for e in sink.events))

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
