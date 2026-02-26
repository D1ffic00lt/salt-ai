import json
import os
import tempfile
import unittest

from saltai.engine.runner.runner import Runner


class TestRunnerArtifacts(unittest.TestCase):
    def test_runner_saves_events_as_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            r = Runner(record_events=True, store_artifacts=True)
            res = r.run({"run": {"id": "r1"}, "seed": 42, "paths": {"root": d}}, body=lambda ctx: None)

            run_dir = os.path.join(d, "r1")
            self.assertTrue(os.path.exists(os.path.join(run_dir, "events.jsonl")))
            self.assertTrue(os.path.exists(res.manifest_path))

            with open(os.path.join(run_dir, "events.jsonl"), "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            self.assertTrue(len(lines) >= 2)

            with open(res.manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)

            arts = m["outputs"]["artifacts"]
            self.assertTrue(len(arts) >= 1)

            ev = [a for a in arts if a["kind"] == "log" and a["name"] == "events"]
            self.assertEqual(len(ev), 1)
            self.assertTrue(ev[0]["uri"].startswith("file://"))