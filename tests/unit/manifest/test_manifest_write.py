import json
import os
import tempfile
import unittest

from saltai.manifest.io.writer import write_manifest_atomic
from saltai.manifest.model.run import RunManifest


class TestManifestWrite(unittest.TestCase):
    def test_write_manifest_atomic(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "manifest.json")
            m = RunManifest(
                run_id="r1",
                status="success",
                started_ts=1.0,
                finished_ts=2.0,
                config_hash="abc",
                inputs={},
                outputs={"artifacts": []},
                metrics={},
                error=None,
                extra={},
            )
            write_manifest_atomic(m, path)

            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            self.assertEqual(obj["run_id"], "r1")
            self.assertEqual(obj["status"], "success")
            self.assertEqual(obj["config_hash"], "abc")
