from __future__ import annotations

import json
import os
import unittest

from tempfile import TemporaryDirectory
from typing import Any, Mapping

from saltai.engine.runner.runner import Runner
from saltai.utils.errors.base import CheckpointError


class DummyObj(object):
    def __init__(self, state: dict[str, Any] | None = None):
        self._state = dict(state or {})

    def state_dict(self) -> Mapping[str, Any]:
        return dict(self._state)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._state = dict(state)


class TestRunnerCheckpoints(unittest.TestCase):
    def test_runner_writes_checkpoints_to_manifest(self) -> None:
        with TemporaryDirectory() as td:
            cfg = {"run": {"id": "r1"}, "seed": 0, "paths": {"root": td}}

            def body(ctx):
                obj = DummyObj({"x": 1})
                ctx.io.save_latest(obj, step=1)
                ctx.io.save_best(obj, metric=0.1, step=1)
                obj2 = DummyObj({"x": 2})
                ctx.io.save_best(obj2, metric=0.9, step=2)

            r = Runner(enable_checkpoints=True, record_events=False, store_artifacts=False)
            res = r.run(cfg, body=body)

            self.assertTrue(os.path.exists(res.manifest_path))

            with open(res.manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)

            self.assertIn("outputs", m)
            self.assertIn("checkpoints", m["outputs"])
            cp = m["outputs"]["checkpoints"]

            self.assertIsNotNone(cp["latest"])
            self.assertIsNotNone(cp["best"])

            self.assertEqual(cp["latest"]["meta"]["tag"], "latest")
            self.assertEqual(cp["latest"]["meta"]["step"], 1)

            self.assertEqual(cp["best"]["meta"]["tag"], "best")
            self.assertEqual(cp["best"]["meta"]["step"], 2)
            self.assertEqual(float(cp["best"]["meta"]["metric"]), 0.9)

    def test_runner_resume_from_latest(self) -> None:
        with TemporaryDirectory() as td:
            cfg = {"run": {"id": "r2"}, "seed": 0, "paths": {"root": td}}

            def body_save(ctx):
                obj = DummyObj({"hello": "world", "k": 123})
                ctx.io.save_latest(obj, step=5)

            r1 = Runner(enable_checkpoints=True, record_events=False, store_artifacts=False)
            _ = r1.run(cfg, body=body_save)

            def body_resume(ctx):
                self.assertIsNotNone(ctx.io.resume_ref)
                self.assertIsNotNone(ctx.io.resume_payload)
                payload = ctx.io.resume_payload
                self.assertIn("state", payload)
                self.assertEqual(payload["state"], {"hello": "world", "k": 123})

                obj2 = DummyObj()
                ctx.io.ckpt.load(obj2, ctx.io.resume_ref)  # type: ignore[union-attr]
                self.assertEqual(obj2.state_dict(), {"hello": "world", "k": 123})

            r2 = Runner(enable_checkpoints=True, record_events=False, store_artifacts=False)
            _ = r2.run(cfg, body=body_resume, resume_from="latest")

    def test_runner_resume_requires_checkpoints_enabled(self) -> None:
        with TemporaryDirectory() as td:
            cfg = {"run": {"id": "r3"}, "seed": 0, "paths": {"root": td}}

            r = Runner(enable_checkpoints=False)
            with self.assertRaises(CheckpointError):
                _ = r.run(cfg, body=lambda ctx: None, resume_from="latest")
