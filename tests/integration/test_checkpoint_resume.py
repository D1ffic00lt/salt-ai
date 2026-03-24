import json
import os
import tempfile
import unittest

from saltai import Runner, Trainer


class _TinyDataModule(object):
    def __init__(self):
        self._train = [
            (-2.0, 0.0),
            (-1.0, 0.0),
            (1.0, 1.0),
            (2.0, 1.0),
        ]

    def prepare(self):
        return None

    def train_iter(self):
        return iter(self._train)

    def val_iter(self):
        return None

    def test_iter(self):
        return None


class _TinyModel(object):
    def __init__(self, lr=0.1):
        self.w = 0.0
        self.b = 0.0
        self.lr = float(lr)

        self._x = 0.0
        self._y = 0.0
        self._pred = 0.0
        self._grad_w = 0.0
        self._grad_b = 0.0

    def forward(self, batch):
        x, y = batch
        self._x = float(x)
        self._y = float(y)
        self._pred = self.w * self._x + self.b
        return self._pred

    def loss(self, pred, batch):
        _, y = batch
        err = float(pred) - float(y)
        return err * err

    def zero_grad(self):
        self._grad_w = 0.0
        self._grad_b = 0.0

    def backward(self, loss):
        err = self._pred - self._y
        self._grad_w = 2.0 * err * self._x
        self._grad_b = 2.0 * err

    def step(self):
        self.w -= self.lr * self._grad_w
        self.b -= self.lr * self._grad_b

    def state_dict(self):
        return {
            "w": self.w,
            "b": self.b,
            "lr": self.lr,
        }

    def load_state_dict(self, state):
        self.w = float(state["w"])
        self.b = float(state["b"])
        self.lr = float(state["lr"])


class TestCheckpointResume(unittest.TestCase):
    def test_runner_resumes_from_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = {
                "run": {"id": "resume-local-run"},
                "seed": 42,
                "paths": {"root": d},
            }

            saved = {}

            def first_body(ctx):
                datamodule = _TinyDataModule()
                model = _TinyModel(lr=0.2)

                trainer = Trainer(
                    event_bus=ctx.io.bus,
                    run_id=ctx.run_id,
                )

                train = trainer.fit(
                    model,
                    datamodule,
                    epochs=3,
                )

                step = int(train.extra["steps"])
                ref = ctx.io.save_latest(model, step=step)

                saved["step"] = step
                saved["state"] = dict(model.state_dict())
                saved["ref_uri"] = ref.uri

                return {
                    "step": step,
                    "state": saved["state"],
                }

            first_runner = Runner(
                record_events=True,
                store_artifacts=True,
                enable_checkpoints=True,
            )

            first_res = first_runner.run(cfg, body=first_body)

            self.assertEqual(first_res.status, "success")
            self.assertIn("step", saved)
            self.assertIn("state", saved)
            self.assertTrue(os.path.exists(saved["ref_uri"].replace("file://", "", 1)))

            restored = {}

            def second_body(ctx):
                restored["has_resume_ref"] = ctx.io.resume_ref is not None
                restored["payload"] = ctx.io.resume_payload

                payload = ctx.io.resume_payload
                if payload is None:
                    return {"loaded": False}

                model = _TinyModel(lr=999.0)
                model.load_state_dict(payload["state"])

                restored["state"] = dict(model.state_dict())

                return {
                    "loaded": True,
                    "resume": {
                        "tag": payload["tag"],
                        "step": payload["step"],
                    },
                    "state": restored["state"],
                }

            second_runner = Runner(
                record_events=True,
                store_artifacts=True,
                enable_checkpoints=True,
            )

            second_res = second_runner.run(
                cfg,
                body=second_body,
                resume_from="latest",
            )

            self.assertEqual(second_res.status, "success")
            self.assertTrue(restored["has_resume_ref"])
            self.assertIsNotNone(restored["payload"])

            payload = restored["payload"]
            self.assertEqual(payload["tag"], "latest")
            self.assertEqual(payload["step"], saved["step"])
            self.assertEqual(restored["state"], saved["state"])

            self.assertEqual(second_res.metrics.values["loaded"], True)
            self.assertEqual(second_res.metrics.values["resume"]["tag"], "latest")
            self.assertEqual(second_res.metrics.values["resume"]["step"], saved["step"])
            self.assertEqual(second_res.metrics.values["state"], saved["state"])

            with open(second_res.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            self.assertEqual(manifest["status"], "success")
            self.assertEqual(manifest["metrics"], second_res.metrics.values)
            self.assertIsNotNone(manifest["inputs"]["resume"])
            self.assertIsNotNone(manifest["outputs"]["checkpoints"]["resume_from"])
            self.assertEqual(
                manifest["inputs"]["resume"],
                manifest["outputs"]["checkpoints"]["resume_from"],
            )

            resume_path = manifest["inputs"]["resume"]["uri"].replace("file://", "", 1)
            self.assertTrue(os.path.exists(resume_path))
