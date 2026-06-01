import json
import math
import os
import tempfile
import unittest

from saltai import Runner, Trainer


class _TinyBinaryDataModule(object):
    def __init__(self):
        self._train = [
            (-2.0, 0),
            (-1.5, 0),
            (-1.0, 0),
            (-0.5, 0),
            (0.5, 1),
            (1.0, 1),
            (1.5, 1),
            (2.0, 1),
        ]
        self._val = [
            (-1.25, 0),
            (-0.25, 0),
            (0.25, 1),
            (1.25, 1),
        ]

    def prepare(self):
        return None

    def train_iter(self):
        return iter(self._train)

    def val_iter(self):
        return iter(self._val)

    def test_iter(self):
        return None


class _TinyLogisticModel(object):
    def __init__(self, lr=0.25):
        self.w = 0.0
        self.b = 0.0
        self.lr = float(lr)

        self._x = 0.0
        self._y = 0
        self._pred = 0.5
        self._grad_w = 0.0
        self._grad_b = 0.0

    def _sigmoid(self, z):
        return 1.0 / (1.0 + math.exp(-z))

    def forward(self, batch):
        x, y = batch
        self._x = float(x)
        self._y = int(y)
        self._pred = self._sigmoid(self.w * self._x + self.b)
        return self._pred

    def loss(self, pred, batch):
        _, y = batch
        p = min(max(float(pred), 1e-12), 1.0 - 1e-12)
        y_i = int(y)
        return -(y_i * math.log(p) + (1 - y_i) * math.log(1.0 - p))

    def zero_grad(self):
        self._grad_w = 0.0
        self._grad_b = 0.0

    def backward(self, loss):
        err = self._pred - self._y
        self._grad_w = err * self._x
        self._grad_b = err

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


class _AccuracyMetric(object):
    name = "accuracy"

    def __init__(self):
        self.correct = 0
        self.total = 0

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, batch, pred):
        _, y = batch
        label = int(float(pred) >= 0.5)
        self.correct += int(label == int(y))
        self.total += 1

    def compute(self):
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class _AverageLossMetric(object):
    name = "loss"

    def __init__(self):
        self.total_loss = 0.0
        self.total = 0

    def reset(self):
        self.total_loss = 0.0
        self.total = 0

    def update(self, batch, pred):
        _, y = batch
        p = min(max(float(pred), 1e-12), 1.0 - 1e-12)
        y_i = int(y)
        loss = -(y_i * math.log(p) + (1 - y_i) * math.log(1.0 - p))
        self.total_loss += loss
        self.total += 1

    def compute(self):
        if self.total == 0:
            return 0.0
        return self.total_loss / self.total


class TestFullLocalRun(unittest.TestCase):
    def test_full_local_run_writes_metrics_events_manifest_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = {
                "run": {
                    "id": "full-local-run-test",
                },
                "seed": 42,
                "paths": {
                    "root": d,
                },
            }

            runner = Runner(
                record_events=True,
                store_artifacts=True,
                enable_checkpoints=True,
            )

            def body(ctx):
                datamodule = _TinyBinaryDataModule()
                model = _TinyLogisticModel(lr=0.25)

                trainer = Trainer(
                    event_bus=ctx.io.bus,
                    run_id=ctx.run_id,
                )

                train = trainer.fit(
                    model,
                    datamodule,
                    metrics=[
                        _AverageLossMetric(),
                        _AccuracyMetric(),
                    ],
                    epochs=8,
                )

                val = trainer.evaluate(
                    model,
                    datamodule,
                    metrics=[
                        _AverageLossMetric(),
                        _AccuracyMetric(),
                    ],
                    split="val",
                )

                ctx.io.save_latest(model, step=train.extra["steps"])
                ctx.io.save_best(model, metric=val.values["accuracy"], step=train.extra["steps"])

                return {
                    "train": train.values,
                    "val": val.values,
                }

            res = runner.run(cfg, body=body)

            self.assertEqual(res.status, "success")
            self.assertEqual(str(res.run_id), "full-local-run-test")

            self.assertIn("train", res.metrics.values)
            self.assertIn("val", res.metrics.values)
            self.assertIn("loss", res.metrics.values["train"])
            self.assertIn("accuracy", res.metrics.values["train"])
            self.assertIn("loss", res.metrics.values["val"])
            self.assertIn("accuracy", res.metrics.values["val"])

            self.assertGreaterEqual(res.metrics.values["train"]["accuracy"], 0.5)
            self.assertGreaterEqual(res.metrics.values["val"]["accuracy"], 0.5)

            run_dir = res.context["run_dir"]
            self.assertTrue(os.path.isdir(run_dir))
            self.assertTrue(os.path.isfile(res.manifest_path))

            events_path = os.path.join(run_dir, "events.jsonl")
            checkpoints_dir = os.path.join(run_dir, "checkpoints")

            self.assertTrue(os.path.isfile(events_path))
            self.assertTrue(os.path.isdir(checkpoints_dir))

            with open(res.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            self.assertEqual(manifest["run_id"], "full-local-run-test")
            self.assertEqual(manifest["status"], "success")
            self.assertIsNone(manifest["error"])
            self.assertEqual(manifest["metrics"], res.metrics.values)

            self.assertIsNotNone(manifest["outputs"]["checkpoints"]["latest"])
            self.assertIsNotNone(manifest["outputs"]["checkpoints"]["best"])
            self.assertIsNone(manifest["outputs"]["checkpoints"]["resume_from"])

            self.assertGreaterEqual(len(manifest["outputs"]["artifacts"]), 1)
            self.assertEqual(manifest["outputs"]["artifacts"][0]["kind"], "log")
            self.assertEqual(manifest["outputs"]["artifacts"][0]["name"], "events")

            with open(events_path, "r", encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]

            event_types = [event["type"] for event in events]

            self.assertIn("run_started", event_types)
            self.assertIn("stage_started", event_types)
            self.assertIn("epoch_started", event_types)
            self.assertIn("step_started", event_types)
            self.assertIn("metric", event_types)
            self.assertIn("checkpoint_saved", event_types)
            self.assertIn("stage_finished", event_types)
            self.assertIn("run_finished", event_types)

            ckpt_refs = manifest["outputs"]["checkpoints"]
            latest_path = ckpt_refs["latest"]["uri"].replace("file://", "", 1)
            best_path = ckpt_refs["best"]["uri"].replace("file://", "", 1)

            self.assertTrue(os.path.isfile(latest_path))
            self.assertTrue(os.path.isfile(best_path))

            with open(latest_path, "r", encoding="utf-8") as f:
                latest_payload = json.load(f)

            with open(best_path, "r", encoding="utf-8") as f:
                best_payload = json.load(f)

            self.assertEqual(latest_payload["tag"], "latest")
            self.assertEqual(best_payload["tag"], "best")
            self.assertIn("state", latest_payload)
            self.assertIn("state", best_payload)
            self.assertIn("w", latest_payload["state"])
            self.assertIn("b", latest_payload["state"])


if __name__ == "__main__":
    unittest.main()
