from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from saltai import Runner, Trainer


class TinyBinaryDataModule(object):
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

    def prepare(self) -> None:
        return None

    def train_iter(self):
        return iter(self._train)

    def val_iter(self):
        return iter(self._val)

    def test_iter(self):
        return None


class TinyLogisticModel(object):
    def __init__(self, lr: float = 0.2):
        self.w = 0.0
        self.b = 0.0
        self.lr = float(lr)

        self._x = 0.0
        self._y = 0
        self._pred = 0.5
        self._grad_w = 0.0
        self._grad_b = 0.0

    @staticmethod
    def _sigmoid(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z))

    def forward(self, batch: tuple[float, int]) -> float:
        x, y = batch
        self._x = float(x)
        self._y = int(y)
        self._pred = self._sigmoid(self.w * self._x + self.b)
        return self._pred

    def loss(self, pred: float, batch: tuple[float, int]) -> float:
        _, y = batch
        p = min(max(float(pred), 1e-12), 1.0 - 1e-12)
        y_i = int(y)
        return -(y_i * math.log(p) + (1 - y_i) * math.log(1.0 - p))

    def zero_grad(self) -> None:
        self._grad_w = 0.0
        self._grad_b = 0.0

    def backward(self, loss: float) -> None:
        err = self._pred - self._y
        self._grad_w = err * self._x
        self._grad_b = err

    def step(self) -> None:
        self.w -= self.lr * self._grad_w
        self.b -= self.lr * self._grad_b

    def state_dict(self) -> dict[str, Any]:
        return {
            "w": self.w,
            "b": self.b,
            "lr": self.lr,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.w = float(state["w"])
        self.b = float(state["b"])
        self.lr = float(state["lr"])


class AccuracyMetric(object):
    name = "accuracy"

    def __init__(self):
        self.correct = 0
        self.total = 0

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, batch: tuple[float, int], pred: float) -> None:
        _, y = batch
        label = int(float(pred) >= 0.5)
        self.correct += int(label == int(y))
        self.total += 1

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class AverageLossMetric(object):
    name = "loss"

    def __init__(self):
        self.total_loss = 0.0
        self.total = 0

    def reset(self) -> None:
        self.total_loss = 0.0
        self.total = 0

    def update(self, batch: tuple[float, int], pred: float) -> None:
        _, y = batch
        p = min(max(float(pred), 1e-12), 1.0 - 1e-12)
        y_i = int(y)
        loss = -(y_i * math.log(p) + (1 - y_i) * math.log(1.0 - p))
        self.total_loss += loss
        self.total += 1

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.total_loss / self.total


def main() -> None:
    cfg = {
        "run": {
            "id": "full-local-run",
        },
        "seed": 42,
        "paths": {
            "root": "examples/runs",
        },
    }

    runner = Runner(
        record_events=True,
        store_artifacts=True,
        enable_checkpoints=True,
    )

    def body(ctx):
        datamodule = TinyBinaryDataModule()
        model = TinyLogisticModel(lr=0.25)

        trainer = Trainer(
            event_bus=ctx.io.bus,
            run_id=ctx.run_id,
        )

        train = trainer.fit(
            model,
            datamodule,
            metrics=[
                AverageLossMetric(),
                AccuracyMetric(),
            ],
            epochs=8,
        )

        val = trainer.evaluate(
            model,
            datamodule,
            metrics=[
                AverageLossMetric(),
                AccuracyMetric(),
            ],
            split="val",
        )

        ctx.io.save_latest(model, step=train.extra["steps"])
        ctx.io.save_best(model, metric=val.values["accuracy"], step=train.extra["steps"])

        return {
            "train": train.values,
            "val": val.values,
        }

    result = runner.run(cfg, body=body)

    print("run_id:", result.run_id)
    print("status:", result.status)
    print("run_dir:", result.context["run_dir"])
    print("manifest_path:", result.manifest_path)
    print("metrics:", json.dumps(result.metrics.values, indent=2, ensure_ascii=False))

    manifest_path = Path(result.manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("manifest metrics:", json.dumps(manifest["metrics"], indent=2, ensure_ascii=False))
    print("events:", manifest_path.parent / "events.jsonl")
    print("checkpoints:", manifest_path.parent / "checkpoints")


if __name__ == "__main__":
    main()
