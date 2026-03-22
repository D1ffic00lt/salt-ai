from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from saltai.engine.event_bus.bus import EventBus
from saltai.engine.trainer.trainer import Trainer
from saltai.logging.sinks.console import ConsoleLogger
from saltai.utils.typing.core import ModelAdapter


class ToyDataModule(object):
    def prepare(self) -> None:
        self._train = [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 4.0}]
        self._val = [{"x": 3.0, "y": 6.0}]

    def train_iter(self):
        return iter(self._train)

    def val_iter(self):
        return iter(self._val)

    def test_iter(self):
        return iter(self._val)


class ToyModel(object):
    def __init__(self) -> None:
        self.w = 1.0

    def forward(self, batch: dict[str, float]) -> float:
        return batch["x"] * self.w

    def loss(self, pred: float, batch: dict[str, float]) -> float:
        return abs(pred - batch["y"])

    def zero_grad(self) -> None:
        return

    def backward(self, loss: float) -> None:
        _ = loss

    def step(self) -> None:
        self.w += 0.1

    def state_dict(self) -> dict[str, float]:
        return {"w": self.w}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.w = float(state["w"])


@dataclass
class MAE(object):
    name: str = "mae"

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.err = 0.0
        self.n = 0

    def update(self, batch: dict[str, float], pred: float) -> None:
        self.err += abs(pred - batch["y"])
        self.n += 1

    def compute(self) -> float:
        return self.err / max(1, self.n)


def main() -> None:
    trainer = Trainer(event_bus=EventBus([ConsoleLogger(pretty=False)]), run_id="example-trainer")
    fit_model = cast(ModelAdapter[Any, Any, Any], cast(object, ToyModel()))
    eval_model = cast(ModelAdapter[Any, Any, Any], cast(object, ToyModel()))

    fit_summary = trainer.fit(
        fit_model,
        ToyDataModule(),
        metrics=[MAE()],
        epochs=2,
    )
    eval_summary = trainer.evaluate(
        eval_model,
        ToyDataModule(),
        metrics=[MAE()],
        split="val",
    )

    print("fit:", fit_summary.values)
    print("eval:", eval_summary.values)


if __name__ == "__main__":
    main()
