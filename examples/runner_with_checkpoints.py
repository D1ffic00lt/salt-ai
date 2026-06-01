from __future__ import annotations

from saltai.engine.event_bus.bus import EventBus
from saltai.engine.runner.runner import Runner
from saltai.logging.sinks.console import ConsoleLogger


class DummyModel(object):
    def __init__(self) -> None:
        self.weight = 0.0

    def state_dict(self) -> dict[str, float]:
        return {"weight": self.weight}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.weight = float(state["weight"])


def main() -> None:
    cfg = {
        "run": {"id": "example-runner"},
        "seed": 42,
        "paths": {"root": "./runs"},
    }

    bus = EventBus([ConsoleLogger(pretty=False)])
    runner = Runner(
        event_bus=bus,
        record_events=True,
        store_artifacts=True,
        enable_checkpoints=True,
        checkpoint_keep_last=2,
    )

    model = DummyModel()

    def body(ctx) -> None:
        model.weight = 0.1
        ctx.io.save_latest(model, step=1)

        model.weight = 0.2
        ctx.io.save_best(model, metric=0.91, step=2)

    result = runner.run(cfg, body=body)
    print("status:", result.status)
    print("manifest:", result.manifest_path)


if __name__ == "__main__":
    main()
