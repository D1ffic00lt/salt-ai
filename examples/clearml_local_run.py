from __future__ import annotations

from saltai import Runner
from saltai.engine.event_bus.bus import EventBus
from saltai_ext.clearml import ClearMLLogger


def main():
    clearml_logger = ClearMLLogger(
        project_name="SaltAI",
        task_name="clearml-local-run",
        report_events=True,
    )

    runner = Runner(
        event_bus=EventBus([clearml_logger]),
        record_events=True,
        store_artifacts=True,
    )

    cfg = {
        "run": {
            "id": "clearml-demo",
        },
        "seed": 42,
        "paths": {
            "root": "examples/runs",
        },
    }

    def body(ctx):
        ctx.io.publish_metric if False else None
        return {"accuracy": 0.95, "loss": 0.12}

    result = runner.run(cfg, body=body)

    print("status:", result.status)
    print("metrics:", result.metrics.values)
    print("manifest:", result.manifest_path)


if __name__ == "__main__":
    main()
