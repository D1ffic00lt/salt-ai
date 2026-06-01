from __future__ import annotations

import json
from pathlib import Path

from saltai import Runner, Trainer
from full_local_run import (
    AccuracyMetric,
    AverageLossMetric,
    TinyBinaryDataModule,
    TinyLogisticModel,
)


def main() -> None:
    cfg = {
        "run": {
            "id": "resume-local-run",
        },
        "seed": 42,
        "paths": {
            "root": "examples/runs",
        },
    }

    first_runner = Runner(
        record_events=True,
        store_artifacts=True,
        enable_checkpoints=True,
    )

    def first_body(ctx):
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
            epochs=4,
        )

        ctx.io.save_latest(model, step=train.extra["steps"])

        return {
            "train": train.values,
            "step": train.extra["steps"],
            "state": model.state_dict(),
        }

    first_result = first_runner.run(cfg, body=first_body)

    second_runner = Runner(
        record_events=True,
        store_artifacts=True,
        enable_checkpoints=True,
    )

    def second_body(ctx):
        payload = ctx.io.resume_payload

        model = TinyLogisticModel(lr=999.0)
        model.load_state_dict(payload["state"])

        datamodule = TinyBinaryDataModule()

        trainer = Trainer(
            event_bus=ctx.io.bus,
            run_id=ctx.run_id,
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

        return {
            "loaded": True,
            "resume": {
                "tag": payload["tag"],
                "step": payload["step"],
            },
            "state": model.state_dict(),
            "val": val.values,
        }

    second_result = second_runner.run(
        cfg,
        body=second_body,
        resume_from="latest",
    )

    print("first_status:", first_result.status)
    print("second_status:", second_result.status)
    print("first_manifest:", first_result.manifest_path)
    print("second_manifest:", second_result.manifest_path)
    print("second_metrics:", json.dumps(second_result.metrics.values, indent=2, ensure_ascii=False))

    manifest_path = Path(second_result.manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("resume_from:", json.dumps(manifest["inputs"]["resume"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
