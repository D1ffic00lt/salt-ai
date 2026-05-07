from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from saltai import Runner
from saltai.integrations.cloud import create_cloud_run_context


def main() -> None:
    base_url = os.environ.get("SALTAI_CLOUD_BASE_URL", "http://localhost:8080")
    api_token = require_env("SALTAI_CLOUD_API_TOKEN")
    project_id = require_env("SALTAI_CLOUD_PROJECT_ID")

    run_name = f"miniapp-demo-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    config = {
        "model": "tiny-demo-model",
        "dataset": "synthetic-binary",
        "epochs": 6,
        "learning_rate": 0.01,
        "seed": 42,
    }

    manifest = {
        "entrypoint": "examples/cloud_demo_run.py",
        "kind": "miniapp-demo",
        "created_at": now_iso(),
    }

    client, cloud_run, event_bus, artifact_store = create_cloud_run_context(
        base_url=base_url,
        api_token=api_token,
        project_id=project_id,
        run_name=run_name,
        config=config,
        manifest=manifest,
        tags=["demo", "miniapp", "cloud"],
        raise_on_error=True,
        log_lifecycle_events=True,
        log_artifacts=False,
    )

    runner = Runner(
        event_bus=event_bus,
        record_events=True,
        store_artifacts=False,
        enable_checkpoints=False,
        artifact_store_factory=lambda _: artifact_store,
    )

    cfg = {
        "run": {
            "id": run_name,
        },
        "seed": 42,
        "paths": {
            "root": "examples/runs",
        },
    }

    def body(ctx):
        output_dir = Path(ctx.run_dir) / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        loss = 1.0
        accuracy = 0.45

        for epoch in range(1, config["epochs"] + 1):
            loss = max(0.03, loss * random.uniform(0.68, 0.82))
            accuracy = min(0.99, accuracy + random.uniform(0.055, 0.105))

            ctx.io.log_metric(
                "train/loss",
                round(loss, 6),
                step=epoch,
                epoch=epoch,
                split="train",
                extra={"source": "saltai-runner"},
            )

            ctx.io.log_metric(
                "val/accuracy",
                round(accuracy, 6),
                step=epoch,
                epoch=epoch,
                split="val",
                extra={"source": "saltai-runner"},
            )

            client.log_event(
                str(cloud_run["id"]),
                "epoch_finished",
                level="info",
                message=f"Epoch {epoch} finished",
                payload={
                    "epoch": epoch,
                    "loss": round(loss, 6),
                    "accuracy": round(accuracy, 6),
                },
            )

            time.sleep(0.1)

        report_path = output_dir / "demo-report.json"
        report_path.write_text(
            json.dumps(
                {
                    "cloud_run_id": cloud_run["id"],
                    "local_run_id": ctx.run_id,
                    "config": config,
                    "final_metrics": {
                        "train/loss": round(loss, 6),
                        "val/accuracy": round(accuracy, 6),
                    },
                    "created_at": now_iso(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        checkpoint_path = output_dir / "demo-checkpoint.txt"
        checkpoint_path.write_text(
            "\n".join(
                [
                    "SaltAI Cloud demo checkpoint",
                    f"cloud_run_id={cloud_run['id']}",
                    f"local_run_id={ctx.run_id}",
                    f"loss={round(loss, 6)}",
                    f"accuracy={round(accuracy, 6)}",
                    f"created_at={now_iso()}",
                ]
            ),
            encoding="utf-8",
        )

        ctx.io.save_artifact(
            str(report_path),
            kind="manifest",
            name="demo-report.json",
            meta={
                "demo": True,
                "source": "saltai.integrations.cloud",
            },
        )

        ctx.io.save_artifact(
            str(checkpoint_path),
            kind="checkpoint",
            name="demo-checkpoint.txt",
            meta={
                "demo": True,
                "epoch": config["epochs"],
            },
        )

        ctx.io.bus.flush()

        events_path = Path(ctx.run_dir) / "events.jsonl"
        if events_path.exists():
            ctx.io.save_artifact(
                str(events_path),
                kind="log",
                name="events.jsonl",
                meta={
                    "demo": True,
                    "source": "saltai-runner",
                },
            )

        return {
            "train/loss": round(loss, 6),
            "val/accuracy": round(accuracy, 6),
        }

    result = runner.run(cfg, body=body)

    details = client.get_run_details(str(cloud_run["id"]))

    print("local_run_id:", result.run_id)
    print("cloud_run_id:", cloud_run["id"])
    print("status:", result.status)
    print("manifest_path:", result.manifest_path)
    print("cloud_metrics:", len(details["metrics"]))
    print("cloud_events:", len(details["events"]))
    print("cloud_artifacts:", len(details["artifacts"]))


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable is required: {name}")
    return value


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
