from __future__ import annotations

import os
import tempfile
import time
from pprint import pprint

from saltai import EventBus, Runner
from saltai.integrations.cloud import CloudClient, CloudRunLogger

base_url = os.environ["SALTAI_CLOUD_BASE_URL"]
api_token = os.environ["SALTAI_CLOUD_API_TOKEN"]
project_id = os.environ["SALTAI_CLOUD_PROJECT_ID"]

client = CloudClient(
    base_url=base_url,
    api_token=api_token,
)

local_run_id = f"cloud-failed-smoke-{int(time.time())}"

cfg = {
    "run": {
        "id": local_run_id,
    },
    "seed": 42,
    "paths": {
        "root": tempfile.mkdtemp(prefix="saltai-cloud-failed-smoke-"),
    },
}

cloud_run = client.create_run(
    project_id=project_id,
    name=local_run_id,
    config=cfg,
    manifest={
        "source": "saltai-cloud-failed-smoke",
        "local_run_id": local_run_id,
    },
    tags=["smoke", "cloud", "runner", "failed"],
)

cloud_run_id = str(cloud_run["id"])

cloud_logger = CloudRunLogger(
    client=client,
    run_id=cloud_run_id,
    raise_on_error=True,
)

bus = EventBus([cloud_logger], fail_fast=True)
runner = Runner(event_bus=bus)


def body(ctx):
    ctx.io.log_metric(
        "loss",
        0.999,
        step=1,
        epoch=1,
        split="train",
        extra={"source": "failed-smoke"},
    )

    raise RuntimeError("intentional cloud failed smoke")


try:
    runner.run(cfg, body=body)
except RuntimeError as e:
    print("\nexpected local failure:")
    print(repr(e))

print("\ncloud run after failed runner:")
pprint(client.get_run(cloud_run_id))

print("\ncloud metrics:")
pprint(client.list_metrics(cloud_run_id))

print("\ncloud events:")
pprint(client.list_events(cloud_run_id))

print("\ncloud logger errors:")
pprint([repr(e) for e in cloud_logger.errors])
