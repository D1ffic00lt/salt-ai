from __future__ import annotations

import os
import tempfile
import time
from pprint import pprint

from saltai import Runner
from saltai.integrations.cloud import create_cloud_event_bus

base_url = os.environ["SALTAI_CLOUD_BASE_URL"]
api_token = os.environ["SALTAI_CLOUD_API_TOKEN"]
project_id = os.environ["SALTAI_CLOUD_PROJECT_ID"]

local_run_id = f"cloud-helper-smoke-{int(time.time())}"

cfg = {
    "run": {
        "id": local_run_id,
    },
    "seed": 42,
    "paths": {
        "root": tempfile.mkdtemp(prefix="saltai-cloud-helper-smoke-"),
    },
}

client, cloud_run, event_bus = create_cloud_event_bus(
    base_url=base_url,
    api_token=api_token,
    project_id=project_id,
    run_name=local_run_id,
    config=cfg,
    manifest={
        "source": "saltai-cloud-helper-smoke",
        "local_run_id": local_run_id,
    },
    tags=["smoke", "cloud", "helper"],
    raise_on_error=True,
)

runner = Runner(event_bus=event_bus)


def body(ctx):
    ctx.io.log_metric("loss", 0.321, step=1, epoch=1, split="train")
    return {"loss": 0.321}


result = runner.run(cfg, body=body)

cloud_run_id = str(cloud_run["id"])

print("\nlocal runner result:")
pprint(
    {
        "run_id": str(result.run_id),
        "status": result.status,
        "metrics": result.metrics.values,
    }
)

print("\ncloud run:")
pprint(client.get_run(cloud_run_id))

print("\ncloud metrics:")
pprint(client.list_metrics(cloud_run_id))

print("\ncloud events:")
pprint(client.list_events(cloud_run_id))
