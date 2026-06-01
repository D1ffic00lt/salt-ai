from __future__ import annotations

import logging
import time

from saltai.config.validation.validate import validate_config
from saltai.engine.event_bus.bus import EventBus
from saltai.logging.setup import setup_logging
from saltai.logging.sinks.console import ConsoleLogger
from saltai.logging.sinks.jsonl import JsonlLogger
from saltai.logging.sinks.python_logging import PythonLoggingSink
from saltai.manifest.io.writer import write_manifest_atomic
from saltai.manifest.model.run import RunManifest
from saltai.utils.hashing.stable import sha256_text, stable_json_dumps
from saltai.utils.typing.core import RunId
from saltai.utils.typing.events import RunStarted


def main() -> None:
    cfg = {
        "run": {"id": "example-config"},
        "seed": 7,
        "paths": {"root": "./runs"},
    }

    resolved = validate_config(cfg)
    canonical = stable_json_dumps(cfg)
    print("resolved:", resolved)
    print("hash:", sha256_text(canonical))

    m = RunManifest(
        run_id="example-manifest",
        status="success",
        started_ts=time.time() - 1,
        finished_ts=time.time(),
        config_hash=resolved.config_hash,
        inputs={"cfg": cfg},
        outputs={"artifacts": []},
        metrics={"acc": 0.99},
        error=None,
        extra={},
    )
    write_manifest_atomic(m, "./tmp/manifest.json")

    setup_logging(level=logging.INFO, file="./tmp/app.log")
    bus = EventBus(
        [
            ConsoleLogger(pretty=True),
            JsonlLogger("./tmp/events.jsonl", flush_each=True),
            PythonLoggingSink("saltai.events", logging.INFO),
        ]
    )
    bus.publish(RunStarted(type="run_started", run_id=RunId("example"), ts=time.time(), data={}))
    bus.flush()
    bus.close()


if __name__ == "__main__":
    main()
