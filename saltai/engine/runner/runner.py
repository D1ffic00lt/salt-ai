from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from typing import Callable, Any

from saltai.config.validation.validate import validate_config
from saltai.manifest.io.writer import write_manifest_atomic
from saltai.manifest.model.run import RunManifest
from saltai.utils.errors.base import SaltAIError
from saltai.utils.errors.helpers import wrap_unknown
from saltai.utils.typing.core import MetricSummary, RunId, RunResult
from saltai.utils.typing.events import RunStarted, RunFinished, StageStarted, StageFinished
from saltai.engine.event_bus.bus import EventBus

from saltai.logging.sinks.jsonl import JsonlLogger
from saltai.artifacts.store.local import LocalArtifactStore


@dataclass(frozen=True, slots=True)
class RunContext(object):
    run_id: str
    run_dir: str
    config_hash: str


class Runner(object):
    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
        record_events: bool = False,
        store_artifacts: bool = False,
    ):
        self._bus = event_bus or EventBus([])
        self._record_events = record_events
        self._store_artifacts = store_artifacts

    def run(self, cfg: dict, *, body: Callable[[RunContext], Any] | None = None) -> RunResult:
        rcfg = validate_config(cfg)

        run_dir = os.path.join(rcfg.root, rcfg.run_id)
        os.makedirs(run_dir, exist_ok=True)

        manifest_path = os.path.join(run_dir, "manifest.json")
        started = time.time()

        ctx = RunContext(run_id=rcfg.run_id, run_dir=run_dir, config_hash=rcfg.config_hash)

        event_logger = None
        if self._record_events:
            event_logger = JsonlLogger(os.path.join(run_dir, "events.jsonl"), flush_each=True)
            self._bus.add_sink(event_logger)

        def pub(ev: object):
            self._bus.publish(ev, context={"run_id": rcfg.run_id, "run_dir": run_dir})

        pub(RunStarted(type="run_started", run_id=RunId(rcfg.run_id), ts=time.time(), data={}))

        status = "success"
        err_info = None
        artifacts = []

        try:
            pub(StageStarted(type="stage_started", run_id=RunId(rcfg.run_id), ts=time.time(), data={}, stage="run"))
            if body is not None:
                body(ctx)
            pub(StageFinished(type="stage_finished", run_id=RunId(rcfg.run_id), ts=time.time(), data={}, stage="run"))
        except BaseException as e:
            status = "failed"
            se = e if isinstance(e, SaltAIError) else wrap_unknown(e, context={"run_id": rcfg.run_id})
            err_info = asdict(se.to_info())
        finally:
            pub(RunFinished(type="run_finished", run_id=RunId(rcfg.run_id), ts=time.time(), data={"status": status}))
            finished = time.time()

            if self._store_artifacts and self._record_events:
                store = LocalArtifactStore(root=os.path.join(run_dir, "artifacts"))
                ref = store.put(os.path.join(run_dir, "events.jsonl"), kind="log", name="events", meta={})
                artifacts.append(ref)

            m = RunManifest(
                run_id=rcfg.run_id,
                status=status,
                started_ts=started,
                finished_ts=finished,
                config_hash=rcfg.config_hash,
                inputs={},
                outputs={"artifacts": [asdict(a) for a in artifacts]},
                metrics={},
                error=err_info,
                extra={},
            )
            write_manifest_atomic(m, manifest_path)

            self._bus.flush()
            self._bus.close()

        return RunResult(
            run_id=RunId(rcfg.run_id),
            status=status,
            metrics=MetricSummary(values={}, extra={}),
            artifacts=tuple(artifacts),
            manifest_path=manifest_path,
            context={"run_dir": run_dir, "config_hash": rcfg.config_hash},
        )
