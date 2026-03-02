from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable

from saltai.artifacts.checkpoints.manager import CheckpointManager
from saltai.artifacts.store.local import LocalArtifactStore
from saltai.config.validation.validate import validate_config
from saltai.engine.event_bus.bus import EventBus
from saltai.logging.sinks.jsonl import JsonlLogger
from saltai.manifest.io.writer import write_manifest_atomic
from saltai.manifest.model.run import RunManifest
from saltai.utils.errors.base import SaltAIError, CheckpointError
from saltai.utils.errors.codes import EC
from saltai.utils.errors.helpers import wrap_unknown
from saltai.utils.typing.core import ArtifactRef, Checkpointable, MetricSummary, RunId, RunResult
from saltai.utils.typing.events import RunStarted, RunFinished, StageStarted, StageFinished, CheckpointSaved


class RunIO(object):
    __slots__ = (
        "run_id",
        "run_dir",
        "config_hash",
        "bus",
        "store",
        "ckpt",
        "artifacts",
        "latest_ckpt",
        "best_ckpt",
        "resume_ref",
        "resume_payload",
    )

    def __init__(
        self,
        *,
        run_id: RunId,
        run_dir: str,
        config_hash: str,
        bus: EventBus,
        store: LocalArtifactStore,
        ckpt: CheckpointManager | None,
    ):
        self.run_id = run_id
        self.run_dir = run_dir
        self.config_hash = config_hash
        self.bus = bus
        self.store = store
        self.ckpt = ckpt

        self.artifacts: list[ArtifactRef] = []
        self.latest_ckpt: ArtifactRef | None = None
        self.best_ckpt: ArtifactRef | None = None

        self.resume_ref: ArtifactRef | None = None
        self.resume_payload: dict[str, Any] | None = None

    def publish(self, ev: object) -> None:
        self.bus.publish(ev, context={"run_id": str(self.run_id), "run_dir": self.run_dir})

    def save_latest(self, obj: Checkpointable, *, step: int) -> ArtifactRef:
        if self.ckpt is None:
            raise CheckpointError(
                EC.CKPT_SAVE_FAILED,
                "Checkpointing is disabled",
                hint="Enable checkpoints in Runner(enable_checkpoints=True)",
                context={"step": int(step)},
                cause=None,
            )

        ref = self.ckpt.save_latest(obj, step=int(step))
        self.latest_ckpt = ref
        self.publish(
            CheckpointSaved(
                type="checkpoint_saved",
                run_id=self.run_id,
                ts=time.time(),
                data={},
                ref=ref,
                tag="latest",
            )
        )
        return ref

    def save_best(self, obj: Checkpointable, *, metric: float, step: int) -> ArtifactRef:
        if self.ckpt is None:
            raise CheckpointError(
                EC.CKPT_SAVE_FAILED,
                "Checkpointing is disabled",
                hint="Enable checkpoints in Runner(enable_checkpoints=True)",
                context={"step": int(step), "metric": float(metric)},
                cause=None,
            )

        ref = self.ckpt.save_best(obj, metric=float(metric), step=int(step))
        self.best_ckpt = ref
        self.publish(
            CheckpointSaved(
                type="checkpoint_saved",
                run_id=self.run_id,
                ts=time.time(),
                data={},
                ref=ref,
                tag="best",
            )
        )
        return ref


@dataclass(frozen=True, slots=True)
class RunContext(object):
    run_id: str
    run_dir: str
    config_hash: str
    io: RunIO


class Runner(object):
    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
        record_events: bool = False,
        store_artifacts: bool = False,
        enable_checkpoints: bool = False,
        checkpoint_keep_last: int = 3,
    ):
        self._bus = event_bus or EventBus([])
        self._record_events = bool(record_events)
        self._store_artifacts = bool(store_artifacts)
        self._enable_checkpoints = bool(enable_checkpoints)
        self._checkpoint_keep_last = int(checkpoint_keep_last)

    def run(
        self,
        cfg: dict,
        *,
        body: Callable[[RunContext], Any] | None = None,
        resume_from: ArtifactRef | str | None = None,
    ) -> RunResult:
        rcfg = validate_config(cfg)

        run_dir = os.path.join(rcfg.root, rcfg.run_id)
        os.makedirs(run_dir, exist_ok=True)

        manifest_path = os.path.join(run_dir, "manifest.json")
        started = time.time()

        store = LocalArtifactStore(root=os.path.join(run_dir, "artifacts"))

        ckpt = None
        if self._enable_checkpoints:
            ckpt = CheckpointManager(
                root=os.path.join(run_dir, "checkpoints"),
                keep_last=self._checkpoint_keep_last,
            )

        io = RunIO(
            run_id=RunId(rcfg.run_id),
            run_dir=run_dir,
            config_hash=rcfg.config_hash,
            bus=self._bus,
            store=store,
            ckpt=ckpt,
        )

        ctx = RunContext(
            run_id=rcfg.run_id,
            run_dir=run_dir,
            config_hash=rcfg.config_hash,
            io=io,
        )

        if self._record_events:
            event_logger = JsonlLogger(os.path.join(run_dir, "events.jsonl"), flush_each=True)
            self._bus.add_sink(event_logger)

        def pub(ev: object) -> None:
            self._bus.publish(ev, context={"run_id": rcfg.run_id, "run_dir": run_dir})

        pub(RunStarted(type="run_started", run_id=RunId(rcfg.run_id), ts=time.time(), data={}))

        status = "success"
        err_info = None

        if resume_from is not None:
            if ckpt is None:
                raise CheckpointError(
                    EC.CKPT_LOAD_FAILED,
                    "resume_from is set but checkpoints are disabled",
                    hint="Enable checkpoints in Runner(enable_checkpoints=True)",
                    context={"resume_from": str(resume_from)},
                    cause=None,
                )

            if isinstance(resume_from, str):
                ref = ckpt.resolve(resume_from)
                if ref is None:
                    raise CheckpointError(
                        EC.CKPT_LOAD_FAILED,
                        "No checkpoint found for resume",
                        hint="Save at least one checkpoint (latest/best) before resuming",
                        context={"resume_from": resume_from},
                        cause=None,
                    )
            else:
                ref = resume_from

            io.resume_ref = ref
            io.resume_payload = ckpt.read(ref)

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

            if ckpt is not None:
                if io.latest_ckpt is None:
                    io.latest_ckpt = ckpt.find_latest()
                if io.best_ckpt is None:
                    io.best_ckpt = ckpt.find_best()

            if self._record_events:
                self._bus.flush()

            if self._store_artifacts and self._record_events:
                events_path = os.path.join(run_dir, "events.jsonl")
                ref = store.put(events_path, kind="log", name="events", meta={})
                io.artifacts.append(ref)

            checkpoints_out = {
                "latest": asdict(io.latest_ckpt) if io.latest_ckpt is not None else None,
                "best": asdict(io.best_ckpt) if io.best_ckpt is not None else None,
                "resume_from": asdict(io.resume_ref) if io.resume_ref is not None else None,
            }

            m = RunManifest(
                run_id=rcfg.run_id,
                status=status,
                started_ts=started,
                finished_ts=finished,
                config_hash=rcfg.config_hash,
                inputs={"resume": checkpoints_out["resume_from"]},
                outputs={
                    "artifacts": [asdict(a) for a in io.artifacts],
                    "checkpoints": checkpoints_out,
                },
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
            artifacts=tuple(io.artifacts),
            manifest_path=manifest_path,
            context={"run_dir": run_dir, "config_hash": rcfg.config_hash},
        )
