from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .core import (
    ArtifactRef,
    Epoch,
    MetricPoint,
    RunId, Split,
    StageName,
    Step
)
from .json_types import JSONObject

__all__ = (
    "EventType",
    "BaseEvent",
    "RunStarted",
    "RunFinished",
    "StageStarted",
    "StageFinished",
    "EpochStarted",
    "EpochFinished",
    "StepStarted",
    "StepFinished",
    "MetricLogged",
    "ArtifactSaved",
    "CheckpointSaved",
    "WarningRaised",
)

EventType = Literal[
    "run_started",
    "run_finished",
    "stage_started",
    "stage_finished",
    "epoch_started",
    "epoch_finished",
    "step_started",
    "step_finished",
    "metric",
    "artifact_saved",
    "checkpoint_saved",
    "warning",
]


@dataclass(frozen=True, slots=True)
class BaseEvent:
    type: EventType
    run_id: RunId
    ts: float
    data: JSONObject


@dataclass(frozen=True, slots=True)
class RunStarted(BaseEvent):
    type: Literal["run_started"]


@dataclass(frozen=True, slots=True)
class RunFinished(BaseEvent):
    type: Literal["run_finished"]


@dataclass(frozen=True, slots=True)
class StageStarted(BaseEvent):
    type: Literal["stage_started"]
    stage: StageName


@dataclass(frozen=True, slots=True)
class StageFinished(BaseEvent):
    type: Literal["stage_finished"]
    stage: StageName


@dataclass(frozen=True, slots=True)
class EpochStarted(BaseEvent):
    type: Literal["epoch_started"]
    epoch: Epoch
    split: Split | None


@dataclass(frozen=True, slots=True)
class EpochFinished(BaseEvent):
    type: Literal["epoch_finished"]
    epoch: Epoch
    split: Split | None


@dataclass(frozen=True, slots=True)
class StepStarted(BaseEvent):
    type: Literal["step_started"]
    step: Step
    epoch: Epoch | None
    split: Split | None


@dataclass(frozen=True, slots=True)
class StepFinished(BaseEvent):
    type: Literal["step_finished"]
    step: Step
    epoch: Epoch | None
    split: Split | None


@dataclass(frozen=True, slots=True)
class MetricLogged(BaseEvent):
    type: Literal["metric"]
    point: MetricPoint


@dataclass(frozen=True, slots=True)
class ArtifactSaved(BaseEvent):
    type: Literal["artifact_saved"]
    ref: ArtifactRef


@dataclass(frozen=True, slots=True)
class CheckpointSaved(BaseEvent):
    type: Literal["checkpoint_saved"]
    ref: ArtifactRef
    tag: str


@dataclass(frozen=True, slots=True)
class WarningRaised(BaseEvent):
    type: Literal["warning"]
    code: str
    message: str
