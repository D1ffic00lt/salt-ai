from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, NewType, Protocol, Sequence, TypeVar, runtime_checkable

from .json_types import Context, JSONObject, PathLike

__all__ = (
    "RunId",
    "ArtifactId",
    "Split",
    "StageName",
    "Scalar",
    "Step",
    "Epoch",
    "BatchT",
    "PredT",
    "LossT",
    "StateT",
    "ArtifactRef",
    "Checkpointable",
    "DataModule",
    "ModelAdapter",
    "Logger",
    "Metric",
    "MetricPoint",
    "MetricSummary",
    "RunResult",
    "ArtifactStore"
)

RunId = NewType("RunId", str)
ArtifactId = NewType("ArtifactId", str)

Split = str
StageName = str

Scalar = float | int
Step = int
Epoch = int

BatchT = TypeVar("BatchT")
PredT = TypeVar("PredT")
LossT = TypeVar("LossT")
StateT = TypeVar("StateT")


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    id: ArtifactId
    kind: str
    name: str
    uri: str
    sha256: str | None
    size_bytes: int | None
    meta: JSONObject


@runtime_checkable
class Checkpointable(Protocol):
    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


@runtime_checkable
class DataModule(Protocol[BatchT]):
    def prepare(self) -> None: ...

    def train_iter(self) -> Iterable[BatchT]: ...

    def val_iter(self) -> Iterable[BatchT] | None: ...

    def test_iter(self) -> Iterable[BatchT] | None: ...


@runtime_checkable
class ModelAdapter(Protocol[BatchT, PredT, LossT]):
    def forward(self, batch: BatchT) -> PredT: ...

    def loss(self, pred: PredT, batch: BatchT) -> LossT: ...

    def zero_grad(self) -> None: ...

    def backward(self, loss: LossT) -> None: ...

    def step(self) -> None: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


@runtime_checkable
class Metric(Protocol[BatchT, PredT]):
    name: str

    def reset(self) -> None: ...

    def update(self, batch: BatchT, pred: PredT) -> None: ...

    def compute(self) -> Scalar | Mapping[str, Scalar]: ...


@runtime_checkable
class Logger(Protocol):
    def log(self, event: object) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class ArtifactStore(Protocol):
    def put(self, local_path: PathLike, *, kind: str, name: str, meta: JSONObject | None = None) -> ArtifactRef: ...

    def get(self, ref: ArtifactRef, *, dst_dir: PathLike) -> PathLike: ...

    def exists(self, ref: ArtifactRef) -> bool: ...

    def list(self, *, kind: str | None = None) -> Sequence[ArtifactRef]: ...


@runtime_checkable
class Fingerprinter(Protocol):
    def fingerprint_inputs(self) -> JSONObject: ...

    def fingerprint_code(self) -> JSONObject: ...


@dataclass(frozen=True, slots=True)
class MetricPoint:
    name: str
    value: Scalar
    step: Step | None
    epoch: Epoch | None
    split: Split | None
    extra: JSONObject


@dataclass(frozen=True, slots=True)
class MetricSummary:
    values: dict[str, Scalar]
    extra: JSONObject


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: RunId
    status: str
    metrics: MetricSummary
    artifacts: Sequence[ArtifactRef]
    manifest_path: PathLike
    context: Context
