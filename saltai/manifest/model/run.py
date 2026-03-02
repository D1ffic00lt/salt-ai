from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True, slots=True)
class RunManifest(object):
    run_id: str
    status: str
    started_ts: float
    finished_ts: float
    config_hash: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    metrics: dict[str, Any]
    error: dict[str, Any] | None
    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
