from __future__ import annotations

from dataclasses import dataclass

from ..typing.json_types import Context


@dataclass(frozen=True, slots=True)
class StopRun(Exception):
    reason: str
    context: Context


@dataclass(frozen=True, slots=True)
class EarlyStop(StopRun): ...


@dataclass(frozen=True, slots=True)
class Cancelled(StopRun): ...
