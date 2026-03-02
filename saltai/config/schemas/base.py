from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResolvedConfig(object):
    run_id: str
    seed: int
    root: str
    config_hash: str
