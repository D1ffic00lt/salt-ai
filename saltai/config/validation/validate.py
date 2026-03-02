from __future__ import annotations

from saltai.config.schemas.base import ResolvedConfig
from saltai.utils.errors.base import ConfigError
from saltai.utils.errors.codes import EC
from saltai.utils.hashing.stable import sha256_text, stable_json_dumps


def _missing(msg: str, *, ctx: dict) -> None:
    raise ConfigError(
        EC.CONFIG_MISSING_FIELD,
        msg,
        hint="Add the missing field to config",
        context=ctx,
    )


def _bad_type(msg: str, *, ctx: dict) -> None:
    raise ConfigError(
        EC.CONFIG_INVALID_TYPE,
        msg,
        hint="Fix the field type",
        context=ctx,
    )


def validate_config(cfg: dict) -> ResolvedConfig:
    if not isinstance(cfg, dict):
        _bad_type("Config must be a dict", ctx={"got": type(cfg).__name__})

    run = cfg.get("run")
    if run is None:
        _missing("Missing field: run", ctx={"field": "run"})
    if not isinstance(run, dict):
        _bad_type("Field run must be a dict", ctx={"field": "run", "got": type(run).__name__})

    run_id = run.get("id")
    if run_id is None:
        _missing("Missing field: run.id", ctx={"field": "run.id"})
    if not isinstance(run_id, str):
        _bad_type("Field run.id must be str", ctx={"field": "run.id", "got": type(run_id).__name__})

    seed = cfg.get("seed")
    if seed is None:
        _missing("Missing field: seed", ctx={"field": "seed"})
    if not isinstance(seed, int):
        _bad_type("Field seed must be int", ctx={"field": "seed", "got": type(seed).__name__})

    paths = cfg.get("paths")
    if paths is None:
        _missing("Missing field: paths", ctx={"field": "paths"})
    if not isinstance(paths, dict):
        _bad_type("Field paths must be a dict", ctx={"field": "paths", "got": type(paths).__name__})

    root = paths.get("root")
    if root is None:
        _missing("Missing field: paths.root", ctx={"field": "paths.root"})
    if not isinstance(root, str):
        _bad_type("Field paths.root must be str", ctx={"field": "paths.root", "got": type(root).__name__})

    canonical = stable_json_dumps(cfg)
    config_hash = sha256_text(canonical)

    return ResolvedConfig(run_id=run_id, seed=seed, root=root, config_hash=config_hash)
