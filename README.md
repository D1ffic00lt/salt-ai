<h1 align="center">SaltAI</h1>
<div align="center">
	<img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/D1ffic00lt/salt-ai">
	<img alt="GitHub code size" src="https://img.shields.io/github/languages/code-size/D1ffic00lt/salt-ai">
	<img alt="GitHub commits stats" src="https://img.shields.io/github/commit-activity/y/D1ffic00lt/salt-ai">
	<img alt="GitHub License" src="https://img.shields.io/github/license/D1ffic00lt/salt-ai">
	<a href="https://github.com/D1ffic00lt/salt-ai/actions/workflows/unit-tests.yml">
		<img alt="Tests" src="https://github.com/D1ffic00lt/salt-ai/actions/workflows/unit-tests.yml/badge.svg?branch=dev">
	</a>
</div>

<p align="center"><strong>SaltAI</strong> is a lightweight local-first ML pipeline library for reproducible experiments. It helps organize training runs, metrics, manifests, events, artifacts and checkpoints without forcing users to deploy a heavy MLOps platform.</p>

## About the library

SaltAI is designed to standardize the most common parts of ML and DL experiments.  
The library gives you a simple run lifecycle, where each experiment has its own configuration, output directory, manifest, event log and optional checkpoints.

The main workflow looks like this:

```text
config -> Runner -> Trainer -> metrics -> manifest -> events -> checkpoints -> resume
````

The project is focused on a **library-first** approach. You can use the core package locally, while cloud and external MLOps integrations can be added later as optional extensions.

### Core features

* `Runner`

  Runs the experiment body, creates the run directory, validates config, writes `manifest.json`, records events and collects returned metrics.

* `Trainer`

  Provides a minimal framework-agnostic training and evaluation loop. It works with user-defined model adapters, data modules and metrics.

* `Run manifest`

  Every run produces a `manifest.json` file with the run status, config hash, metrics, artifacts, checkpoints and resume information.

* `Events`

  Structured events can be written to `events.jsonl`. This includes run, stage, epoch, step, metric and checkpoint events.

* `Artifacts`

  Local artifacts can be stored through the artifact store. For example, SaltAI can store the event log as a run artifact.

* `Checkpoints`

  SaltAI supports checkpoint saving through:

  * `latest`
  * `best`

* `Resume`

  A run can be resumed from a saved checkpoint:

  ```python
  runner.run(cfg, body=body, resume_from="latest")
  ```

## Installation

### 1. Installation for local development

To install the library locally, clone the repository and install the package in editable mode:

```bash
git clone https://github.com/D1ffic00lt/salt-ai.git
cd salt-ai
git checkout dev
python -m pip install -e .
```

### 2. Requirements

SaltAI currently supports Python versions:

```text
>=3.10,<3.13
```

The core library has no heavy ML framework dependency. You can connect your own model, data module and metrics through small protocol-like interfaces.

## Quickstart

The minimal run can be written like this:

```python
from saltai import Runner


class Model(object):
    def __init__(self):
        self.w = 1.0

    def state_dict(self):
        return {"w": self.w}

    def load_state_dict(self, state):
        self.w = float(state["w"])


cfg = {
    "run": {"id": "quickstart"},
    "seed": 42,
    "paths": {"root": "runs"},
}

runner = Runner(
    record_events=True,
    store_artifacts=True,
    enable_checkpoints=True,
)


def body(ctx):
    model = Model()
    model.w = 2.0

    ctx.io.save_latest(model, step=1)

    return {
        "loss": 0.1,
        "accuracy": 1.0,
    }


result = runner.run(cfg, body=body)

print(result.status)
print(result.metrics.values)
print(result.manifest_path)
```

After running this script, SaltAI creates a directory like this:

```text
runs/
└── quickstart
    ├── manifest.json
    ├── events.jsonl
    ├── artifacts
    └── checkpoints
```

## Resume from checkpoint

SaltAI can restore checkpoint payload before running the experiment body.

```python
from saltai import Runner


class Model(object):
    def __init__(self):
        self.w = 0.0

    def state_dict(self):
        return {"w": self.w}

    def load_state_dict(self, state):
        self.w = float(state["w"])


cfg = {
    "run": {"id": "quickstart"},
    "seed": 42,
    "paths": {"root": "runs"},
}

runner = Runner(
    record_events=True,
    store_artifacts=True,
    enable_checkpoints=True,
)


def body(ctx):
    assert ctx.io.resume_ref is not None
    assert ctx.io.resume_payload is not None

    model = Model()
    model.load_state_dict(ctx.io.resume_payload["state"])

    return {
        "loaded": True,
        "w": model.w,
    }


result = runner.run(
    cfg,
    body=body,
    resume_from="latest",
)

print(result.metrics.values)
```

## Full local example

The repository contains a complete local example with a tiny model, data module, metrics, training, evaluation, checkpoints, manifest and events.

```bash
python examples/full_local_run.py
```

The example demonstrates:

* `Runner.run`
* `Trainer.fit`
* `Trainer.evaluate`
* metric collection
* `ctx.io.save_latest`
* `ctx.io.save_best`
* `manifest.json`
* `events.jsonl`
* checkpoint files

## Public API

The main objects are available from the top-level package:

```python
from saltai import (
    Runner,
    Trainer,
    RunContext,
    RunIO,
    RunId,
    RunResult,
    ArtifactRef,
    Checkpointable,
    DataModule,
    ModelAdapter,
    Metric,
    MetricPoint,
    MetricSummary,
    Logger,
)
```

## Running tests

The project uses `unittest`.

```bash
python -m unittest discover tests
```

If you use Poetry:

```bash
poetry install
poetry run python -m unittest discover -s tests -p "test*.py" -v
```
