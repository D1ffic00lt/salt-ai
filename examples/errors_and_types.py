from __future__ import annotations

from saltai.utils.errors.base import ConfigError
from saltai.utils.errors.codes import EC
from saltai.utils.errors.helpers import ensure, guard, wrap_unknown
from saltai.utils.errors.signals import Cancelled, EarlyStop
from saltai.utils.typing.core import ArtifactId, ArtifactRef, MetricPoint, MetricSummary, RunId, RunResult
from saltai.utils.typing.events import MetricLogged


def main() -> None:
    ensure(
        cond=True,
        code=EC.CONFIG_CONSTRAINT,
        message="This should not fail",
        exc_type=ConfigError,
    )

    print("guard:", guard("division", lambda: 10 / 2))

    try:
        int("oops")
    except Exception as e:
        wrapped = wrap_unknown(e, context={"module": "examples/errors"})
        print("wrapped:", wrapped.to_info())

    point = MetricPoint(name="loss", value=0.25, step=3, epoch=0, split="train", extra={})
    event = MetricLogged(type="metric", run_id=RunId("r1"), ts=0.0, data={}, point=point)
    print("event:", event)

    result = RunResult(
        run_id=RunId("r1"),
        status="success",
        metrics=MetricSummary(values={"loss": 0.25}, extra={}),
        artifacts=(
            ArtifactRef(
                id=ArtifactId("id1"),
                kind="log",
                name="events",
                uri="file:///tmp/events.jsonl",
                sha256=None,
                size_bytes=None,
                meta={},
            ),
        ),
        manifest_path="/tmp/manifest.json",
        context={"source": "example"},
    )
    print("result:", result)

    # just instantiate stop signals to demonstrate usage
    print(EarlyStop(reason="metric plateau", context={"epoch": 4}))
    print(Cancelled(reason="user requested", context={"source": "cli"}))


if __name__ == "__main__":
    main()
