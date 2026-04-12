from __future__ import annotations

from typing import Final, Callable, Mapping

from saltai.utils.typing.core import Logger
from saltai.utils.typing.events import EventType

EventFilter = Callable[[object], bool]


def _event_types(*events: EventType) -> frozenset[EventType]:
    return frozenset(events)


CORE_LIFECYCLE_EVENTS: Final[frozenset[EventType]] = _event_types(
    "run_started",
    "run_finished",
    "run_failed",
    "stage_started",
    "stage_finished",
)

PROGRESS_EVENTS: Final[frozenset[EventType]] = _event_types(
    "epoch_started",
    "epoch_finished",
    "step_started",
    "step_finished",
)

OUTPUT_EVENTS: Final[frozenset[EventType]] = _event_types(
    "metric",
    "artifact_saved",
    "checkpoint_saved",
)

CLEARML_DEFAULT_EVENTS: Final[frozenset[EventType]] = _event_types(
    "metric",
    "artifact_saved",
    "checkpoint_saved",
)


def _event_type(event: object) -> str | None:
    if isinstance(event, Mapping):
        value = event.get("type")
    else:
        value = getattr(event, "type", None)

    if isinstance(value, str):
        return value

    return None


class EventTypeFilter(object):
    def __init__(
            self,
            *,
            include: set[str] | frozenset[str] | list[str] | tuple[str, ...] | None = None,
            exclude: set[str] | frozenset[str] | list[str] | tuple[str, ...] | None = None,
    ):
        self.include = frozenset(include) if include is not None else None
        self.exclude = frozenset(exclude or ())

    def __call__(self, event: object) -> bool:
        event_type = _event_type(event)

        if self.include is not None and event_type not in self.include:
            return False

        if event_type in self.exclude:
            return False

        return True


class FilteredLogger(Logger):
    def __init__(self, sink: Logger, event_filter: EventFilter):
        self.sink = sink
        self.event_filter = event_filter

    def log(self, event: object) -> None:
        if self.event_filter(event):
            self.sink.log(event)

    def flush(self) -> None:
        self.sink.flush()

    def close(self) -> None:
        self.sink.close()
