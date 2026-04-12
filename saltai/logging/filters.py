from __future__ import annotations

from collections.abc import Callable, Mapping

from saltai.utils.typing.core import Logger

EventFilter = Callable[[object], bool]


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
