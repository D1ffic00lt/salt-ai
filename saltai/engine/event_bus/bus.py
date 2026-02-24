from __future__ import annotations

from typing import Sequence

from saltai.utils.errors import LoggerError
from saltai.utils.errors.codes import EC
from saltai.utils.typing.core import Logger
from saltai.utils.typing.json_types import Context


class EventBus(object):
    def __init__(self, sinks: Sequence[Logger] | None = None, *, fail_fast: bool = True):
        self._sinks = list(sinks or [])
        self._fail_fast = fail_fast

    def add_sink(self, sink: Logger) -> None:
        self._sinks.append(sink)

    def publish(self, event: object, *, context: Context | None = None) -> None:
        errs: list[BaseException] = []
        for sink in self._sinks:
            try:
                sink.log(event)
            except BaseException as e:
                errs.append(e)
                if self._fail_fast:
                    raise LoggerError(
                        EC.LOG_SINK_FAILED,
                        "Logger sink failed while processing event",
                        hint="Check sink configuration and file permissions",
                        context={**(context or {}), "sink": type(sink).__name__, "event_type": type(event).__name__},
                        cause=e,
                    ) from e
        if errs:
            raise LoggerError(
                EC.LOG_SINK_FAILED,
                "One or more logger sinks failed while processing event",
                hint="Check sink configuration and file permissions",
                context={**(context or {}), "failed_sinks": len(errs), "event_type": type(event).__name__},
                cause=errs[0],
            ) from errs[0]

    def flush(self) -> None:
        for sink in self._sinks:
            sink.flush()

    def close(self) -> None:
        for sink in self._sinks:
            sink.close()
