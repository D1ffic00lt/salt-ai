from __future__ import annotations

from collections.abc import Callable
from typing import cast

from saltai.utils.typing.core import Logger


class BaseLogger(Logger):
    def log(self, event: object) -> None:
        event_type = getattr(event, "type", None)
        if isinstance(event_type, str):
            handler = getattr(self, f"on_{event_type}", None)
            if callable(handler):
                cast(Callable[[object], None], handler)(event)
                return

        self.on_event(event)

    def on_event(self, event: object) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class NoOpLogger(BaseLogger):
    pass
