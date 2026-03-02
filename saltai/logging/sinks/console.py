from __future__ import annotations

from saltai.logging.utils.jsonable import to_jsonable


class ConsoleLogger(object):
    def __init__(self, *, pretty: bool = False):
        self.pretty = pretty

    def log(self, event: object) -> None:
        x = to_jsonable(event)
        if self.pretty:
            print(x)
        else:
            t = type(event).__name__
            print(f"{t}: {x}")

    def flush(self) -> None:
        return

    def close(self) -> None:
        return
