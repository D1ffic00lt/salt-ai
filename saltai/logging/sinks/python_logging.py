from __future__ import annotations

import json
import logging

from saltai.logging.utils.jsonable import to_jsonable


class PythonLoggingSink(object):
    def __init__(self, name: str = "saltai.events", level: int = logging.INFO):
        self._log = logging.getLogger(name)
        self._level = level

    def log(self, event: object) -> None:
        payload = to_jsonable(event)
        msg = json.dumps(payload, ensure_ascii=False)
        self._log.log(self._level, msg)

    def flush(self) -> None:
        for h in self._log.handlers:
            try:
                h.flush()
            except Exception as e:
                self._log.exception(f"Failed to flush {h}, error: {e}")

    def close(self) -> None:
        return
