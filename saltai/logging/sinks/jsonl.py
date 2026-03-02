from __future__ import annotations

import json
from pathlib import Path

from saltai.logging.utils.jsonable import to_jsonable


class JsonlLogger(object):
    def __init__(self, path: str, *, flush_each: bool = False):
        self.path = str(path)
        self.flush_each = flush_each
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")

    def log(self, event: object) -> None:
        payload = to_jsonable(event)
        self._f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        if self.flush_each:
            self._f.flush()

    def flush(self) -> None:
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.flush()
        finally:
            self._f.close()
