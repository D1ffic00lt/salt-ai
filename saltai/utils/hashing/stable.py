from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_json_dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()
