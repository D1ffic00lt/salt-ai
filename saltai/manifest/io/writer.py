from __future__ import annotations

import json
import os
from pathlib import Path

from saltai.manifest.model.run import RunManifest


def write_manifest_atomic(m: RunManifest, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    tmp = p.with_suffix(p.suffix + ".tmp")
    data = m.to_dict()

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    os.replace(tmp, p)