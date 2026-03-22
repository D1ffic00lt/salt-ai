import json
import os
import tempfile

from saltai.engine.event_bus.bus import EventBus
from saltai.engine.runner.runner import Runner
from saltai.utils.errors.base import DataError
from saltai.utils.errors.codes import EC


class PrintSink(object):
    def log(self, event):
        print(type(event).__name__, event)

    def flush(self):
        return None

    def close(self):
        return None


def main():
    with tempfile.TemporaryDirectory() as root:
        run_id = "demo_domain_fail"
        run_dir = os.path.join(root, run_id)

        bus = EventBus([PrintSink()])
        r = Runner(event_bus=bus, record_events=True, store_artifacts=True)

        def body(ctx):
            raise DataError(
                EC.DATA_NOT_FOUND,
                "Dataset missing",
                hint="Put dataset into data/ or set correct path",
                context={"path": "data/train.csv"},
            )

        res = r.run({"run": {"id": run_id}, "seed": 42, "paths": {"root": root}}, body=body)

        print("\n=== RUN RESULT ===")
        print("status:", res.status)
        print("run_dir:", run_dir)
        print("manifest:", res.manifest_path)

        with open(res.manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)

        print("\n=== MANIFEST ERROR ===")
        print(m["error"])


if __name__ == "__main__":
    main()