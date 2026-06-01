import json
import os
import tempfile

from saltai.engine.event_bus.bus import EventBus
from saltai.engine.runner.runner import Runner


class PrintSink(object):
    def log(self, event):
        print(type(event).__name__, event)

    def flush(self):
        return None

    def close(self):
        return None


def main():
    with tempfile.TemporaryDirectory() as root:
        run_id = "demo1"
        run_dir = os.path.join(root, run_id)

        bus = EventBus([PrintSink()])

        r = Runner(
            event_bus=bus,
            record_events=True,
            store_artifacts=True,
        )

        def body(ctx):
            p = os.path.join(ctx.run_dir, "hello.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write("hello from saltai\n")

        res = r.run(
            {"run": {"id": run_id}, "seed": 42, "paths": {"root": root}},
            body=body,
        )

        print("\n=== RUN RESULT ===")
        print("status:", res.status)
        print("run_dir:", run_dir)
        print("manifest:", res.manifest_path)

        with open(res.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        print("\n=== MANIFEST (short) ===")
        print("run_id:", manifest["run_id"])
        print("status:", manifest["status"])
        print("config_hash:", manifest["config_hash"])
        print("artifacts:", len(manifest["outputs"]["artifacts"]))

        events_path = os.path.join(run_dir, "events.jsonl")
        if os.path.exists(events_path):
            print("\n=== EVENTS.JSONL (first 5 lines) ===")
            with open(events_path, "r", encoding="utf-8") as f:
                for i, line in zip(range(5), f):
                    print(line.rstrip())

        art_root = os.path.join(run_dir, "artifacts")
        if os.path.exists(art_root):
            print("\n=== ARTIFACTS DIR TREE ===")
            for dp, _, fns in os.walk(art_root):
                for fn in fns:
                    print(os.path.join(dp, fn))


if __name__ == "__main__":
    main()
