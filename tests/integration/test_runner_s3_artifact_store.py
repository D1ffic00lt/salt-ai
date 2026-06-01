import json
import os
import tempfile
import unittest

from saltai import Runner
from saltai.engine.event_bus.bus import EventBus
from saltai.integrations.s3 import S3ArtifactStore


class FakeS3Client(object):
    def __init__(self):
        self.objects = {}
        self.uploads = []

    def upload_file(self, local_path, bucket, key):
        with open(local_path, "rb") as f:
            self.objects[(bucket, key)] = f.read()
        self.uploads.append((local_path, bucket, key))

    def download_file(self, bucket, key, dst):
        with open(dst, "wb") as f:
            f.write(self.objects[(bucket, key)])

    def head_object(self, *, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise RuntimeError("not found")
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def list_objects_v2(self, *, Bucket, Prefix, ContinuationToken=None):
        contents = []
        for bucket, key in sorted(self.objects):
            if bucket == Bucket and key.startswith(Prefix):
                contents.append(
                    {
                        "Key": key,
                        "Size": len(self.objects[(bucket, key)]),
                    }
                )
        return {"Contents": contents}


class _Sink(object):
    def __init__(self):
        self.events = []

    def log(self, event):
        self.events.append(event)

    def flush(self):
        return None

    def close(self):
        return None


class TestRunnerS3ArtifactStore(unittest.TestCase):
    def test_runner_uses_s3_artifact_store_factory_for_runio_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            fake = FakeS3Client()
            created = []
            sink = _Sink()
            bus = EventBus([sink])

            src = os.path.join(d, "model.txt")
            with open(src, "w", encoding="utf-8") as f:
                f.write("model")

            def make_store(run_dir):
                created.append(run_dir)
                return S3ArtifactStore(
                    bucket="salt-bucket",
                    prefix=f"runs/{os.path.basename(run_dir)}",
                    client=fake,
                )

            r = Runner(event_bus=bus, artifact_store_factory=make_store)

            def body(ctx):
                ref = ctx.io.save_artifact(
                    src,
                    kind="model",
                    name="tiny-model",
                    meta={"format": "txt"},
                )
                return {"artifact_uri": ref.uri}

            res = r.run(
                {"run": {"id": "r_s3_store_factory"}, "seed": 42, "paths": {"root": d}},
                body=body,
            )

            self.assertEqual(res.status, "success")
            self.assertEqual(created, [os.path.join(d, "r_s3_store_factory")])

            self.assertEqual(len(res.artifacts), 1)
            ref = res.artifacts[0]

            self.assertEqual(ref.kind, "model")
            self.assertEqual(ref.name, "tiny-model")
            self.assertEqual(ref.meta, {"format": "txt"})
            self.assertIsNotNone(ref.sha256)
            self.assertEqual(ref.size_bytes, len(b"model"))
            self.assertTrue(
                ref.uri.startswith("s3://salt-bucket/runs/r_s3_store_factory/model/tiny-model__")
            )
            self.assertTrue(ref.uri.endswith(".txt"))

            self.assertEqual(res.metrics.values["artifact_uri"], ref.uri)

            key = ref.uri.replace("s3://salt-bucket/", "")
            self.assertEqual(fake.objects[("salt-bucket", key)], b"model")
            self.assertEqual(fake.uploads, [(src, "salt-bucket", key)])

            artifact_events = [e for e in sink.events if type(e).__name__ == "ArtifactSaved"]
            self.assertEqual(len(artifact_events), 1)
            self.assertEqual(artifact_events[0].ref.uri, ref.uri)
            self.assertEqual(artifact_events[0].ref.kind, "model")
            self.assertEqual(artifact_events[0].ref.name, "tiny-model")

            with open(res.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            self.assertEqual(manifest["outputs"]["artifacts"][0]["uri"], ref.uri)
            self.assertEqual(manifest["outputs"]["artifacts"][0]["kind"], "model")
            self.assertEqual(manifest["outputs"]["artifacts"][0]["name"], "tiny-model")
            self.assertEqual(manifest["metrics"]["artifact_uri"], ref.uri)


if __name__ == "__main__":
    unittest.main()
