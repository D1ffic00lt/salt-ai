import os
import tempfile
import unittest

from saltai.artifacts.store.s3 import S3ArtifactStore
from saltai.utils.errors.base import ArtifactError
from saltai.utils.errors.codes import EC


class FakeS3Client:
    def __init__(self):
        self.objects = {}

    def upload_file(self, filename, bucket, key):
        with open(filename, "rb") as f:
            self.objects[(bucket, key)] = f.read()

    def download_file(self, bucket, key, filename):
        data = self.objects[(bucket, key)]
        with open(filename, "wb") as f:
            f.write(data)

    def head_object(self, *, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise KeyError((Bucket, Key))
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def list_objects_v2(self, **kwargs):
        bucket = kwargs["Bucket"]
        prefix = kwargs.get("Prefix", "")

        contents = []
        for (obj_bucket, key), data in sorted(self.objects.items()):
            if obj_bucket != bucket:
                continue
            if not key.startswith(prefix):
                continue
            contents.append({"Key": key, "Size": len(data)})

        return {
            "IsTruncated": False,
            "Contents": contents,
        }


class TestS3ArtifactStore(unittest.TestCase):
    def test_put_get_exists(self):
        with tempfile.TemporaryDirectory() as d:
            client = FakeS3Client()
            store = S3ArtifactStore(bucket="bucket", prefix="runs/r1", client=client)

            src = os.path.join(d, "model.txt")
            with open(src, "w", encoding="utf-8") as f:
                f.write("weights")

            ref = store.put(src, kind="model", name="best", meta={"epoch": 2})

            self.assertEqual(ref.kind, "model")
            self.assertEqual(ref.name, "best")
            self.assertTrue(ref.uri.startswith("s3://bucket/runs/r1/model/best__"))
            self.assertEqual(ref.size_bytes, len("weights"))
            self.assertEqual(ref.meta, {"epoch": 2})
            self.assertTrue(store.exists(ref))

            out_dir = os.path.join(d, "out")
            dst = store.get(ref, dst_dir=out_dir)

            self.assertTrue(os.path.exists(dst))
            with open(dst, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "weights")

    def test_list_all_and_by_kind(self):
        with tempfile.TemporaryDirectory() as d:
            client = FakeS3Client()
            store = S3ArtifactStore(bucket="bucket", prefix="runs/r1", client=client)

            for i in range(2):
                src = os.path.join(d, f"m{i}.txt")
                with open(src, "w", encoding="utf-8") as f:
                    f.write(str(i))
                store.put(src, kind="model", name=f"m{i}")

            src = os.path.join(d, "events.jsonl")
            with open(src, "w", encoding="utf-8") as f:
                f.write("{}")
            store.put(src, kind="log", name="events")

            models = store.list(kind="model")
            logs = store.list(kind="log")
            all_ = store.list()

            self.assertEqual(len(models), 2)
            self.assertEqual(len(logs), 1)
            self.assertEqual(len(all_), 3)

            self.assertEqual({r.kind for r in models}, {"model"})
            self.assertEqual(logs[0].kind, "log")
            self.assertEqual(logs[0].name, "events")

    def test_missing_local_file(self):
        client = FakeS3Client()
        store = S3ArtifactStore(bucket="bucket", client=client)

        with self.assertRaises(ArtifactError) as cm:
            store.put("/no/such/file.txt", kind="model", name="x")

        self.assertEqual(cm.exception.code, EC.ARTIFACT_NOT_FOUND)

    def test_rejects_unsafe_key(self):
        with tempfile.TemporaryDirectory() as d:
            client = FakeS3Client()
            store = S3ArtifactStore(bucket="bucket", client=client)

            src = os.path.join(d, "x.txt")
            with open(src, "w", encoding="utf-8") as f:
                f.write("x")

            with self.assertRaises(ArtifactError) as cm:
                store.put(src, kind="model", name="../x")

            self.assertEqual(cm.exception.code, EC.ARTIFACT_INVALID_REF)

    def test_exists_false_for_missing_object(self):
        with tempfile.TemporaryDirectory() as d:
            client = FakeS3Client()
            store = S3ArtifactStore(bucket="bucket", client=client)

            src = os.path.join(d, "x.txt")
            with open(src, "w", encoding="utf-8") as f:
                f.write("x")

            ref = store.put(src, kind="model", name="x")
            client.objects.clear()

            self.assertFalse(store.exists(ref))
