import builtins
import hashlib
import os
import tempfile
import unittest
from unittest.mock import patch

from saltai.integrations.s3 import S3ArtifactStore
from saltai.integrations.s3.store import Boto3NotInstalledError, _load_boto3_client
from saltai.utils.typing.core import ArtifactId, ArtifactRef


class FakeS3Client(object):
    def __init__(self, *, page_size=100):
        self.objects = {}
        self.uploads = []
        self.downloads = []
        self.heads = []
        self.page_size = page_size

    def upload_file(self, local_path, bucket, key):
        with open(local_path, "rb") as f:
            self.objects[(bucket, key)] = f.read()
        self.uploads.append((local_path, bucket, key))

    def download_file(self, bucket, key, dst):
        self.downloads.append((bucket, key, dst))
        data = self.objects[(bucket, key)]
        with open(dst, "wb") as f:
            f.write(data)

    def head_object(self, *, Bucket, Key):
        self.heads.append((Bucket, Key))
        if (Bucket, Key) not in self.objects:
            raise RuntimeError("not found")
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def list_objects_v2(self, *, Bucket, Prefix, ContinuationToken=None):
        keys = sorted(
            key
            for bucket, key in self.objects
            if bucket == Bucket and key.startswith(Prefix)
        )

        start = int(ContinuationToken or 0)
        page = keys[start:start + self.page_size]

        response = {
            "Contents": [
                {
                    "Key": key,
                    "Size": len(self.objects[(Bucket, key)]),
                }
                for key in page
            ]
        }

        next_start = start + self.page_size
        if next_start < len(keys):
            response["NextContinuationToken"] = str(next_start)

        return response


class TestS3ArtifactStore(unittest.TestCase):
    def test_put_returns_correct_s3_ref(self):
        fake = FakeS3Client()

        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "model.bin")
            with open(src, "wb") as f:
                f.write(b"artifact")

            store = S3ArtifactStore(bucket="salt-bucket", prefix="runs/r1", client=fake)
            ref = store.put(src, kind="model", name="weights", meta={"format": "bin"})

            self.assertEqual(ref.kind, "model")
            self.assertEqual(ref.name, "weights")
            self.assertRegex(ref.uri, r"^s3://salt-bucket/runs/r1/model/weights__[0-9a-f]{32}\.bin$")
            self.assertEqual(ref.sha256, hashlib.sha256(b"artifact").hexdigest())
            self.assertEqual(ref.size_bytes, len(b"artifact"))
            self.assertEqual(ref.meta, {"format": "bin"})

            key = ref.uri.replace("s3://salt-bucket/", "")
            self.assertEqual(fake.objects[("salt-bucket", key)], b"artifact")
            self.assertEqual(fake.uploads, [(src, "salt-bucket", key)])

    def test_get_downloads_file(self):
        fake = FakeS3Client()
        fake.objects[("salt-bucket", "runs/r1/model/weights__a1.bin")] = b"artifact"

        ref = ArtifactRef(
            id=ArtifactId("a1"),
            kind="model",
            name="weights",
            uri="s3://salt-bucket/runs/r1/model/weights__a1.bin",
            sha256=None,
            size_bytes=None,
            meta={},
        )

        with tempfile.TemporaryDirectory() as d:
            store = S3ArtifactStore(bucket="salt-bucket", prefix="runs/r1", client=fake)
            dst = store.get(ref, dst_dir=d)

            self.assertEqual(os.path.basename(dst), "weights__a1.bin")
            with open(dst, "rb") as f:
                self.assertEqual(f.read(), b"artifact")

            self.assertEqual(fake.downloads, [("salt-bucket", "runs/r1/model/weights__a1.bin", dst)])

    def test_exists_true_false(self):
        fake = FakeS3Client()
        fake.objects[("salt-bucket", "runs/r1/model/weights__a1.bin")] = b"artifact"

        existing = ArtifactRef(
            id=ArtifactId("a1"),
            kind="model",
            name="weights",
            uri="s3://salt-bucket/runs/r1/model/weights__a1.bin",
            sha256=None,
            size_bytes=None,
            meta={},
        )
        missing = ArtifactRef(
            id=ArtifactId("a2"),
            kind="model",
            name="missing",
            uri="s3://salt-bucket/runs/r1/model/missing__a2.bin",
            sha256=None,
            size_bytes=None,
            meta={},
        )

        store = S3ArtifactStore(bucket="salt-bucket", prefix="runs/r1", client=fake)

        self.assertTrue(store.exists(existing))
        self.assertFalse(store.exists(missing))

    def test_list_returns_refs_from_prefix(self):
        fake = FakeS3Client(page_size=1)
        fake.objects[("salt-bucket", "runs/r1/checkpoint/latest__c1.pkl")] = b"checkpoint"
        fake.objects[("salt-bucket", "runs/r1/model/weights__a1.bin")] = b"artifact"
        fake.objects[("salt-bucket", "runs/r2/model/other__a2.bin")] = b"other"

        store = S3ArtifactStore(bucket="salt-bucket", prefix="runs/r1", client=fake)

        refs = store.list()
        self.assertEqual(len(refs), 2)

        by_uri = {ref.uri: ref for ref in refs}

        checkpoint_ref = by_uri["s3://salt-bucket/runs/r1/checkpoint/latest__c1.pkl"]
        self.assertEqual(checkpoint_ref.id, ArtifactId(""))
        self.assertEqual(checkpoint_ref.kind, "checkpoint")
        self.assertEqual(checkpoint_ref.name, "latest")
        self.assertIsNone(checkpoint_ref.sha256)
        self.assertEqual(checkpoint_ref.size_bytes, len(b"checkpoint"))
        self.assertEqual(checkpoint_ref.meta, {})

        model_ref = by_uri["s3://salt-bucket/runs/r1/model/weights__a1.bin"]
        self.assertEqual(model_ref.id, ArtifactId(""))
        self.assertEqual(model_ref.kind, "model")
        self.assertEqual(model_ref.name, "weights")
        self.assertIsNone(model_ref.sha256)
        self.assertEqual(model_ref.size_bytes, len(b"artifact"))
        self.assertEqual(model_ref.meta, {})

        model_refs = store.list(kind="model")
        self.assertEqual(len(model_refs), 1)
        self.assertEqual(model_refs[0].uri, "s3://salt-bucket/runs/r1/model/weights__a1.bin")

    def test_missing_boto3_has_helpful_error(self):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "boto3":
                raise ImportError("No module named boto3")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(Boto3NotInstalledError, "salt-ai\\[s3\\]"):
                _load_boto3_client()


if __name__ == "__main__":
    unittest.main()
