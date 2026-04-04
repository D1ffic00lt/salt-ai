import os
import tempfile
import unittest
from unittest.mock import patch

from saltai.integrations.s3 import Boto3NotInstalledError, S3ArtifactStore
from saltai.utils.typing.core import ArtifactId, ArtifactRef


class FakeS3Client(object):
    def __init__(self):
        self.objects = {}
        self.uploads = []
        self.downloads = []
        self.heads = []
        self.lists = []

    def upload_file(self, filename, bucket, key):
        self.uploads.append((filename, bucket, key))
        with open(filename, "rb") as f:
            self.objects[(bucket, key)] = f.read()

    def download_file(self, bucket, key, filename):
        self.downloads.append((bucket, key, filename))
        with open(filename, "wb") as f:
            f.write(self.objects[(bucket, key)])

    def head_object(self, *, Bucket, Key):
        self.heads.append((Bucket, Key))
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def list_objects_v2(self, **kwargs):
        self.lists.append(dict(kwargs))
        bucket = kwargs["Bucket"]
        prefix = kwargs.get("Prefix", "")

        contents = []
        for (obj_bucket, key), data in self.objects.items():
            if obj_bucket == bucket and key.startswith(prefix):
                contents.append({"Key": key, "Size": len(data)})

        return {
            "IsTruncated": False,
            "Contents": contents,
        }


class TestS3ArtifactStore(unittest.TestCase):
    def test_put_uploads_file_and_returns_s3_ref(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "model.txt")
            with open(src, "w", encoding="utf-8") as f:
                f.write("model")

            client = FakeS3Client()
            store = S3ArtifactStore(bucket="saltai", prefix="runs/r1", client=client)

            ref = store.put(src, kind="model", name="tiny-model", meta={"format": "txt"})

            self.assertEqual(ref.kind, "model")
            self.assertEqual(ref.name, "tiny-model")
            self.assertTrue(ref.uri.startswith("s3://saltai/runs/r1/model/tiny-model__"))
            self.assertTrue(ref.uri.endswith(".txt"))
            self.assertEqual(ref.size_bytes, 5)
            self.assertIsNotNone(ref.sha256)
            self.assertEqual(ref.meta, {"format": "txt"})

            self.assertEqual(len(client.uploads), 1)
            self.assertEqual(client.uploads[0][0], src)
            self.assertEqual(client.uploads[0][1], "saltai")

    def test_get_downloads_file(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "model.txt")
            with open(src, "w", encoding="utf-8") as f:
                f.write("model")

            client = FakeS3Client()
            store = S3ArtifactStore(bucket="saltai", prefix="runs/r1", client=client)

            ref = store.put(src, kind="model", name="tiny-model")
            dst = store.get(ref, dst_dir=os.path.join(d, "downloads"))

            self.assertTrue(os.path.exists(dst))
            with open(dst, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "model")

            self.assertEqual(len(client.downloads), 1)

    def test_exists_returns_true_for_existing_object(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "model.txt")
            with open(src, "w", encoding="utf-8") as f:
                f.write("model")

            client = FakeS3Client()
            store = S3ArtifactStore(bucket="saltai", prefix="runs/r1", client=client)

            ref = store.put(src, kind="model", name="tiny-model")

            self.assertTrue(store.exists(ref))

    def test_exists_returns_false_for_missing_object(self):
        client = FakeS3Client()
        store = S3ArtifactStore(bucket="saltai", prefix="runs/r1", client=client)

        ref = ArtifactRef(
            id=ArtifactId("missing"),
            kind="model",
            name="missing",
            uri="s3://saltai/runs/r1/model/missing.txt",
            sha256=None,
            size_bytes=None,
            meta={},
        )

        self.assertFalse(store.exists(ref))

    def test_list_returns_refs_from_s3_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "model.txt")
            with open(src, "w", encoding="utf-8") as f:
                f.write("model")

            client = FakeS3Client()
            store = S3ArtifactStore(bucket="saltai", prefix="runs/r1", client=client)

            _ = store.put(src, kind="model", name="tiny-model")
            refs = store.list(kind="model")

            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0].kind, "model")
            self.assertEqual(refs[0].name, "tiny-model")
            self.assertTrue(refs[0].uri.startswith("s3://saltai/runs/r1/model/tiny-model__"))

    def test_missing_boto3_has_helpful_error(self):
        with patch(
                "saltai.integrations.s3.store._load_boto3_client",
                side_effect=Boto3NotInstalledError(
                    "boto3 is not installed. Install it with `pip install salt-ai[s3]`."
                ),
        ):
            with self.assertRaisesRegex(Boto3NotInstalledError, "salt-ai\\[s3\\]"):
                S3ArtifactStore(bucket="saltai")


if __name__ == "__main__":
    unittest.main()
