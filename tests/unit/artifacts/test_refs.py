import unittest

from saltai.artifacts.refs import (
    artifact_local_path,
    artifact_ref_scheme,
    artifact_uri_scheme,
    is_local_artifact_ref,
    make_artifact_ref,
    validate_artifact_key,
    validate_artifact_ref,
)
from saltai.utils.errors.base import ArtifactError
from saltai.utils.errors.codes import EC
from saltai.utils.typing.core import ArtifactId, ArtifactRef


class TestArtifactRefs(unittest.TestCase):
    def test_make_artifact_ref_defaults_meta(self):
        ref = make_artifact_ref(
            id="a1",
            kind="model",
            name="best",
            uri="s3://bucket/path/model.pkl",
        )

        self.assertEqual(ref.id, ArtifactId("a1"))
        self.assertEqual(ref.kind, "model")
        self.assertEqual(ref.name, "best")
        self.assertEqual(ref.meta, {})
        self.assertEqual(artifact_ref_scheme(ref), "s3")

    def test_uri_scheme(self):
        self.assertEqual(artifact_uri_scheme("file:///tmp/x.txt"), "file")
        self.assertEqual(artifact_uri_scheme("s3://bucket/key"), "s3")
        self.assertEqual(artifact_uri_scheme("/tmp/x.txt"), "")

    def test_local_path(self):
        ref = make_artifact_ref(
            id="a1",
            kind="log",
            name="events",
            uri="file:///tmp/events.jsonl",
        )

        self.assertTrue(is_local_artifact_ref(ref))
        self.assertEqual(artifact_local_path(ref), "/tmp/events.jsonl")

    def test_validate_ref_rejects_missing_scheme(self):
        ref = ArtifactRef(
            id=ArtifactId("a1"),
            kind="model",
            name="best",
            uri="/tmp/model.pkl",
            sha256=None,
            size_bytes=None,
            meta={},
        )

        with self.assertRaises(ArtifactError) as cm:
            validate_artifact_ref(ref)

        self.assertEqual(cm.exception.code, EC.ARTIFACT_INVALID_REF)

    def test_validate_ref_rejects_wrong_scheme(self):
        ref = make_artifact_ref(
            id="a1",
            kind="model",
            name="best",
            uri="s3://bucket/model.pkl",
        )

        with self.assertRaises(ArtifactError) as cm:
            validate_artifact_ref(ref, allowed_schemes=("file",))

        self.assertEqual(cm.exception.code, EC.ARTIFACT_INVALID_REF)

    def test_validate_ref_rejects_bad_sha(self):
        ref = ArtifactRef(
            id=ArtifactId("a1"),
            kind="model",
            name="best",
            uri="file:///tmp/model.pkl",
            sha256="bad",
            size_bytes=None,
            meta={},
        )

        with self.assertRaises(ArtifactError) as cm:
            validate_artifact_ref(ref)

        self.assertEqual(cm.exception.code, EC.ARTIFACT_INVALID_REF)

    def test_validate_key(self):
        self.assertEqual(validate_artifact_key(kind="model", name="best"), ("model", "best"))

        with self.assertRaises(ArtifactError):
            validate_artifact_key(kind="../model", name="best")

        with self.assertRaises(ArtifactError):
            validate_artifact_key(kind="model", name="nested/best")
