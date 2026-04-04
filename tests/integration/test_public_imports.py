import builtins
import importlib
import sys
import unittest
from unittest.mock import patch


class TestPublicImports(unittest.TestCase):
    def test_core_public_imports(self):
        from saltai import Runner, Trainer

        self.assertEqual(Runner.__name__, "Runner")
        self.assertEqual(Trainer.__name__, "Trainer")

    def test_clearml_public_import_does_not_require_clearml(self):
        from saltai.integrations.clearml import ClearMLLogger

        self.assertEqual(ClearMLLogger.__name__, "ClearMLLogger")

    def test_s3_public_import_does_not_require_boto3(self):
        from saltai.integrations.s3 import S3ArtifactStore

        self.assertEqual(S3ArtifactStore.__name__, "S3ArtifactStore")

    def test_import_saltai_does_not_import_optional_dependencies(self):
        for module_name in list(sys.modules):
            if module_name == "saltai" or module_name.startswith("saltai."):
                del sys.modules[module_name]
            if module_name == "clearml" or module_name.startswith("clearml."):
                del sys.modules[module_name]
            if module_name == "boto3" or module_name.startswith("boto3."):
                del sys.modules[module_name]

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name in {"clearml", "boto3"}:
                raise AssertionError(f"optional dependency imported during core import: {name}")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            saltai = importlib.import_module("saltai")

        self.assertTrue(hasattr(saltai, "Runner"))
        self.assertTrue(hasattr(saltai, "Trainer"))


if __name__ == "__main__":
    unittest.main()
