import unittest


class TestPublicImports(unittest.TestCase):
    def test_core_public_imports(self):
        from saltai import Runner, Trainer

        self.assertIsNotNone(Runner)
        self.assertIsNotNone(Trainer)

    def test_clearml_public_import_path(self):
        from saltai.integrations.clearml import ClearMLLogger

        self.assertIsNotNone(ClearMLLogger)


if __name__ == "__main__":
    unittest.main()
