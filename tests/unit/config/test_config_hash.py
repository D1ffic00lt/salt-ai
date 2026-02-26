import unittest

from saltai.config.validation.validate import validate_config


class TestConfigHash(unittest.TestCase):
    def test_config_hash_stable(self):
        a = validate_config({"run": {"id": "r1"}, "seed": 42, "paths": {"root": "runs"}})
        b = validate_config({"paths": {"root": "runs"}, "seed": 42, "run": {"id": "r1"}})
        self.assertEqual(a.config_hash, b.config_hash)