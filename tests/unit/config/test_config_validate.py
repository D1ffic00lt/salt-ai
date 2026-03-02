import unittest

from saltai.utils.errors.base import ConfigError
from saltai.utils.errors.codes import EC

from saltai.config.validation.validate import validate_config
from saltai.config.schemas.base import ResolvedConfig


class TestConfigValidate(unittest.TestCase):
    def test_missing_field(self):
        with self.assertRaises(ConfigError) as cm:
            validate_config({"seed": 42})
        self.assertEqual(cm.exception.code, EC.CONFIG_MISSING_FIELD)

    def test_invalid_type(self):
        with self.assertRaises(ConfigError) as cm:
            validate_config({"run": {"id": "x"}, "seed": "42", "paths": {"root": "runs"}})
        self.assertEqual(cm.exception.code, EC.CONFIG_INVALID_TYPE)

    def test_ok_returns_resolved(self):
        cfg = validate_config({"run": {"id": "r1"}, "seed": 42, "paths": {"root": "runs"}})
        self.assertIsInstance(cfg, ResolvedConfig)
        self.assertEqual(cfg.run_id, "r1")
        self.assertEqual(cfg.seed, 42)
        self.assertEqual(cfg.root, "runs")
