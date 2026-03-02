import unittest

from saltai.utils.errors.base import (
    ArtifactError,
    ConfigError,
    DataError,
    EngineError,
    ErrorInfo,
    InternalError,
    LoggerError,
    ManifestError,
    MetricError,
    ReproducibilityError,
    SaltAIError,
)
from saltai.utils.errors.codes import EC


class TestErrorsBase(unittest.TestCase):
    def test_saltaierror_fields_and_str(self):
        e = SaltAIError(
            EC.CONFIG_MISSING_FIELD,
            "missing field",
            hint="add it",
            context={"k": "v"},
            cause=ValueError("x"),
        )
        self.assertEqual(e.code, EC.CONFIG_MISSING_FIELD)
        self.assertEqual(e.message, "missing field")
        self.assertEqual(e.hint, "add it")
        self.assertEqual(e.context, {"k": "v"})
        self.assertIsInstance(e.cause, ValueError)

        s = str(e)
        self.assertIn(EC.CONFIG_MISSING_FIELD, s)
        self.assertIn("missing field", s)
        self.assertIn("hint:", s)

    def test_to_info(self):
        e = SaltAIError(EC.DATA_NOT_FOUND, "no data", hint="check path", context={"p": "x"})
        info = e.to_info()
        self.assertIsInstance(info, ErrorInfo)
        self.assertEqual(info.code, EC.DATA_NOT_FOUND)
        self.assertEqual(info.message, "no data")
        self.assertEqual(info.hint, "check path")
        self.assertEqual(info.context, {"p": "x"})

    def test_with_context_merges(self):
        e = SaltAIError(EC.ENGINE_BAD_STATE, "bad", context={"a": 1})
        e2 = e.with_context(b=2)
        self.assertIsNot(e2, e)
        self.assertEqual(e2.context, {"a": 1, "b": 2})
        self.assertEqual(e.context, {"a": 1})

    def test_error_subclasses(self):
        for cls in [
            ConfigError,
            ManifestError,
            DataError,
            EngineError,
            ArtifactError,
            LoggerError,
            MetricError,
            ReproducibilityError,
            InternalError,
        ]:
            e = cls(EC.INTERNAL, "x")
            self.assertIsInstance(e, SaltAIError)
            self.assertIs(type(e), cls)
