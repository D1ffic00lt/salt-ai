import unittest

from saltai.utils.errors.base import InternalError, SaltAIError
from saltai.utils.errors.codes import EC
from saltai.utils.errors.helpers import ensure, guard, wrap_unknown


class TestErrorsHelpers(unittest.TestCase):
    def test_ensure_ok(self):
        ensure(True, code=EC.INTERNAL, message="nope")

    def test_ensure_raises(self):
        with self.assertRaises(SaltAIError) as cm:
            ensure(False, code=EC.CONFIG_CONSTRAINT, message="bad", hint="fix", context={"x": 1})
        e = cm.exception
        self.assertEqual(e.code, EC.CONFIG_CONSTRAINT)
        self.assertEqual(e.hint, "fix")
        self.assertEqual(e.context["x"], 1)

    def test_wrap_unknown_passthrough(self):
        e0 = SaltAIError(EC.DATA_NOT_FOUND, "x")
        e1 = wrap_unknown(e0)
        self.assertIs(e1, e0)

    def test_wrap_unknown_valueerror(self):
        e = wrap_unknown(ValueError("boom"), context={"a": 1})
        self.assertIsInstance(e, InternalError)
        self.assertEqual(e.code, EC.INTERNAL)
        self.assertEqual(e.context["a"], 1)
        self.assertIsInstance(e.cause, ValueError)

    def test_guard_wraps_exception_with_stage(self):
        def f():
            raise ValueError("boom")

        with self.assertRaises(InternalError) as cm:
            guard("train", f, context={"run_id": "1"})
        e = cm.exception
        self.assertEqual(e.code, EC.INTERNAL)
        self.assertEqual(e.context["stage"], "train")
        self.assertEqual(e.context["run_id"], "1")
        self.assertIsInstance(e.cause, ValueError)
