import unittest

from saltai.utils.errors.signals import Cancelled, EarlyStop, StopRun


class TestErrorsSignals(unittest.TestCase):
    def test_stoprun_is_exception(self):
        e = StopRun("reason", {"a": 1})
        self.assertIsInstance(e, Exception)
        self.assertEqual(e.reason, "reason")
        self.assertEqual(e.context, {"a": 1})

    def test_earlystop_cancelled_are_stoprun(self):
        e1 = EarlyStop("x", {})
        e2 = Cancelled("y", {"k": "v"})
        self.assertIsInstance(e1, StopRun)
        self.assertIsInstance(e2, StopRun)
        self.assertEqual(e2.context["k"], "v")
