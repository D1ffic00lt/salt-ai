import unittest

from saltai.engine.event_bus.bus import EventBus
from saltai.utils.errors.base import LoggerError
from saltai.utils.errors.codes import EC


class _SinkOK(object):
    def __init__(self):
        self.events = []
        self.flushed = 0
        self.closed = 0

    def log(self, event):
        self.events.append(event)

    def flush(self):
        self.flushed += 1

    def close(self):
        self.closed += 1


class _SinkFail:
    def __init__(self, exc=None):
        self.events = []
        self.exc = exc or RuntimeError("sink boom")

    def log(self, event):
        self.events.append(event)
        raise self.exc

    def flush(self):
        return None

    def close(self):
        return None


class TestEventBus(unittest.TestCase):
    def test_publish_to_all_sinks(self):
        a = _SinkOK()
        b = _SinkOK()
        bus = EventBus([a, b])

        ev = {"type": "x", "k": 1}
        bus.publish(ev)

        self.assertEqual(a.events, [ev])
        self.assertEqual(b.events, [ev])

    def test_flush_and_close(self):
        a = _SinkOK()
        b = _SinkOK()
        bus = EventBus([a, b])

        bus.flush()
        bus.close()

        self.assertEqual(a.flushed, 1)
        self.assertEqual(b.flushed, 1)
        self.assertEqual(a.closed, 1)
        self.assertEqual(b.closed, 1)

    def test_fail_fast_true_raises_on_first_failure(self):
        ok = _SinkOK()
        bad = _SinkFail()
        ok2 = _SinkOK()

        bus = EventBus([ok, bad, ok2], fail_fast=True)

        with self.assertRaises(LoggerError) as cm:
            bus.publish({"type": "x"}, context={"run_id": "r1"})

        e = cm.exception
        self.assertEqual(e.code, EC.LOG_SINK_FAILED)
        self.assertIn("sink", e.context)
        self.assertEqual(e.context.get("event_type"), "dict")
        self.assertEqual(e.context.get("run_id"), "r1")

        self.assertEqual(len(ok.events), 1)
        self.assertEqual(len(bad.events), 1)
        self.assertEqual(len(ok2.events), 0)

    def test_fail_fast_false_collects_failures(self):
        ok = _SinkOK()
        bad1 = _SinkFail(RuntimeError("boom1"))
        bad2 = _SinkFail(RuntimeError("boom2"))

        bus = EventBus([bad1, ok, bad2], fail_fast=False)

        with self.assertRaises(LoggerError) as cm:
            bus.publish({"type": "x"})

        e = cm.exception
        self.assertEqual(e.code, EC.LOG_SINK_FAILED)
        self.assertEqual(e.context.get("event_type"), "dict")
        self.assertEqual(e.context.get("failed_sinks"), 2)

        self.assertEqual(len(bad1.events), 1)
        self.assertEqual(len(ok.events), 1)
        self.assertEqual(len(bad2.events), 1)
