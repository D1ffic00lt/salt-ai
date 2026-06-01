import unittest

from saltai import BaseLogger, EventBus, Logger, NoOpLogger


class _Event:
    def __init__(self, type):
        self.type = type


class _TypedLogger(BaseLogger):
    def __init__(self):
        self.metrics = []
        self.fallback = []

    def on_metric(self, event):
        self.metrics.append(event)

    def on_event(self, event):
        self.fallback.append(event)


class TestBaseLogger(unittest.TestCase):
    def test_dispatches_to_typed_handler(self):
        logger = _TypedLogger()
        event = _Event("metric")

        logger.log(event)

        self.assertEqual(logger.metrics, [event])
        self.assertEqual(logger.fallback, [])

    def test_falls_back_to_on_event(self):
        logger = _TypedLogger()
        event = _Event("run_started")

        logger.log(event)

        self.assertEqual(logger.metrics, [])
        self.assertEqual(logger.fallback, [event])

    def test_noop_logger_satisfies_logger_protocol(self):
        logger: Logger = NoOpLogger()

        logger.log(_Event("metric"))
        logger.flush()
        logger.close()

    def test_event_bus_accepts_base_logger(self):
        logger = _TypedLogger()
        event = _Event("metric")
        bus = EventBus([logger])

        bus.publish(event)

        self.assertEqual(logger.metrics, [event])
