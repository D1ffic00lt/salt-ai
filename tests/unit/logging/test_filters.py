import unittest
from dataclasses import dataclass

from saltai.logging.filters import EventTypeFilter, FilteredLogger
from saltai.utils.typing.core import Logger


@dataclass(frozen=True)
class DummyEvent:
    type: str
    value: int = 1


class DummyLogger:
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


class TestLoggingFilters(unittest.TestCase):
    def test_event_type_filter_include_allows_only_selected_types(self):
        event_filter = EventTypeFilter(include={"metric", "artifact_saved"})

        self.assertTrue(event_filter(DummyEvent(type="metric")))
        self.assertTrue(event_filter(DummyEvent(type="artifact_saved")))
        self.assertFalse(event_filter(DummyEvent(type="run_started")))

    def test_event_type_filter_exclude_blocks_selected_types(self):
        event_filter = EventTypeFilter(exclude={"step_started", "step_finished"})

        self.assertTrue(event_filter(DummyEvent(type="metric")))
        self.assertFalse(event_filter(DummyEvent(type="step_started")))
        self.assertFalse(event_filter(DummyEvent(type="step_finished")))

    def test_event_type_filter_supports_mapping_events(self):
        event_filter = EventTypeFilter(include={"metric"})

        self.assertTrue(event_filter({"type": "metric"}))
        self.assertFalse(event_filter({"type": "run_finished"}))

    def test_filtered_logger_routes_only_allowed_events(self):
        sink = DummyLogger()
        logger = FilteredLogger(
            sink,
            EventTypeFilter(include={"metric"}),
        )

        metric = DummyEvent(type="metric")
        run_started = DummyEvent(type="run_started")

        logger.log(metric)
        logger.log(run_started)

        self.assertIsInstance(logger, Logger)
        self.assertEqual(sink.events, [metric])

    def test_filtered_logger_delegates_flush_and_close(self):
        sink = DummyLogger()
        logger = FilteredLogger(
            sink,
            EventTypeFilter(include={"metric"}),
        )

        logger.flush()
        logger.close()

        self.assertEqual(sink.flushed, 1)
        self.assertEqual(sink.closed, 1)


if __name__ == "__main__":
    unittest.main()
