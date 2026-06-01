import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from saltai import Runner
from saltai.engine.event_bus.bus import EventBus
from saltai.utils.typing.core import ArtifactId, ArtifactRef, MetricPoint, RunId
from saltai.utils.typing.events import ArtifactSaved, CheckpointSaved, MetricLogged, RunStarted
from saltai.integrations.clearml import ClearMLLogger, ClearMLNotInstalledError
from saltai.logging.filters import EventTypeFilter, FilteredLogger


class FakeClearMLInnerLogger(object):
    def __init__(self):
        self.scalars = []
        self.texts = []
        self.flushed = False

    def report_scalar(self, *, title, series, value, iteration):
        self.scalars.append((title, series, value, iteration))

    def report_text(self, text):
        self.texts.append(text)

    def flush(self):
        self.flushed = True


class FakeClearMLTask(object):
    def __init__(self):
        self.logger = FakeClearMLInnerLogger()
        self.artifacts = []
        self.connected = []
        self.closed = False

    def get_logger(self):
        return self.logger

    def upload_artifact(self, *, name, artifact_object, metadata=None):
        self.artifacts.append((name, artifact_object, metadata))

    def connect(self, obj, name=None):
        self.connected.append((obj, name))
        return obj

    def close(self):
        self.closed = True


class TestClearMLLogger(unittest.TestCase):
    def test_import_does_not_require_clearml_when_task_is_injected(self):
        task = FakeClearMLTask()

        lg = ClearMLLogger(task=task)

        self.assertIs(lg.task, task)

    def test_missing_clearml_has_helpful_error(self):
        with patch(
                "saltai.integrations.clearml.logger._load_task_class",
                side_effect=ClearMLNotInstalledError(
                    "ClearML is not installed. Install it with `pip install salt-ai[clearml]`."
                ),
        ):
            with self.assertRaisesRegex(ClearMLNotInstalledError, "salt-ai\\[clearml\\]"):
                ClearMLLogger(project_name="SaltAI", task_name="run")

    def test_reports_metric_event_as_scalar(self):
        task = FakeClearMLTask()
        lg = ClearMLLogger(task=task)

        lg.log(
            MetricLogged(
                type="metric",
                run_id=RunId("r1"),
                ts=time.time(),
                data={},
                point=MetricPoint(
                    name="accuracy",
                    value=0.91,
                    step=7,
                    epoch=1,
                    split="val",
                    extra={},
                ),
            )
        )

        self.assertEqual(task.logger.scalars, [("val", "accuracy", 0.91, 7)])

    def test_uploads_artifact_event(self):
        task = FakeClearMLTask()
        lg = ClearMLLogger(task=task)

        ref = ArtifactRef(
            id=ArtifactId("a1"),
            kind="model",
            name="model.pt",
            uri="/tmp/model.pt",
            sha256=None,
            size_bytes=None,
            meta={"format": "pt"},
        )

        lg.log(
            ArtifactSaved(
                type="artifact_saved",
                run_id=RunId("r1"),
                ts=time.time(),
                data={},
                ref=ref,
            )
        )

        self.assertEqual(len(task.artifacts), 1)
        self.assertEqual(task.artifacts[0][0], "model.pt")
        self.assertEqual(task.artifacts[0][1], Path("/tmp/model.pt"))
        self.assertEqual(task.artifacts[0][2], {"format": "pt"})

    def test_uploads_checkpoint_event_with_checkpoint_name(self):
        task = FakeClearMLTask()
        lg = ClearMLLogger(task=task)

        ref = ArtifactRef(
            id=ArtifactId("c1"),
            kind="checkpoint",
            name="latest.pkl",
            uri="file:///tmp/latest.pkl",
            sha256=None,
            size_bytes=None,
            meta={},
        )

        lg.log(
            CheckpointSaved(
                type="checkpoint_saved",
                run_id=RunId("r1"),
                ts=time.time(),
                data={},
                ref=ref,
                tag="latest",
            )
        )

        self.assertEqual(task.artifacts[0][0], "checkpoint/latest")
        self.assertEqual(task.artifacts[0][1], Path("/tmp/latest.pkl"))

    def test_report_events_is_opt_in(self):
        task = FakeClearMLTask()
        lg = ClearMLLogger(task=task, report_events=True)

        lg.log(RunStarted(type="run_started", run_id=RunId("r1"), ts=time.time(), data={}))

        self.assertEqual(len(task.logger.texts), 1)
        self.assertIn("run_started", task.logger.texts[0])

    def test_runner_io_helpers_reach_clearml_logger(self):
        with tempfile.TemporaryDirectory() as d:
            task = FakeClearMLTask()
            lg = ClearMLLogger(task=task)
            bus = EventBus([
                FilteredLogger(
                    lg,
                    EventTypeFilter(include={"metric", "artifact_saved"}),
                )
            ])

            src = os.path.join(d, "manual.txt")
            with open(src, "w", encoding="utf-8") as f:
                f.write("artifact")

            runner = Runner(event_bus=bus)

            def body(ctx):
                point = ctx.io.log_metric(
                    "manual_accuracy",
                    0.93,
                    step=5,
                    epoch=1,
                    split="val",
                    extra={"source": "body"},
                )
                ref = ctx.io.save_artifact(
                    src,
                    kind="model",
                    name="manual-model",
                    meta={"format": "txt"},
                )

                return {
                    "manual_accuracy": point.value,
                    "artifact_name": ref.name,
                }

            res = runner.run(
                {"run": {"id": "r_clearml_helpers"}, "seed": 42, "paths": {"root": d}},
                body=body,
            )

            self.assertEqual(res.status, "success")
            self.assertEqual(res.metrics.values["manual_accuracy"], 0.93)
            self.assertEqual(res.metrics.values["artifact_name"], "manual-model")

            self.assertEqual(task.logger.scalars, [("val", "manual_accuracy", 0.93, 5)])

            self.assertEqual(len(task.artifacts), 1)
            self.assertEqual(task.artifacts[0][0], "manual-model")
            self.assertIsInstance(task.artifacts[0][1], Path)
            self.assertTrue(task.artifacts[0][1].exists())
            self.assertEqual(task.artifacts[0][2], {"format": "txt"})

    def test_flush_and_close(self):
        task = FakeClearMLTask()
        lg = ClearMLLogger(task=task)

        lg.close()

        self.assertTrue(task.logger.flushed)
        self.assertTrue(task.closed)


if __name__ == "__main__":
    unittest.main()
