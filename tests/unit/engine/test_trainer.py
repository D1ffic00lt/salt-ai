import unittest

from saltai.engine.event_bus.bus import EventBus
from saltai.engine.trainer import Trainer


class _Sink(object):
    def __init__(self):
        self.events = []

    def log(self, event):
        self.events.append(event)

    def flush(self):
        return None

    def close(self):
        return None


class _DataModule(object):
    def __init__(self):
        self.prepared = 0

    def prepare(self):
        self.prepared += 1

    def train_iter(self):
        return iter([1, 2, 3])

    def val_iter(self):
        return iter([4, 5])

    def test_iter(self):
        return iter([6])


class _Model(object):
    def __init__(self):
        self.step_calls = 0

    def forward(self, batch):
        return batch * 2

    def loss(self, pred, batch):
        return pred - batch

    def zero_grad(self):
        return None

    def backward(self, loss):
        return None

    def step(self):
        self.step_calls += 1

    def state_dict(self):
        return {"step_calls": self.step_calls}

    def load_state_dict(self, state):
        self.step_calls = int(state["step_calls"])


class _Metric(object):
    name = "sum_pred"

    def __init__(self):
        self.total = 0

    def reset(self):
        self.total = 0

    def update(self, batch, pred):
        self.total += pred

    def compute(self):
        return self.total


class TestTrainer(unittest.TestCase):
    def test_fit_emits_events_and_returns_metrics(self):
        sink = _Sink()
        trainer = Trainer(event_bus=EventBus([sink]), run_id="run-fit")
        dm = _DataModule()
        model = _Model()
        metric = _Metric()

        out = trainer.fit(model, dm, metrics=[metric], epochs=2)

        self.assertEqual(dm.prepared, 1)
        self.assertEqual(model.step_calls, 6)
        self.assertEqual(out.values, {"sum_pred": 12})
        self.assertEqual(out.extra["epochs"], 2)
        self.assertEqual(out.extra["steps"], 6)

        event_types = [type(e).__name__ for e in sink.events]
        self.assertEqual(event_types.count("EpochStarted"), 2)
        self.assertEqual(event_types.count("EpochFinished"), 2)
        self.assertEqual(event_types.count("StepStarted"), 6)
        self.assertEqual(event_types.count("StepFinished"), 6)
        self.assertEqual(event_types.count("MetricLogged"), 2)

    def test_evaluate_val_split(self):
        sink = _Sink()
        trainer = Trainer(event_bus=EventBus([sink]), run_id="run-eval")
        dm = _DataModule()
        model = _Model()
        metric = _Metric()

        out = trainer.evaluate(model, dm, metrics=[metric], split="val")

        self.assertEqual(dm.prepared, 1)
        self.assertEqual(model.step_calls, 0)
        self.assertEqual(out.values, {"sum_pred": 18})
        self.assertEqual(out.extra, {"split": "val", "steps": 2})

        event_types = [type(e).__name__ for e in sink.events]
        self.assertEqual(event_types.count("EpochStarted"), 1)
        self.assertEqual(event_types.count("EpochFinished"), 1)
        self.assertEqual(event_types.count("StepStarted"), 2)
        self.assertEqual(event_types.count("StepFinished"), 2)
        self.assertEqual(event_types.count("MetricLogged"), 1)


if __name__ == "__main__":
    unittest.main()
