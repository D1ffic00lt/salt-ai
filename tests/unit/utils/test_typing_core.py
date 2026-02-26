import unittest
from dataclasses import FrozenInstanceError

from saltai.utils.typing.core import (
    ArtifactId,
    ArtifactRef,
    ArtifactStore,
    DataModule,
    Logger,
    Metric,
    ModelAdapter,
    RunId,
)


class DummyDM:
    def prepare(self) -> None:
        self.prepared = True

    def train_iter(self):
        return iter([{"x": 1}])

    def val_iter(self):
        return None

    def test_iter(self):
        return None


class DummyModel:
    def forward(self, batch):
        return {"p": batch["x"]}

    def loss(self, pred, batch):
        return 0.0

    def zero_grad(self) -> None:
        return None

    def backward(self, loss) -> None:
        return None

    def step(self) -> None:
        return None

    def state_dict(self):
        return {"w": 1}

    def load_state_dict(self, state):
        self.loaded = state


class DummyMetric:
    name = "m"

    def __init__(self):
        self.s = 0

    def reset(self) -> None:
        self.s = 0

    def update(self, batch, pred) -> None:
        self.s += 1

    def compute(self):
        return float(self.s)


class DummyLogger:
    def __init__(self):
        self.events = []

    def log(self, event: object) -> None:
        self.events.append(event)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class DummyStore:
    def put(self, local_path, *, kind: str, name: str, meta=None):
        raise NotImplementedError

    def get(self, ref, *, dst_dir):
        raise NotImplementedError

    def exists(self, ref):
        return False

    def list(self, *, kind=None):
        return []


class TestTypingCore(unittest.TestCase):
    def test_runtime_protocol_checks(self):
        self.assertIsInstance(DummyDM(), DataModule)
        self.assertIsInstance(DummyModel(), ModelAdapter)
        self.assertIsInstance(DummyMetric(), Metric)
        self.assertIsInstance(DummyLogger(), Logger)
        self.assertIsInstance(DummyStore(), ArtifactStore)

    def test_artifactref_frozen(self):
        ref = ArtifactRef(
            id=ArtifactId("a"),
            kind="ckpt",
            name="best",
            uri="file://x",
            sha256=None,
            size_bytes=None,
            meta={},
        )
        with self.assertRaises(FrozenInstanceError):
            ref.name = "x"

    def test_newtypes_are_strings(self):
        rid = RunId("r1")
        aid = ArtifactId("a1")
        self.assertEqual(str(rid), "r1")
        self.assertEqual(str(aid), "a1")
