import os
import tempfile
import unittest
from dataclasses import dataclass

from saltai.artifacts.refs import make_artifact_ref
from saltai.artifacts.store.base import BaseArtifactStore
from saltai.engine.event_bus.bus import EventBus
from saltai.engine.runner.runner import Runner
from saltai.logging.base import BaseLogger
from saltai.utils.typing.core import ArtifactRef, ArtifactStore, Logger


@dataclass(frozen=True)
class DummyEvent:
    type: str
    value: int = 1


class InheritedLogger(BaseLogger):
    def __init__(self):
        self.metrics = []
        self.events = []

    def on_metric(self, event: object) -> None:
        self.metrics.append(event)

    def on_event(self, event: object) -> None:
        self.events.append(event)


class ProtocolOnlyLogger:
    def __init__(self):
        self.events = []
        self.flushed = False
        self.closed = False

    def log(self, event: object) -> None:
        self.events.append(event)

    def flush(self) -> None:
        self.flushed = True

    def close(self) -> None:
        self.closed = True


class InheritedArtifactStore(BaseArtifactStore):
    def __init__(self):
        self.refs: list[ArtifactRef] = []

    def put(self, local_path: str, *, kind: str, name: str, meta=None) -> ArtifactRef:
        ref = make_artifact_ref(
            id=f"a{len(self.refs)}",
            kind=kind,
            name=name,
            uri=f"memory://{kind}/{name}/{len(self.refs)}",
            sha256=None,
            size_bytes=os.path.getsize(local_path),
            meta=meta or {},
        )
        self.refs.append(ref)
        return ref

    def get(self, ref: ArtifactRef, *, dst_dir: str) -> str:
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, f"{ref.name}.artifact")
        with open(dst, "w", encoding="utf-8") as f:
            f.write(ref.uri)
        return dst

    def exists(self, ref: ArtifactRef) -> bool:
        return any(item.id == ref.id for item in self.refs)

    def list(self, *, kind: str | None = None):
        if kind is None:
            return tuple(self.refs)
        return tuple(ref for ref in self.refs if ref.kind == kind)


class ProtocolOnlyArtifactStore:
    def __init__(self):
        self.refs: list[ArtifactRef] = []

    def put(self, local_path: str, *, kind: str, name: str, meta=None) -> ArtifactRef:
        ref = make_artifact_ref(
            id=f"p{len(self.refs)}",
            kind=kind,
            name=name,
            uri=f"memory://{kind}/{name}/{len(self.refs)}",
            sha256=None,
            size_bytes=os.path.getsize(local_path),
            meta=meta or {},
        )
        self.refs.append(ref)
        return ref

    def get(self, ref: ArtifactRef, *, dst_dir: str) -> str:
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, f"{ref.name}.artifact")
        with open(dst, "w", encoding="utf-8") as f:
            f.write(ref.uri)
        return dst

    def exists(self, ref: ArtifactRef) -> bool:
        return any(item.id == ref.id for item in self.refs)

    def list(self, *, kind: str | None = None):
        if kind is None:
            return tuple(self.refs)
        return tuple(ref for ref in self.refs if ref.kind == kind)


class TestExtensionContracts(unittest.TestCase):
    def test_custom_logger_via_base_logger(self):
        logger = InheritedLogger()

        metric_event = DummyEvent(type="metric", value=10)
        fallback_event = DummyEvent(type="unknown", value=20)

        logger.log(metric_event)
        logger.log(fallback_event)

        self.assertIsInstance(logger, Logger)
        self.assertEqual(logger.metrics, [metric_event])
        self.assertEqual(logger.events, [fallback_event])

    def test_custom_logger_protocol_only(self):
        logger = ProtocolOnlyLogger()
        event = DummyEvent(type="custom")

        logger.log(event)
        logger.flush()
        logger.close()

        self.assertIsInstance(logger, Logger)
        self.assertEqual(logger.events, [event])
        self.assertTrue(logger.flushed)
        self.assertTrue(logger.closed)

    def test_custom_artifact_store_via_base_artifact_store(self):
        with tempfile.TemporaryDirectory() as d:
            store = InheritedArtifactStore()
            src = os.path.join(d, "artifact.txt")

            with open(src, "w", encoding="utf-8") as f:
                f.write("payload")

            ref = store.put(src, kind="file", name="artifact", meta={"source": "test"})

            self.assertIsInstance(store, ArtifactStore)
            self.assertIsInstance(store, BaseArtifactStore)
            self.assertTrue(store.exists(ref))
            self.assertEqual(store.list(kind="file"), (ref,))

    def test_custom_artifact_store_protocol_only(self):
        with tempfile.TemporaryDirectory() as d:
            store = ProtocolOnlyArtifactStore()
            src = os.path.join(d, "artifact.txt")

            with open(src, "w", encoding="utf-8") as f:
                f.write("payload")

            ref = store.put(src, kind="file", name="artifact")

            self.assertIsInstance(store, ArtifactStore)
            self.assertTrue(store.exists(ref))
            self.assertEqual(store.list(), (ref,))

    def test_runner_works_with_custom_artifact_store_factory(self):
        with tempfile.TemporaryDirectory() as d:
            stores: list[InheritedArtifactStore] = []

            def factory(_run_dir: str):
                store = InheritedArtifactStore()
                stores.append(store)
                return store

            runner = Runner(artifact_store_factory=factory)

            def body(ctx):
                src = os.path.join(ctx.run_dir, "payload.txt")
                with open(src, "w", encoding="utf-8") as f:
                    f.write("payload")

                ctx.io.save_artifact(src, kind="file", name="payload", meta={"x": 1})
                return {"val": {"acc": 0.9}}

            result = runner.run(
                {
                    "run": {"id": "custom_store_runner"},
                    "seed": 42,
                    "paths": {"root": d},
                },
                body=body,
            )

            self.assertEqual(result.status, "success")
            self.assertEqual(len(stores), 1)
            self.assertEqual(len(stores[0].refs), 1)
            self.assertEqual(len(result.artifacts), 1)
            self.assertEqual(result.artifacts[0].uri, "memory://file/payload/0")
            self.assertEqual(result.metrics.values["val"]["acc"], 0.9)

    def test_event_bus_works_with_multiple_custom_logger_sinks(self):
        first = ProtocolOnlyLogger()
        second = InheritedLogger()
        bus = EventBus([first, second])

        event = DummyEvent(type="custom")

        bus.publish(event)
        bus.flush()
        bus.close()

        self.assertEqual(first.events, [event])
        self.assertEqual(second.events, [event])


if __name__ == "__main__":
    unittest.main()
