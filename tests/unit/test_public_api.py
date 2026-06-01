import unittest

import saltai
from saltai import errors, events
from saltai.artifacts.store import BaseArtifactStore
from saltai.errors import SaltAIError
from saltai.events import ArtifactSaved, MetricLogged


class TestPublicAPI(unittest.TestCase):
    def test_root_exports_core_entrypoints(self):
        self.assertIs(saltai.Runner, saltai.Runner)
        self.assertIs(saltai.Trainer, saltai.Trainer)
        self.assertIs(saltai.EventBus, saltai.EventBus)
        self.assertIs(saltai.BaseLogger, saltai.BaseLogger)
        self.assertIs(saltai.NoOpLogger, saltai.NoOpLogger)
        self.assertIs(saltai.BaseArtifactStore, BaseArtifactStore)
        self.assertIs(saltai.LocalArtifactStore, saltai.LocalArtifactStore)

    def test_root_exports_namespaces(self):
        self.assertIs(saltai.errors, errors)
        self.assertIs(saltai.events, events)

    def test_errors_namespace(self):
        self.assertIs(errors.SaltAIError, SaltAIError)

    def test_events_namespace(self):
        self.assertIs(events.MetricLogged, MetricLogged)
        self.assertIs(events.ArtifactSaved, ArtifactSaved)

    def test_error_classes_are_not_root_exports(self):
        self.assertNotIn("SaltAIError", saltai.__all__)
        self.assertFalse(hasattr(saltai, "SaltAIError"))


if __name__ == "__main__":
    unittest.main()
