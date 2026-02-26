import os
import tempfile
import unittest

from saltai.artifacts.checkpoints.manager import CheckpointManager


class DummyState(object):
    def __init__(self):
        self.w = 0

    def state_dict(self):
        return {"w": self.w}

    def load_state_dict(self, state):
        self.w = int(state["w"])


class TestCheckpointManager(unittest.TestCase):
    def test_save_and_load_latest(self):
        with tempfile.TemporaryDirectory() as d:
            mgr = CheckpointManager(root=d, keep_last=3)
            obj = DummyState()
            obj.w = 7

            ref = mgr.save_latest(obj, step=10)
            self.assertTrue(os.path.exists(mgr.path_of(ref)))

            obj2 = DummyState()
            mgr.load(obj2, ref)
            self.assertEqual(obj2.w, 7)

    def test_save_best_keeps_best_only(self):
        with tempfile.TemporaryDirectory() as d:
            mgr = CheckpointManager(root=d, keep_last=3)
            obj = DummyState()

            obj.w = 1
            r1 = mgr.save_best(obj, metric=0.10, step=1)

            obj.w = 2
            r2 = mgr.save_best(obj, metric=0.20, step=2)

            self.assertTrue(os.path.exists(mgr.path_of(r2)))
            self.assertFalse(os.path.exists(mgr.path_of(r1)))

    def test_retention_latest(self):
        with tempfile.TemporaryDirectory() as d:
            mgr = CheckpointManager(root=d, keep_last=2)
            obj = DummyState()

            obj.w = 1
            r1 = mgr.save_latest(obj, step=1)
            obj.w = 2
            r2 = mgr.save_latest(obj, step=2)
            obj.w = 3
            r3 = mgr.save_latest(obj, step=3)

            self.assertFalse(os.path.exists(mgr.path_of(r1)))
            self.assertTrue(os.path.exists(mgr.path_of(r2)))
            self.assertTrue(os.path.exists(mgr.path_of(r3)))
