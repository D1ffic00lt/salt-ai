import os
import tempfile
import unittest

from saltai.artifacts.store.local import LocalArtifactStore


class TestLocalArtifactStore(unittest.TestCase):
    def test_put_and_get(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(root=os.path.join(d, "store"))

            src = os.path.join(d, "x.txt")
            with open(src, "w", encoding="utf-8") as f:
                f.write("hello")

            ref = store.put(src, kind="file", name="x")

            self.assertTrue(store.exists(ref))

            out_dir = os.path.join(d, "out")
            os.makedirs(out_dir, exist_ok=True)
            dst = store.get(ref, dst_dir=out_dir)

            self.assertTrue(os.path.exists(dst))
            with open(dst, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "hello")

    def test_list_by_kind(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(root=os.path.join(d, "store"))

            for i in range(3):
                p = os.path.join(d, f"a{i}.txt")
                with open(p, "w", encoding="utf-8") as f:
                    f.write(str(i))
                store.put(p, kind="k1", name=f"a{i}")

            p2 = os.path.join(d, "b.txt")
            with open(p2, "w", encoding="utf-8") as f:
                f.write("b")
            store.put(p2, kind="k2", name="b")

            k1 = store.list(kind="k1")
            k2 = store.list(kind="k2")
            all_ = store.list()

            self.assertEqual(len(k1), 3)
            self.assertEqual(len(k2), 1)
            self.assertEqual(len(all_), 4)
