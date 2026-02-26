import json
import tempfile
import unittest

from saltai.logging.sinks.jsonl import JsonlLogger


class TestJsonlLogger(unittest.TestCase):
    def test_writes_jsonl_lines(self):
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/events.jsonl"
            lg = JsonlLogger(path, flush_each=True)

            lg.log({"type": "a", "x": 1})
            lg.log({"type": "b", "y": 2})
            lg.close()

            with open(path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["type"], "a")
            self.assertEqual(json.loads(lines[1])["type"], "b")
