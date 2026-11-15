"""Runner tests with a fake backend (written before runner.py)."""
import csv
import tempfile
import unittest
from pathlib import Path

from runner import RESULT_COLUMNS, load_dataset, run_all, write_results
from tokcount import FakeBackend, TokenCounter

ROWS = [
    {"id": "1", "category": "question", "english": "What is 2+2?", "bangla": "২+২ কত?"},
    {"id": "2", "category": "long", "english": "One. Two. Three.", "bangla": "এক। দুই। তিন।"},
]


def fake(payload):
    text = payload["messages"][1]["content"]
    return {"message": {"content": "ans:" + text}, "prompt_eval_count": 24 + len(text),
            "eval_count": 5, "done_reason": "stop"}


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend(fake)
        self.tc = TokenCounter(self.backend, "m", "SYS", {"temperature": 0, "seed": 42, "num_predict": 8})
        self.tc.overhead_tokens = 24

    def test_one_row_per_prompt_per_language(self):
        out = run_all(self.tc, ROWS)
        self.assertEqual(len(out), 4)
        self.assertEqual(sorted((r["id"], r["language"]) for r in out),
                         [("1", "bangla"), ("1", "english"), ("2", "bangla"), ("2", "english")])

    def test_same_settings_for_every_call(self):
        run_all(self.tc, ROWS)
        first = self.backend.payloads[0]
        for p in self.backend.payloads:
            self.assertEqual(p["model"], first["model"])
            self.assertEqual(p["options"], first["options"])
            self.assertEqual(p["messages"][0], first["messages"][0])

    def test_order_alternates_to_cancel_drift(self):
        out = run_all(self.tc, ROWS)
        self.assertEqual([r["language"] for r in out], ["english", "bangla", "bangla", "english"])

    def test_fields_recorded(self):
        out = run_all(self.tc, ROWS)
        r = next(x for x in out if x["id"] == "1" and x["language"] == "bangla")
        self.assertEqual(r["prompt"], "২+২ কত?")
        self.assertEqual(r["output"], "ans:২+২ কত?")
        self.assertEqual(r["input_tokens"], 24 + len("২+২ কত?"))
        self.assertEqual(r["content_tokens"], len("২+২ কত?"))
        self.assertEqual(r["output_tokens"], 5)
        self.assertAlmostEqual(r["cost_usd"], r["input_tokens"] * 0.15e-6 + 5 * 0.60e-6)
        self.assertEqual(r["category"], "question")
        for col in RESULT_COLUMNS:
            self.assertIn(col, r)

    def test_write_and_reload_csv_keeps_bangla(self):
        out = run_all(self.tc, ROWS)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "results.csv"
            write_results(out, path)
            with open(path, encoding="utf-8", newline="") as f:
                back = list(csv.DictReader(f))
        self.assertEqual(list(back[0].keys()), RESULT_COLUMNS)
        self.assertEqual(len(back), 4)
        self.assertIn("২+২ কত?", [b["prompt"] for b in back])

    def test_load_dataset(self):
        rows = load_dataset()
        self.assertEqual(len(rows), 50)


if __name__ == "__main__":
    unittest.main()
