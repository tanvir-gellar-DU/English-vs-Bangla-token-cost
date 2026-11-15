"""Validation tests for data/dataset.tsv (written before the dataset)."""
import csv
import re
import unittest
from collections import Counter
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "data" / "dataset.tsv"
CATEGORIES = {"question", "instruction", "explanation", "arithmetic", "summarise", "long"}
BENGALI = re.compile(r"[ঀ-৿]")
LATIN = re.compile(r"[A-Za-z]")
SENT_END = re.compile(r"[.?!।]")


def load():
    with open(PATH, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load()

    def test_columns(self):
        self.assertEqual(list(self.rows[0].keys()), ["id", "category", "english", "bangla"])

    def test_fifty_pairs_unique_ids(self):
        self.assertEqual(len(self.rows), 50)
        self.assertEqual(sorted(int(r["id"]) for r in self.rows), list(range(1, 51)))

    def test_no_empty_cells(self):
        for r in self.rows:
            for k, v in r.items():
                self.assertTrue(v and v.strip(), f"empty {k} in id {r['id']}")

    def test_categories_valid_and_mixed(self):
        counts = Counter(r["category"] for r in self.rows)
        self.assertLessEqual(set(counts), CATEGORIES)
        self.assertEqual(set(counts), CATEGORIES)
        for c, n in counts.items():
            self.assertGreaterEqual(n, 5, c)

    def test_scripts(self):
        for r in self.rows:
            self.assertFalse(BENGALI.search(r["english"]), f"Bengali in english id {r['id']}")
            self.assertTrue(BENGALI.search(r["bangla"]), f"no Bengali in bangla id {r['id']}")
            self.assertFalse(LATIN.search(r["bangla"]), f"Latin letters in bangla id {r['id']}")

    def test_lengths_vary(self):
        words = [len(r["english"].split()) for r in self.rows]
        self.assertGreaterEqual(max(words) / min(words), 5)

    def test_long_prompts_have_3_to_5_sentences(self):
        for r in self.rows:
            if r["category"] == "long":
                n = len(SENT_END.findall(r["english"]))
                self.assertTrue(3 <= n <= 5, f"id {r['id']} has {n} sentences")

    def test_same_sentence_count_in_both_languages(self):
        # Cheap alignment check: Bangla version must not add or drop sentences.
        for r in self.rows:
            self.assertEqual(len(SENT_END.findall(r["english"])),
                             len(SENT_END.findall(r["bangla"])), f"id {r['id']}")


if __name__ == "__main__":
    unittest.main()
