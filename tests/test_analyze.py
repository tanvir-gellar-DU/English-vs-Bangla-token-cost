"""Analysis tests on hand-computed numbers (written before analyze.py)."""
import statistics
import unittest

from analyze import bootstrap_ci, by_category, describe, overall, pair_table


def rec(i, lang, cat, inp, out, lat, trunc=False, overhead=24):
    return {"id": str(i), "language": lang, "category": cat, "input_tokens": inp,
            "content_tokens": inp - overhead, "output_tokens": out,
            "cost_usd": inp * 0.15e-6 + out * 0.60e-6, "latency_s": lat, "truncated": trunc}


RESULTS = [
    rec(1, "english", "question", 34, 10, 1.0), rec(1, "bangla", "question", 44, 20, 2.0),
    rec(2, "english", "long", 74, 100, 3.0), rec(2, "bangla", "long", 174, 100, 6.0, trunc=True),
]


class TestPairTable(unittest.TestCase):
    def setUp(self):
        self.pairs = {p["id"]: p for p in pair_table(RESULTS)}

    def test_one_row_per_pair(self):
        self.assertEqual(len(self.pairs), 2)

    def test_tokens_and_ratios(self):
        p = self.pairs["1"]
        self.assertEqual((p["en_input"], p["bn_input"]), (34, 44))
        self.assertEqual((p["en_output"], p["bn_output"]), (10, 20))
        self.assertAlmostEqual(p["input_ratio"], 44 / 34)
        self.assertAlmostEqual(p["content_ratio"], 20 / 10)
        self.assertAlmostEqual(p["output_ratio"], 2.0)
        en_cost = 34 * 0.15e-6 + 10 * 0.60e-6
        bn_cost = 44 * 0.15e-6 + 20 * 0.60e-6
        self.assertAlmostEqual(p["total_cost_ratio"], bn_cost / en_cost)
        self.assertEqual(p["category"], "question")

    def test_truncation_carried(self):
        self.assertTrue(self.pairs["2"]["bn_truncated"])
        self.assertFalse(self.pairs["2"]["en_truncated"])

    def test_missing_partner_rejected(self):
        with self.assertRaises(ValueError):
            pair_table(RESULTS[:3])


class TestStats(unittest.TestCase):
    def test_describe(self):
        d = describe([1.0, 2.0, 3.0, 10.0])
        self.assertAlmostEqual(d["mean"], 4.0)
        self.assertAlmostEqual(d["median"], 2.5)
        self.assertEqual((d["min"], d["max"]), (1.0, 10.0))
        self.assertLessEqual(d["ci95_low"], d["mean"])
        self.assertGreaterEqual(d["ci95_high"], d["mean"])

    def test_bootstrap_deterministic_and_contains_mean(self):
        vals = [1.5, 1.7, 2.0, 2.2, 2.4, 3.0, 1.9, 2.1]
        a = bootstrap_ci(vals, seed=1)
        b = bootstrap_ci(vals, seed=1)
        self.assertEqual(a, b)
        self.assertLess(a[0], statistics.mean(vals))
        self.assertGreater(a[1], statistics.mean(vals))
        self.assertGreaterEqual(a[0], min(vals))
        self.assertLessEqual(a[1], max(vals))

    def test_bootstrap_constant(self):
        self.assertEqual(bootstrap_ci([2.0] * 5), (2.0, 2.0))


class TestOverall(unittest.TestCase):
    def test_totals_and_ratios(self):
        o = overall(RESULTS)
        self.assertEqual(o["english"]["input_tokens"], 108)
        self.assertEqual(o["bangla"]["input_tokens"], 218)
        self.assertEqual(o["english"]["output_tokens"], 110)
        self.assertEqual(o["bangla"]["output_tokens"], 120)
        en_cost = 108 * 0.15e-6 + 110 * 0.60e-6
        bn_cost = 218 * 0.15e-6 + 120 * 0.60e-6
        self.assertAlmostEqual(o["english"]["cost_usd"], en_cost)
        self.assertAlmostEqual(o["total_cost_ratio"], bn_cost / en_cost)
        self.assertAlmostEqual(o["input_ratio"], 218 / 108)
        self.assertAlmostEqual(o["english"]["latency_mean_s"], 2.0)
        self.assertAlmostEqual(o["bangla"]["latency_mean_s"], 4.0)
        self.assertEqual(o["bangla"]["truncated_calls"], 1)


class TestByCategory(unittest.TestCase):
    def test_grouping(self):
        cats = {c["category"]: c for c in by_category(pair_table(RESULTS))}
        self.assertEqual(set(cats), {"question", "long"})
        self.assertEqual(cats["long"]["n"], 1)
        self.assertAlmostEqual(cats["long"]["input_ratio_mean"], 174 / 74)


if __name__ == "__main__":
    unittest.main()
