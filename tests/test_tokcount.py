"""Tests for tokcount (written before the implementation). Uses a fake backend: no Ollama needed."""
import unittest

from tokcount import (
    PRICE_IN_PER_TOKEN,
    PRICE_OUT_PER_TOKEN,
    FakeBackend,
    TokenCounter,
    check_cache,
    cost_usd,
)


def fake_response(prompt_eval_count=20, eval_count=7, content="ok", done_reason="stop"):
    r = {
        "message": {"role": "assistant", "content": content},
        "eval_count": eval_count,
        "done_reason": done_reason,
        "done": True,
    }
    if prompt_eval_count is not None:
        r["prompt_eval_count"] = prompt_eval_count
    return r


class TestCost(unittest.TestCase):
    def test_prices(self):
        self.assertAlmostEqual(PRICE_IN_PER_TOKEN, 0.15 / 1_000_000)
        self.assertAlmostEqual(PRICE_OUT_PER_TOKEN, 0.60 / 1_000_000)

    def test_one_million_input_tokens(self):
        self.assertAlmostEqual(cost_usd(1_000_000, 0), 0.15)

    def test_one_million_output_tokens(self):
        self.assertAlmostEqual(cost_usd(0, 1_000_000), 0.60)

    def test_mixed(self):
        # 100 in, 50 out = 100*0.15e-6 + 50*0.60e-6 = 0.000045
        self.assertAlmostEqual(cost_usd(100, 50), 0.000045)

    def test_zero(self):
        self.assertEqual(cost_usd(0, 0), 0.0)

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            cost_usd(-1, 0)


class TestTokenCounter(unittest.TestCase):
    def make(self, responses, **kw):
        backend = FakeBackend(responses)
        tc = TokenCounter(backend, model="m", system="SYS",
                          options={"temperature": 0, "seed": 42, "num_predict": 256}, **kw)
        return tc, backend

    def test_extracts_counts_from_ollama_fields(self):
        tc, _ = self.make([fake_response(prompt_eval_count=33, eval_count=12, content="hello")])
        r = tc.call("What is 2+2?")
        self.assertEqual(r.input_tokens, 33)
        self.assertEqual(r.output_tokens, 12)
        self.assertEqual(r.output, "hello")

    def test_cost_attached_to_result(self):
        tc, _ = self.make([fake_response(prompt_eval_count=100, eval_count=50)])
        r = tc.call("x")
        self.assertAlmostEqual(r.cost_usd, 0.000045)

    def test_latency_measured(self):
        tc, _ = self.make([fake_response()])
        r = tc.call("x")
        self.assertGreaterEqual(r.latency_s, 0.0)

    def test_payload_has_same_system_and_settings(self):
        tc, backend = self.make([fake_response(), fake_response()])
        tc.call("english prompt")
        tc.call("বাংলা প্রম্পট")
        p1, p2 = backend.payloads
        self.assertEqual(p1["model"], p2["model"])
        self.assertEqual(p1["options"], p2["options"])
        self.assertEqual(p1["messages"][0], {"role": "system", "content": "SYS"})
        self.assertEqual(p1["messages"][0], p2["messages"][0])
        self.assertEqual(p2["messages"][1], {"role": "user", "content": "বাংলা প্রম্পট"})
        self.assertFalse(p1["stream"])

    def test_keep_alive_passed_through(self):
        tc, backend = self.make([fake_response()], keep_alive=0)
        tc.call("x")
        self.assertEqual(backend.payloads[0]["keep_alive"], 0)

    def test_truncation_flag(self):
        tc, _ = self.make([fake_response(done_reason="length"), fake_response(done_reason="stop")])
        self.assertTrue(tc.call("long").truncated)
        self.assertFalse(tc.call("short").truncated)

    def test_missing_prompt_eval_count_flags_cache(self):
        # Ollama omits/zeroes prompt_eval_count when the whole prompt came from cache.
        tc, _ = self.make([fake_response(prompt_eval_count=None)])
        r = tc.call("x")
        self.assertEqual(r.input_tokens, 0)
        self.assertTrue(r.cache_suspected)

    def test_drop_vs_expected_flags_cache(self):
        # Second call of the same prompt reports fewer tokens -> cache suspected.
        tc, _ = self.make([fake_response(prompt_eval_count=40), fake_response(prompt_eval_count=5)])
        a = tc.call("same")
        b = tc.call("same")
        self.assertFalse(a.cache_suspected)
        self.assertTrue(b.cache_suspected)

    def test_overhead_subtracted(self):
        tc, _ = self.make([fake_response(prompt_eval_count=50)])
        tc.overhead_tokens = 30
        r = tc.call("x")
        self.assertEqual(r.input_tokens, 50)
        self.assertEqual(r.content_tokens, 20)

    def test_measure_baseline(self):
        tc, backend = self.make([fake_response(prompt_eval_count=30)])
        n = tc.measure_baseline("Hi")
        self.assertEqual(n, 30)
        self.assertEqual(tc.overhead_tokens, 30)
        self.assertEqual(backend.payloads[0]["messages"][1]["content"], "Hi")

    def test_fake_backend_callable(self):
        # A function backend lets tests compute counts from the prompt.
        backend = FakeBackend(lambda payload: fake_response(
            prompt_eval_count=len(payload["messages"][1]["content"])))
        tc = TokenCounter(backend, model="m", system="S", options={})
        self.assertEqual(tc.call("abcd").input_tokens, 4)


class TestCacheCheck(unittest.TestCase):
    def test_stable_counts(self):
        backend = FakeBackend(lambda p: fake_response(prompt_eval_count=26))
        tc = TokenCounter(backend, model="m", system="S", options={})
        rows = check_cache(tc, ["a", "b"], repeats=3)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["counts"], [26, 26, 26])
        self.assertTrue(rows[0]["stable"])
        self.assertEqual(len(backend.payloads), 6)

    def test_unstable_counts_detected(self):
        tc = TokenCounter(FakeBackend([fake_response(prompt_eval_count=26),
                                       fake_response(prompt_eval_count=4)]),
                          model="m", system="S", options={})
        rows = check_cache(tc, ["a"], repeats=2)
        self.assertFalse(rows[0]["stable"])
        self.assertTrue(rows[0]["cache_suspected"])


if __name__ == "__main__":
    unittest.main()
