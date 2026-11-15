"""Token counting and notional cost for Ollama chat calls.

Input tokens  = Ollama's `prompt_eval_count` (includes chat template + system prompt).
Output tokens = Ollama's `eval_count`.

Template overhead: `measure_baseline()` sends a tiny prompt once and stores its count in
`overhead_tokens`; each result then also reports `content_tokens = input_tokens - overhead`.

Prompt caching: Ollama may reuse a cached prompt prefix, so `prompt_eval_count` can be smaller
(or missing) on a repeated prompt. Every result carries `cache_suspected`, set when the count is
missing/zero or lower than the first count seen for the same (system, prompt). Pass
`keep_alive=0` to unload the model after every call so each call starts from a fresh state.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import asdict, dataclass

PRICE_IN_PER_TOKEN = 0.15 / 1_000_000   # notional $ per input token
PRICE_OUT_PER_TOKEN = 0.60 / 1_000_000  # notional $ per output token


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be non-negative")
    return input_tokens * PRICE_IN_PER_TOKEN + output_tokens * PRICE_OUT_PER_TOKEN


@dataclass
class CallResult:
    prompt: str
    output: str
    input_tokens: int
    output_tokens: int
    content_tokens: int | None
    cost_usd: float
    latency_s: float
    done_reason: str | None
    truncated: bool
    cache_suspected: bool

    def to_dict(self) -> dict:
        return asdict(self)


class OllamaBackend:
    """Real backend: POST /api/chat (non-streaming)."""

    def __init__(self, host: str = "http://localhost:11434", timeout: float = 600):
        self.url = host.rstrip("/") + "/api/chat"
        self.timeout = timeout

    def chat(self, payload: dict) -> dict:
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


class FakeBackend:
    """Test backend: returns canned responses (list, in order) or calls a function(payload)."""

    def __init__(self, responses):
        self._responses = responses if callable(responses) else list(responses)
        self.payloads: list[dict] = []

    def chat(self, payload: dict) -> dict:
        self.payloads.append(payload)
        if callable(self._responses):
            return self._responses(payload)
        return self._responses.pop(0)


class TokenCounter:
    def __init__(self, backend, model: str, system: str, options: dict, keep_alive=None):
        self.backend = backend
        self.model = model
        self.system = system
        self.options = dict(options)
        self.keep_alive = keep_alive
        self.overhead_tokens: int | None = None
        self._first_seen: dict[tuple[str, str], int] = {}

    def _payload(self, prompt: str) -> dict:
        p = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": dict(self.options),
        }
        if self.keep_alive is not None:
            p["keep_alive"] = self.keep_alive
        return p

    def call(self, prompt: str) -> CallResult:
        payload = self._payload(prompt)
        t0 = time.perf_counter()
        resp = self.backend.chat(payload)
        latency = time.perf_counter() - t0

        inp = int(resp.get("prompt_eval_count") or 0)
        out = int(resp.get("eval_count") or 0)
        done_reason = resp.get("done_reason")

        key = (self.system, prompt)
        cache_suspected = inp == 0
        if key in self._first_seen:
            if inp < self._first_seen[key]:
                cache_suspected = True
        elif inp > 0:
            self._first_seen[key] = inp

        content = None if self.overhead_tokens is None else inp - self.overhead_tokens
        return CallResult(
            prompt=prompt,
            output=resp.get("message", {}).get("content", ""),
            input_tokens=inp,
            output_tokens=out,
            content_tokens=content,
            cost_usd=cost_usd(inp, out),
            latency_s=latency,
            done_reason=done_reason,
            truncated=done_reason == "length",
            cache_suspected=cache_suspected,
        )

    def measure_baseline(self, tiny_prompt: str = "") -> int:
        """Input tokens for a tiny prompt = template + system-prompt overhead (+ the tiny prompt)."""
        r = self.call(tiny_prompt)
        self.overhead_tokens = r.input_tokens
        return r.input_tokens


def check_cache(tc: TokenCounter, prompts, repeats: int = 2) -> list[dict]:
    """Send each prompt `repeats` times in a row; report whether prompt_eval_count stays the same."""
    rows = []
    for prompt in prompts:
        results = [tc.call(prompt) for _ in range(repeats)]
        counts = [r.input_tokens for r in results]
        rows.append({
            "prompt": prompt,
            "counts": counts,
            "latencies_s": [round(r.latency_s, 3) for r in results],
            "stable": len(set(counts)) == 1,
            "cache_suspected": any(r.cache_suspected for r in results),
        })
    return rows
