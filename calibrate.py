"""T3 + T4: measure template/system overhead and check prompt caching. Writes results/*.json."""
import json
from pathlib import Path

from config import MODEL, OPTIONS, SYSTEM
from tokcount import OllamaBackend, TokenCounter, check_cache

OUT = Path("results")


def main():
    OUT.mkdir(exist_ok=True)
    backend = OllamaBackend()
    tc = TokenCounter(backend, MODEL, SYSTEM, OPTIONS)
    tc.call("warm-up")  # load model so later latencies exclude load time

    # --- T3 baseline ---
    overhead = tc.measure_baseline("")  # empty user message = template + system prompt only
    tiny = {p: tc.call(p).input_tokens for p in ["Hi", "হাই", "Hello", "হ্যালো"]}
    baseline = {
        "model": MODEL, "system": SYSTEM, "options": OPTIONS,
        "overhead_tokens_empty_user_message": overhead,
        "tiny_prompts_input_tokens": tiny,
        "tiny_prompts_content_tokens": {p: n - overhead for p, n in tiny.items()},
        "note": "input_tokens in results include this fixed overhead; "
                "content_tokens = input_tokens - overhead.",
    }
    (OUT / "baseline.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=2))
    print(json.dumps(baseline, ensure_ascii=False, indent=2))

    # --- T4 cache check: model kept loaded, same prompt 3x in a row ---
    prompts = ["What is the capital of France?", "ফ্রান্সের রাজধানী কী?",
               "Explain photosynthesis in one sentence.", "এক বাক্যে সালোকসংশ্লেষণ ব্যাখ্যা করো।"]
    warm = check_cache(tc, prompts, repeats=3)

    # fresh state: unload model after every call
    fresh_tc = TokenCounter(backend, MODEL, SYSTEM, OPTIONS, keep_alive=0)
    fresh = check_cache(fresh_tc, prompts[:2], repeats=2)

    same = all(w["counts"][0] == f["counts"][0] for w, f in zip(warm, fresh))
    cache = {"warm_model_repeats": warm, "fresh_model_keep_alive_0": fresh,
             "warm_equals_fresh": same,
             "all_stable": all(r["stable"] for r in warm + fresh)}
    (OUT / "cache_check.json").write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    print(json.dumps(cache, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
