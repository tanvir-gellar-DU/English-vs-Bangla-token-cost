"""T8: send all 50 English + 50 Bangla prompts with identical settings and record every call.

Usage: python3 runner.py [--model qwen2.5:3b] [--out results/results.csv]
"""
import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path

from config import MODEL, OPTIONS, SYSTEM
from tokcount import OllamaBackend, TokenCounter

DATASET = Path(__file__).resolve().parent / "data" / "dataset.tsv"
RESULT_COLUMNS = ["id", "language", "category", "prompt", "output", "input_tokens",
                  "content_tokens", "output_tokens", "cost_usd", "latency_s",
                  "truncated", "done_reason", "cache_suspected"]


def load_dataset(path=DATASET):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def run_all(tc: TokenCounter, rows, progress=False):
    """Alternate which language goes first per pair (odd: en->bn, even: bn->en) to cancel drift."""
    out = []
    for i, row in enumerate(rows):
        langs = ("english", "bangla") if i % 2 == 0 else ("bangla", "english")
        for lang in langs:
            r = tc.call(row[lang])
            out.append({
                "id": row["id"], "language": lang, "category": row["category"],
                "prompt": r.prompt, "output": r.output,
                "input_tokens": r.input_tokens, "content_tokens": r.content_tokens,
                "output_tokens": r.output_tokens, "cost_usd": r.cost_usd,
                "latency_s": round(r.latency_s, 4), "truncated": r.truncated,
                "done_reason": r.done_reason, "cache_suspected": r.cache_suspected,
            })
            if progress:
                print(f"{row['id']:>3} {lang:<7} in={r.input_tokens:<4} out={r.output_tokens:<4} "
                      f"{r.latency_s:5.2f}s{' TRUNC' if r.truncated else ''}"
                      f"{' CACHE?' if r.cache_suspected else ''}", flush=True)
    return out


def write_results(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        w.writeheader()
        w.writerows(results)


def ollama_version(host="http://localhost:11434"):
    try:
        with urllib.request.urlopen(host + "/api/version", timeout=5) as r:
            return json.loads(r.read())["version"]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="results/results.csv")
    args = ap.parse_args()

    tc = TokenCounter(OllamaBackend(), args.model, SYSTEM, OPTIONS)
    tc.call("warm-up")                      # load the model; excluded from results
    overhead = tc.measure_baseline("")      # template + system prompt tokens
    rows = load_dataset()
    t0 = time.time()
    results = run_all(tc, rows, progress=True)
    write_results(results, args.out)

    meta = {"model": args.model, "ollama_version": ollama_version(), "system": SYSTEM,
            "options": OPTIONS, "overhead_tokens": overhead, "calls": len(results),
            "wall_time_s": round(time.time() - t0, 1),
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
            "truncated_calls": sum(r["truncated"] for r in results),
            "cache_suspected_calls": sum(r["cache_suspected"] for r in results)}
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
