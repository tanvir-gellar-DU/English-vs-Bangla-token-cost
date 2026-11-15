"""T11-T12: per-pair and overall Bangla-vs-English comparison, spread, chart.

Usage: python3 analyze.py [--results results/results.csv]
Writes results/comparison.csv, results/summary.json, report/ratio_chart.png
"""
import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

COMPARISON_COLUMNS = [
    "id", "category", "en_input", "bn_input", "en_content", "bn_content",
    "en_output", "bn_output", "en_cost_usd", "bn_cost_usd",
    "input_ratio", "content_ratio", "output_ratio", "total_cost_ratio",
    "en_latency_s", "bn_latency_s", "en_truncated", "bn_truncated"]

CATEGORY_ORDER = ["question", "instruction", "explanation", "arithmetic", "summarise", "long"]
# Validated categorical palette (dataviz skill reference, light mode), fixed order.
CATEGORY_COLORS = dict(zip(CATEGORY_ORDER,
                           ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))


def _bool(v):
    return v if isinstance(v, bool) else str(v).strip().lower() == "true"


def load_results(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("input_tokens", "content_tokens", "output_tokens"):
            r[k] = int(r[k])
        r["cost_usd"] = float(r["cost_usd"])
        r["latency_s"] = float(r["latency_s"])
        r["truncated"] = _bool(r["truncated"])
    return rows


def _ratio(a, b):
    return a / b if b else float("nan")


def pair_table(results):
    by_id = defaultdict(dict)
    for r in results:
        by_id[str(r["id"])][r["language"]] = r
    pairs = []
    for pid in sorted(by_id, key=int):
        d = by_id[pid]
        if set(d) != {"english", "bangla"}:
            raise ValueError(f"pair {pid} is missing a language: {sorted(d)}")
        en, bn = d["english"], d["bangla"]
        pairs.append({
            "id": pid, "category": en["category"],
            "en_input": en["input_tokens"], "bn_input": bn["input_tokens"],
            "en_content": en["content_tokens"], "bn_content": bn["content_tokens"],
            "en_output": en["output_tokens"], "bn_output": bn["output_tokens"],
            "en_cost_usd": en["cost_usd"], "bn_cost_usd": bn["cost_usd"],
            "input_ratio": _ratio(bn["input_tokens"], en["input_tokens"]),
            "content_ratio": _ratio(bn["content_tokens"], en["content_tokens"]),
            "output_ratio": _ratio(bn["output_tokens"], en["output_tokens"]),
            "total_cost_ratio": _ratio(bn["cost_usd"], en["cost_usd"]),
            "en_latency_s": en["latency_s"], "bn_latency_s": bn["latency_s"],
            "en_truncated": en["truncated"], "bn_truncated": bn["truncated"],
        })
    return pairs


def bootstrap_ci(values, n=10_000, seed=0, level=0.95):
    """Percentile bootstrap CI of the mean."""
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(rng.choices(values, k=k)) / k for _ in range(n))
    lo = means[int((1 - level) / 2 * n)]
    hi = means[min(n - 1, int((1 + level) / 2 * n))]
    return (lo, hi)


def describe(values):
    lo, hi = bootstrap_ci(values)
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values), "ci95_low": lo, "ci95_high": hi}


def overall(results):
    out = {}
    for lang in ("english", "bangla"):
        rs = [r for r in results if r["language"] == lang]
        lat = [r["latency_s"] for r in rs]
        out[lang] = {
            "calls": len(rs),
            "input_tokens": sum(r["input_tokens"] for r in rs),
            "content_tokens": sum(r["content_tokens"] for r in rs),
            "output_tokens": sum(r["output_tokens"] for r in rs),
            "input_cost_usd": sum(r["input_tokens"] for r in rs) * 0.15e-6,
            "output_cost_usd": sum(r["output_tokens"] for r in rs) * 0.60e-6,
            "cost_usd": sum(r["cost_usd"] for r in rs),
            "latency_mean_s": statistics.mean(lat), "latency_median_s": statistics.median(lat),
            "latency_total_s": sum(lat),
            "truncated_calls": sum(r["truncated"] for r in rs),
        }
    en, bn = out["english"], out["bangla"]
    out["input_ratio"] = _ratio(bn["input_tokens"], en["input_tokens"])
    out["content_ratio"] = _ratio(bn["content_tokens"], en["content_tokens"])
    out["output_ratio"] = _ratio(bn["output_tokens"], en["output_tokens"])
    out["total_cost_ratio"] = _ratio(bn["cost_usd"], en["cost_usd"])
    out["latency_ratio"] = _ratio(bn["latency_mean_s"], en["latency_mean_s"])
    return out


def by_category(pairs):
    groups = defaultdict(list)
    for p in pairs:
        groups[p["category"]].append(p)
    rows = []
    for cat in sorted(groups, key=lambda c: CATEGORY_ORDER.index(c) if c in CATEGORY_ORDER else 99):
        ps = groups[cat]
        rows.append({
            "category": cat, "n": len(ps),
            "input_ratio_mean": statistics.mean(p["input_ratio"] for p in ps),
            "content_ratio_mean": statistics.mean(p["content_ratio"] for p in ps),
            "output_ratio_mean": statistics.mean(p["output_ratio"] for p in ps),
            "total_cost_ratio_mean": statistics.mean(p["total_cost_ratio"] for p in ps),
            "bn_latency_over_en_mean": statistics.mean(
                p["bn_latency_s"] / p["en_latency_s"] for p in ps),
        })
    return rows


def write_comparison(pairs, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COMPARISON_COLUMNS)
        w.writeheader()
        for p in pairs:
            w.writerow({k: (round(v, 4) if isinstance(v, float) and k.endswith("ratio") else v)
                        for k, v in p.items()})


def plot(pairs, summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ink, muted, grid = "#1f1f1e", "#6b6a64", "#e4e3de"
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    xs = list(range(len(pairs)))
    ys = [p["content_ratio"] for p in pairs]
    ax.bar(xs, ys, width=0.78, color=[CATEGORY_COLORS[p["category"]] for p in pairs],
           edgecolor="#fcfcfb", linewidth=1)
    ax.axhline(1.0, color=muted, lw=1)
    ax.text(len(pairs) - 0.4, 1.0, " parity", va="center", color=muted, fontsize=8)
    m = summary["per_pair"]["content_ratio"]["mean"]
    ax.axhline(m, color=ink, lw=1, ls=(0, (4, 3)))
    ax.text(len(pairs) - 0.4, m, f" mean {m:.2f}×", va="center", color=ink, fontsize=8)

    # Direct category labels under each group (secondary encoding, not colour alone).
    start = 0
    for i in range(1, len(pairs) + 1):
        if i == len(pairs) or pairs[i]["category"] != pairs[start]["category"]:
            ax.text((start + i - 1) / 2, -0.06, pairs[start]["category"], transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8.5, color=ink)
            if i < len(pairs):
                ax.axvline(i - 0.5, color=grid, lw=1, zorder=0)
            start = i
    ax.set_xticks(xs)
    ax.set_xticklabels([p["id"] for p in pairs], fontsize=6, color=muted)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", colors=muted, labelsize=8)
    ax.set_xlim(-0.6, len(pairs) - 0.4)
    ax.set_ylim(0, max(ys) * 1.1)
    ax.yaxis.grid(True, color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.set_ylabel("Bangla ÷ English input tokens\n(template overhead removed)", color=muted, fontsize=8.5)
    ax.set_title("Bangla prompts need more input tokens than English for the same meaning (qwen2.5:3b)",
                 loc="left", fontsize=11, color=ink)
    ax.legend(handles=[Patch(color=CATEGORY_COLORS[c], label=c) for c in CATEGORY_ORDER],
              ncol=6, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.0))
    fig.text(0.01, 0.01, "Bars = pairs 1–50 (prompt id on axis). Content tokens = prompt_eval_count − "
             f"{summary['overhead_tokens']} fixed template/system tokens.", fontsize=7, color=muted)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--chart", default="report/ratio_chart.png")
    args = ap.parse_args()

    results = load_results(args.results)
    pairs = pair_table(results)
    meta_path = Path(args.results).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    summary = {
        "model": meta.get("model"), "overhead_tokens": meta.get("overhead_tokens"),
        "overall": overall(results),
        "per_pair": {k: describe([p[k] for p in pairs])
                     for k in ("input_ratio", "content_ratio", "output_ratio", "total_cost_ratio")},
        "by_category": by_category(pairs),
        "extremes_content_ratio": {
            "max": max(pairs, key=lambda p: p["content_ratio"]),
            "min": min(pairs, key=lambda p: p["content_ratio"])},
    }
    out = Path(args.outdir)
    out.mkdir(exist_ok=True)
    write_comparison(pairs, out / "comparison.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    Path(args.chart).parent.mkdir(exist_ok=True)
    plot(pairs, summary, args.chart)

    o, pp = summary["overall"], summary["per_pair"]
    print(f"{'':22}{'English':>12}{'Bangla':>12}{'ratio':>8}")
    for k in ("input_tokens", "content_tokens", "output_tokens"):
        print(f"{k:22}{o['english'][k]:>12}{o['bangla'][k]:>12}{o['bangla'][k] / o['english'][k]:>8.2f}")
    print(f"{'cost_usd':22}{o['english']['cost_usd']:>12.6f}{o['bangla']['cost_usd']:>12.6f}"
          f"{o['total_cost_ratio']:>8.2f}")
    print(f"{'latency_mean_s':22}{o['english']['latency_mean_s']:>12.2f}{o['bangla']['latency_mean_s']:>12.2f}"
          f"{o['latency_ratio']:>8.2f}")
    print(f"{'truncated_calls':22}{o['english']['truncated_calls']:>12}{o['bangla']['truncated_calls']:>12}")
    print("\nper-pair ratio        mean  median   min    max   95% CI")
    for k, d in pp.items():
        print(f"{k:20}{d['mean']:6.2f}{d['median']:8.2f}{d['min']:6.2f}{d['max']:7.2f}"
              f"   [{d['ci95_low']:.2f}, {d['ci95_high']:.2f}]")
    print("\nby category      input  content  output  total_cost")
    for c in summary["by_category"]:
        print(f"{c['category']:14}{c['input_ratio_mean']:7.2f}{c['content_ratio_mean']:9.2f}"
              f"{c['output_ratio_mean']:8.2f}{c['total_cost_ratio_mean']:11.2f}")


if __name__ == "__main__":
    main()
