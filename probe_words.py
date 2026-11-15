"""Token cost of single words/numbers (used to explain extreme ratios). Writes results/token_probe.json."""
import json

from config import MODEL, OPTIONS, SYSTEM
from tokcount import OllamaBackend, TokenCounter

PAIRS = [("inflation", "মুদ্রাস্ফীতি"), ("sunlight", "সূর্যালোক"), ("Japan", "জাপান"),
         ("the", "এবং"), ("15 + 27", "১৫ + ২৭"), ("1000", "১০০০"), ("Hello", "হ্যালো")]


def main():
    tc = TokenCounter(OllamaBackend(), MODEL, SYSTEM, {**OPTIONS, "num_predict": 1})
    overhead = tc.measure_baseline("")
    rows = []
    for en, bn in PAIRS:
        e, b = tc.call(en).input_tokens - overhead, tc.call(bn).input_tokens - overhead
        rows.append({"english": en, "en_tokens": e, "en_bytes": len(en.encode()),
                     "bangla": bn, "bn_tokens": b, "bn_bytes": len(bn.encode()), "ratio": round(b / e, 2)})
        print(rows[-1])
    with open("results/token_probe.json", "w", encoding="utf-8") as f:
        json.dump({"overhead": overhead, "rows": rows}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
