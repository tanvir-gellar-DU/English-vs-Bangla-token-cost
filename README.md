# English vs Bangla token cost (Assignment 2)

Measures how many more tokens (and notional dollars) a Bangla prompt costs than an English prompt
with the same meaning, on `qwen2.5:3b` via Ollama. Python 3.12, standard library only
(matplotlib for the chart).

## Reproduce (Ollama running locally)

```bash
ollama pull qwen2.5:3b                       # model tag used: qwen2.5:3b (ID 357c53fb659c)
python3 -m unittest discover -s tests -t .   # 42 tests, fake backend, no Ollama needed
python3 runner.py                            # 100 calls -> results/results.csv (+ .meta.json), ~5 min
python3 analyze.py                           # -> results/comparison.csv, summary.json, report/ratio_chart.png
```

Optional: `python3 calibrate.py` (overhead + cache check), `python3 probe_words.py` (single-word token costs).
Temperature 0 and seed 42 are fixed, so token counts should match; latency will differ by machine.

## Files

| Path | What |
|---|---|
| `data/dataset.tsv` | 50 aligned English/Bangla pairs (`id, category, english, bangla`) |
| `config.py` | model, system prompt, options — shared by both languages |
| `tokcount.py` | token-counting module: Ollama counts, cost, overhead baseline, cache check |
| `runner.py` | sends all 100 prompts, writes `results/results.csv` |
| `analyze.py` | per-pair + overall ratios, bootstrap CI, per-category table, chart |
| `calibrate.py`, `probe_words.py` | overhead/cache evidence, single-word token probe |
| `tests/` | unit tests (tokcount, dataset, runner, analyze) |
| `results/` | `results.csv`, `comparison.csv`, `summary.json`, `quality_check.csv`, `baseline.json`, `cache_check.json`, `token_probe.json`, `run2/` |
| `report/report.md` | the report (with `ratio_chart.png`) |
| `AI_LOG.md` | coding-assistant log |
