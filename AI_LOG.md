# AI-assistant log

Coding assistant: Claude Code (Claude Opus 5.5). Each entry: what it wrote · which test proves it · what I reviewed.

## T1–T2 token-counting module
- **Wrote:** `tests/test_tokcount.py` (17 tests, run first and failed with ImportError), then `tokcount.py` (cost, OllamaBackend, FakeBackend, TokenCounter with baseline + cache flag).
- **Proved by:** `python3 -m unittest discover -s tests -t .` → 17 OK.
- **Reviewed:** _(fill in: e.g. checked prices 0.15/0.60 per 1M, payload format, cache-detection logic)_

## T3–T4 baseline + cache check
- **Wrote:** `config.py` (fixed settings), `check_cache()` (tests first: 2 new tests failed, then passed), `calibrate.py`.
- **Proved by:** unit tests + `results/baseline.json` (overhead 24 tokens) and `results/cache_check.json` (counts identical over 3 repeats and with keep_alive=0).
- **Reviewed:** _(fill in)_

## T5–T7 dataset
- **Wrote:** `tests/test_dataset.py` first (failed: no file), then drafted all 50 English prompts and Bangla translations in `data/dataset.tsv`.
- **Proved by:** `tests.test_dataset` (50 rows, 6 categories, scripts, length spread, same sentence count per pair).
- **Reviewed:** _(fill in: I read every Bangla pair and changed …)_

## T8–T9 runner
- **Wrote:** `tests/test_runner.py` first (failed: no module), then `runner.py` (alternating language order per pair, warm-up, baseline, results.csv + meta.json).
- **Proved by:** `tests.test_runner` (6 tests) — full suite 33 OK.
- **Reviewed:** _(fill in)_

## T10–T12 run + analysis
- **Wrote:** `tests/test_analyze.py` first (9 tests, failed: no module), then `analyze.py` (pair table, bootstrap CI, overall, by-category, chart); `probe_words.py`.
- **Proved by:** full suite 42 OK; `results/summary.json` totals re-checked by hand (e.g. 2042×0.15e-6 + 1680×0.60e-6 = $0.001314).
- **Reviewed:** _(fill in)_

## T13–T15 quality check + report
- **Wrote:** draft verdicts in `results/quality_check.csv` and `report/report.md`.
- **Proved by:** numbers in report match `results/summary.json`.
- **Reviewed:** _(fill in: I re-read the 10 Bangla answers and agree/changed …)_
