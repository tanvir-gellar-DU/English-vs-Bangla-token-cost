# Assignment 2 — English vs Bangla token cost: task list

Model: `qwen2.5:3b` (Ollama ID `357c53fb659c`, Q4_K_M) · Ollama 0.40.16 · Python 3.12

## Phase 0 — Setup
- [x] T0. Project skeleton (`tests/`, `data/`, `results/`, `report/`, README, AI_LOG)

## Phase 1 — Token-counting system (20 pts)
- [x] T1. Tests first with a fake backend (`tests/test_tokcount.py`)
- [x] T2. `tokcount.py`: Ollama call, token extraction, cost, swappable backend
- [x] T3. Template/system-prompt overhead baseline → `results/baseline.json`
- [x] T4. Prompt-cache check: same prompt twice, compare `prompt_eval_count`; document fix → `results/cache_check.json` (counts stable; warm == fresh)

## Phase 2 — Dataset (20 pts)
- [x] T5. 50 English prompts: question, instruction, explanation, arithmetic, summarise, long
- [x] T6. Bangla translations (drafted by assistant) + human review of every pair (approved by student)
- [x] T7. Validation test for `data/dataset.tsv`

## Phase 3 — Runner (20 pts)
- [x] T8. `runner.py` (temp 0, seed 42, num_predict 256, same system prompt) → `results/results.csv`
- [x] T9. Runner tests with fake backend
- [x] T10. Real run (+ repeat runs for stability)

## Phase 4 — Cost & comparison (20 pts)
- [x] T11. `analyze.py` (per-pair + overall ratios, mean/median/min/max, bootstrap 95% CI, latency, per category) → `results/comparison.csv`, `results/summary.json`
- [x] T12. Chart → `report/ratio_chart.png`

## Phase 5 — Analysis & report (15 pts)
- [ ] T13. Quality check on 10 pairs — drafted in `results/quality_check.csv`; **student must confirm verdicts**
- [x] T14. Surprising-ratio case
- [x] T15. `report/report.md` (1–2 pages)

## Phase 6 — Reproducibility (5 pts)
- [x] T16. README with ≤5 commands
- [ ] T17. Final submission checklist

## Bonus (+5)
- [ ] T18. Second model, tokenizer comparison
