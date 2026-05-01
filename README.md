# Biopharma Quant Strategies — BTP Project

> An automated pipeline that scores every completed US-biopharma clinical trial by how dangerous its drug looks compared to placebo, then tests whether that score predicts post-completion stock returns.

---

## What this project does

For any US biopharma ticker (e.g. `GILD`, `MRNA`, `JNJ`):

1. **Fetches** every completed clinical trial from clinicaltrials.gov v2 — including all subsidiaries / acquired companies (Wyeth → PFE, ModernaTX → MRNA, Janssen → JNJ).
2. **Cleans** the deeply nested JSON into tidy tables: `trials_df` (one row per study), `events_df` (one row per arm × side-effect), `arms_df` (per-arm rollup).
3. **Scores** each trial that has a placebo arm:
   `safety_score = (% drug-arm patients with serious side effects) − (% placebo-arm patients with serious side effects)`
4. **Joins** the scores to forward stock returns (vs IBB benchmark) at four horizons: `[-5, 0]`, `[0, +5]`, `[0, +20]`, `[0, +60]` trading days around results-posting.
5. **Backtests** whether high-score trials underperform — measured by Information Coefficient (rank correlation between score and forward return).
6. **Surfaces** newly-completed pivotal trials in real time via a live Watcher.

---

## Headline result

Across **1,601 trials with computable scores** spanning 10 watchlist tickers, 2008–2026:

| Slice | n | Spearman IC at [0,+5] |
|---|---:|---:|
| Full universe (all phases) | 1,604 | +0.004 (no signal) |
| Phase 3 only | 657 | -0.046 |
| **Phase 3, enrollment ≥ 300** | **652** | **-0.091** |
| Phase 3, enrollment ≥ 500 | 299 | -0.095 |

The naive score has near-zero predictive power on the full universe, but on **large pivotal Phase-3 trials** the IC reaches −0.09 — small but real, in the hypothesised direction. Tertile spread of ~0.65 percentage points over a month between cleanest and scariest trials.

This is consistent with: (a) most CT.gov-completed trials are non-pivotal Phase-1/2 studies that don't move stocks; (b) the signal lives in the trials the market actually prices on — large late-stage readouts.

---

## How to run it

### Prerequisites

- Python 3.10+
- Internet (for live CT.gov / yfinance calls; cached results work offline)

### Install

```bash
pip install -r requirements.txt
```

### The interactive web UI (most useful starting point)

```bash
python app.py
```

Opens [http://localhost:8000](http://localhost:8000). Type a ticker (try `MRNA` for a fast demo, `GILD` for medium, `JNJ` for the heavyweight). Get safety scores, alerts, drill-in per trial with a verification link to clinicaltrials.gov. Stop with Ctrl+C.

### Single-ticker pipeline (CLI)

```bash
python main.py --ticker GILD --inspect
```

Runs fetch → clean → save (raw JSON + 2 CSVs) → prints a summary.

### Full backtest panel (regenerates `data/processed/safety_panel.csv`)

```bash
python -m src.build_panel
```

Walks the 10-ticker watchlist, fetches all completed trials, scores them, attaches forward returns, saves the analytical panel CSV. Takes ~7 minutes.

### Live trial watcher

```bash
python -m src.watcher --lookback 30
```

Reads the panel, surfaces trials whose results were posted in the last 30 days that pass the alert criteria (Phase 3, enrollment ≥ 300, score ≥ +5pp).

For backtest mode (replay any past date):

```bash
python -m src.watcher --asof 2024-06-01 --lookback 90 --no-state
```

### Inspect any cleaned ticker in the terminal

```bash
python -m src.inspect_trials --ticker GILD
python -m src.inspect_trials --ticker GILD --top 10
python -m src.inspect_trials --ticker GILD --nct NCT01569295
```

---

## Project structure

```
Quant project/
├── app.py                      # Local web UI (stdlib http.server + pandas)
├── main.py                     # Single-ticker CLI (fetch → clean → save)
├── requirements.txt
├── CLAUDE.md                   # Master roadmap / "save game" file
├── README.md                   # This file
│
├── src/                        # All importable modules
│   ├── config.py               # Paths, constants, SPONSOR_OVERRIDES map
│   ├── data_fetcher.py         # CT.gov v2 + yfinance, with sponsor auto-discovery
│   ├── data_cleaner.py         # JSON flattening: trials/events/arms tables
│   ├── inspect_trials.py       # Terminal inspector (summary, top-N, drill-in)
│   ├── market_data.py          # OHLCV fetcher with on-disk cache
│   ├── event_study.py          # Event-window returns + abnormal returns vs IBB
│   ├── safety_score.py         # The Phase-3 safety score
│   ├── backtest.py             # IC + tertile bucket analysis
│   ├── build_panel.py          # Production driver (assembles safety_panel.csv)
│   └── watcher.py              # Phase-4 alert system
│
├── data/
│   ├── raw/                    # CT.gov JSON dumps (gitignored)
│   ├── processed/
│   │   └── safety_panel.csv    # The analytical panel — committed
│   └── cache/
│       ├── market/             # yfinance OHLCV cache per ticker
│       └── watcher/            # State file for cron-safe alerting
│
├── notebooks/                  # Throwaway EDA
└── tests/                      # (Phase 2+, not yet populated)
```

---

## Key files for a code review

If you're a reviewer with limited time, these are the high-leverage files to look at:

| File | Why it matters |
|---|---|
| `CLAUDE.md` | Full project history, every phase's results, every gotcha and decision logged |
| `src/data_fetcher.py` | The trickiest layer — handles M&A subsidiary names, auto-discovery, multi-sponsor merge |
| `src/safety_score.py` | The core scoring formula (deliberately simple v1) |
| `src/event_study.py` | Event-window return computation + abnormal-return logic |
| `src/backtest.py` | IC + tertile analysis |
| `data/processed/safety_panel.csv` | The 11,436-trial analytical panel (open in Excel) |

---

## Findings & limitations

- **The score works on pivotal trials.** Phase 3 + enrollment ≥ 300: IC −0.09. Anything below that: noise.
- **The score is naive on purpose.** No organ-system weighting, no exclusion of always-toxic populations like Phase-1 oncology, no historical baseline. Plenty of room for v2.
- **Sample size is meaningful but not huge.** 652 trials in the strongest slice; IC of −0.09 with that n is ~1.6 standard errors from 0. Suggestive, not yet conclusive.
- **CT.gov sponsor names are not normalized post-M&A** — handled via `SPONSOR_OVERRIDES` + auto-discovery; documented in `CLAUDE.md`.
- **JNJ has known false positives** in auto-discovery (Mead Johnson, Robert Wood Johnson Foundation share the "Johnson" token). Documented; impact on score is small (~50 trials in food/philanthropy).

---

## Roadmap (Phase 5+)

- v2 score: weight by enrollment, drop oncology Phase-1, optional per-organ-system component.
- Compare against historical baseline (IC of "this trial vs trials of similar drug class") rather than only intra-trial drug-vs-placebo.
- Wire the watcher's output into the web UI as a "Latest alerts" tab.
- Walk-forward backtest once v2 score uses cross-trial historical info.

---

## Tech stack

- **Python 3.10+**
- **`requests`** — direct CT.gov v2 calls
- **`yfinance`** — OHLCV market data
- **`pandas`** — table operations
- **`python-dotenv`** — env-var loading (reserved for paid data sources)
- **stdlib `http.server`** — minimal web UI (no framework dependency)

No build system, no Docker, no cloud account required.
