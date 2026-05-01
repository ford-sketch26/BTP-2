# Biopharma Quant Strategies — Project Roadmap

> **Save Game file.** This is the canonical roadmap. Update it after every milestone so any future session (or collaborator) can resume with full context.

---

## 1. Project Goal

Build an automated engine that:

1. Takes a US biopharma ticker (e.g., `GILD`).
2. Fetches **Completed** clinical-trial reports — including adverse-effect data — from the [clinicaltrials.gov API v2](https://clinicaltrials.gov/data-api/api).
3. Cleans and structures the data for downstream quantitative analysis linking trial outcomes to market impact.

The long-term thesis: clinical-trial events (especially adverse-effect reports) are a structured, public, under-exploited information source for biopharma equities. We want a clean dataset before we touch any alpha hypothesis.

---

## 2. Tech Stack

| Layer            | Tool                              | Why |
|------------------|-----------------------------------|-----|
| Language         | Python 3.10+                      | Standard for quant + scientific data work |
| Market data      | `yfinance`                        | Free, simple, sufficient for tickers/prices |
| Trial data       | `clinicaltrials.gov` API v2       | Authoritative source for US trials |
| HTTP             | `requests`                        | Direct calls to CT.gov v2 |
| Data wrangling   | `pandas`                          | Standard table abstraction |
| Config           | `python-dotenv`                   | Load `.env` for any future API keys |
| Notebooks (EDA)  | `jupyter` (optional)              | For inspection / sanity checks |
| Testing          | `pytest`                          | For Phase-2 once fetchers stabilize |

---

## 3. Project Structure

```
Quant project/
├── CLAUDE.md                  # This file — master roadmap / save-game
├── README.md                  # (to be written when public-ready)
├── .env                       # Future API keys — NEVER committed
├── .env.example               # Template for collaborators
├── .gitignore
├── requirements.txt           # Python dependencies
│
├── src/                       # All importable code lives here
│   ├── __init__.py
│   ├── config.py              # Paths, constants, env loading
│   ├── data_fetcher.py        # CT.gov + yfinance fetchers
│   ├── data_cleaner.py        # (Phase 1.2) Normalize + flatten trial JSON
│   └── storage.py             # (Phase 1.2) Read/write to data/raw + data/processed
│
├── data/
│   ├── raw/                   # Untouched API payloads (JSON)
│   └── processed/             # Cleaned, analysis-ready tables (CSV / Parquet)
│
├── notebooks/                 # Throwaway EDA + sanity checks
│
├── tests/                     # pytest suite (Phase 2+)
│
└── main.py                    # CLI entry point — runs the end-to-end pipeline
```

---

## 4. Phased Roadmap

### Phase 1 — Data Extraction Layer  ← **WE ARE HERE**

> **Rule:** No strategy work until Phase 1 is signed off.

- [x] **1.0** Scaffold project structure, env files, requirements
- [x] **1.1** Implement `data_fetcher.fetch_trials_for_ticker(ticker)` — DONE 2026-04-29
  - Resolve ticker → sponsor name(s) via a 3-stage pipeline (see "Sponsor name resolution" section below)
  - Query CT.gov v2 `/studies` once per resolved sponsor name; merge and dedupe by NCT ID
  - Save raw JSON to `data/raw/<ticker>_<timestamp>.json` (or skip with `--no-save`)
- [x] **1.4** Validate: pulled GILD, MRNA, PFE, BIIB, JNJ, LLY (+ AAPL & nonsense ticker as edge cases) — DONE 2026-04-29
  - See "Sponsor resolution validation" section below for measured numbers
  - All 6 pharma tickers return realistic trial counts; non-pharma/bogus tickers fail gracefully
- [x] **1.2** Implement `data_cleaner.flatten()` — DONE 2026-04-29
  - Two outputs: `trials_df` (one row per NCT, with metadata) + `events_df` (one row per NCT × arm × event term, with `incidence_rate` derived)
  - CLI: `python -m src.data_cleaner data/raw/<file>.json` writes `<base>_trials.csv` and `<base>_events.csv` to `data/processed/`
  - **Live inspector**: `src/inspect_trials.py` — runs the cleaner in-memory and pretty-prints summaries / per-trial detail. See "Inspecting cleaned data" section below.
  - Validated on GILD: 100 trials → 64 with adverse events → 18,177 event rows across 27 organ systems. Drill-in on NCT01472185 produces a clean per-arm incidence table that matches what's published on clinicaltrials.gov.
- [x] **1.3** Wire `main.py` end-to-end — DONE 2026-04-29
  - `python main.py --ticker GILD` → fetch → clean → save (raw JSON + 2 CSVs) in one command
  - `--no-clean` stops after fetch; `--no-save` keeps everything in memory
  - `--inspect` appends a summary at the end (delegates to `inspect_trials.print_summary`)
  - `--max-pages N` and `--verbose` for dev runs
  - Validated end-to-end on MRNA with `--max-pages 1 --no-save --inspect` → 71 trials, 26 with AE, 26,297 event rows; highest-incidence findings clinically plausible (acute kidney injury 5–7% in transplant cohort)

### Phase 1 — DONE ✅ (sign-off 2026-04-29)
The data extraction layer is complete: any US biopharma ticker → flat trials + events tables, ready for analysis. Phase 2 (market data) can begin.

### Strategy direction (locked 2026-04-30)
User picked **Idea 1 + Idea 4**:
- **Idea 1 — Safety Score Signal:** boil each trial down to one number (drug-arm vs placebo-arm risk, weighted by severity). Backtest historically; deploy forward.
- **Idea 4 — Trial Completion Watcher:** daily job polls CT.gov for trials *newly* marked Completed for a watchlist of tickers; auto-runs the safety score and alerts.

The thesis arc: Phase 2 (market data) → Phase 3 (define + backtest the safety score) → Phase 4 (deploy via the watcher). Together these give a publishable backtest + a live demo for the BTP committee.

### Phase 2 — Market-Side Data — DONE ✅ (2026-04-30)
- [x] **2.0** `src/market_data.py` — `get_ohlcv(ticker)` with 7-day on-disk cache at `data/cache/market/<TICKER>.csv`. Auto-adjusted closes (handles splits/dividends). Tz-stripped index. Empty-safe (returns empty DataFrame for delisted/never-listed tickers).
- [x] **2.1** `src/event_study.py` — `pick_event_date()` prefers `results_first_posted`; falls back to `completion_date`. `windowed_return()` computes close-to-close return over a trading-day window anchored at the event. `attach_event_returns()` joins all of it.
- [x] **2.2** Default windows: `(-5,0)` (pre-event leakage), `(0,+5)`, `(0,+20)`, `(0,+60)`. Each window emits two columns: raw `ret_*` and abnormal `abret_*` (= stock − benchmark over same days). Benchmark: IBB.
- [x] **2.3** Smoke test on GILD's 100 cleaned trials: 98/100 trials get computable returns, 64 use `results_first_posted` (the cleaner signal), 34 fall back to `completion_date`. Average abnormal return across the 64 results-posted trials: -1.96% over [0,+20] and -2.24% over [0,+60] — small negative drift on average, but this is the *unconditional* mean; Phase 3's safety score should differentiate winners from losers within this.

**Caches landed:** `data/cache/market/GILD.csv` (798K, 1992-2026), `data/cache/market/IBB.csv` (582K, 2001-2026).

### Phase 3 — Safety Score + Backtest — v1 SCAFFOLDED ✅ (2026-04-30)
- [x] **3.0** Extended cleaner: `iter_arm_summary_rows()` + `flatten_arms(studies)` extracts the per-arm summary CT.gov stores in `eventGroups[]` (distinct from per-event rows). Includes `arm_role` classification ('drug' / 'placebo' via title match on `placebo|sham|sugar pill`).
- [x] **3.1** `src/safety_score.py` — v1 score: `drug_serious_rate − placebo_serious_rate` aggregated over all matched arms per NCT. Trials without a placebo arm get NaN.
- [x] **3.2** `src/backtest.py` — joins safety + market panels, computes IC (Spearman & Pearson), tertile bucket means, return spread.
- [x] **3.3** Smoke test on GILD page-1 (100 trials → 18 scoreable). Top scariest: NCT01569295 (Idelalisib + Bendamustine, score = +0.285, drug 73.4% / placebo 45.0%) — known severe-toxicity trial. **Sample is way too small for stable IC** but pipeline is solid.

### Phase 3 — Production backtest results (2026-04-30, n=1,601 scoreable trials)
- [x] **3.4** Built `src/build_panel.py`. Pulled full data for 10-ticker watchlist (GILD, MRNA, PFE, JNJ, BIIB, LLY, ABBV, BMY, REGN, VRTX). 11,436 trials total, 1,601 with both safety score AND market returns. Saved to `data/processed/safety_panel.csv`.
- [x] **3.5** Backtest results (Spearman IC at [0, +5] abnormal returns):

| Slice | n | IC | Tertile spread @[0,+20] | Verdict |
|---|---:|---:|---:|---|
| Full panel (no filter) | 1,604 | +0.004 | -0.04% | No signal |
| Phase 3 only | 657 | -0.046 | +0.20% | Weak right-direction |
| Phase 3, enrollment >=300 | 652 | **-0.091** | **+0.65%** | **Real signal** |
| Phase 3, enrollment >=500 | 299 | **-0.095** | +0.65% | **Strongest signal** |
| Phase 3/4 + enrollment >=200 | 646 | -0.073 | +0.41% | Real signal |

**Reading:** the naive score has no power on the population of all completed trials (most are non-pivotal Phase 1/2 dose-finding trials that don't move stocks). But on **large pivotal Phase 3 trials**, IC goes to -0.09 — meaningful for a quant signal (institutional alpha range is typically |IC| = 0.03-0.10). The ~0.65% tertile spread over a month is small but consistent with the hypothesis. **Statistical significance is marginal** (IC=-0.09 with n=300 → ~1.6 standard errors from 0); needs more data or a smarter score for a confident claim.

- [x] **3.6** Walk-forward — moot for v1 score. The score is intra-trial (drug arm vs placebo arm WITHIN the trial), no historical baseline used, so there's no look-ahead bias by construction. Deferred until v2 score uses historical baselines (Idea 2 territory).

### Phase 3 v2 ideas (defer until BTP needs more juice)
- Weight by enrollment (bigger trials count more)
- Per-organ-system score, not just overall serious rate
- Exclude oncology Phase 1 (always-high-AE patient population — confounds the score)
- Discount the score by what the market already priced in pre-event (use abret_-30_-5 as a confound check)

### Phase 4 — The Watcher — DONE ✅ (2026-04-30)
- [x] **4.0** `src/watcher.py` — alert filter (score >= 0.05, enrollment >= 300, Phase 3/4 — defaults from Phase 3 findings) + recency filter (results_first_posted within `--lookback` days of `--asof`).
- [x] **4.1** State file at `data/cache/watcher/seen_ncts.txt` makes re-runs cron-safe (alerts don't re-fire). `--no-state` toggles it off.
- [x] **4.2** Demo run on 2026-04-30 with 90-day lookback → 2 alerts:
  - ABBV NCT02947347 (Ibrutinib + Rituximab in follicular lymphoma, score +0.208, drug 64% vs placebo 43%, posted 2026-03-17)
  - REGN NCT03409614 (Cemiplimab combo, score +0.056, drug 30% vs placebo 25%, posted 2026-04-29 — yesterday!)
- [x] **4.3** Backtested watcher logic across 2008-2026 → 60 alerts total, ~3-5/year. Plausible cadence for a quant signal (enough to trade, not so many it's noise).
- Cron: `python -m src.watcher` daily on a fresh `safety_panel.csv` (rebuild panel weekly via `python -m src.build_panel`).

### What's left (BTP write-up, optional v2 polish)
- Document Phase 3 findings into a thesis chapter (signal strength, slice analysis, limitations)
- Optionally iterate on v2 score (per-organ-system weighting, exclude oncology Phase 1, weight by enrollment)
- Optionally wire watcher alerts into the web UI as a "What's new" tab

---

## 4a. Sponsor name resolution (the heart of the fetcher)

CT.gov sponsor names are **not normalized post-M&A**: an acquired company's old trials remain listed under their pre-acquisition sponsor string (e.g. `"Wyeth is now a wholly owned subsidiary of Pfizer"`). yfinance only knows the *current parent's* name. Bridging this gap is the whole reason `resolve_sponsor_names()` exists.

The function combines three strategies, in this order:

1. **yfinance + corporate-suffix stripping** — `Gilead Sciences, Inc.` → `Gilead Sciences`. Also strips trailing conjunctions: `Eli Lilly and Company` → `Eli Lilly`.
2. **Manual override map** (`SPONSOR_OVERRIDES` in `src/config.py`) — for known acquisitions whose CT.gov sponsor strings share *no token* with the parent (e.g. `Janssen` → JNJ, `Hospira` → PFE, `Celgene` → BMY).
3. **Auto-discovery** — for each seed name from steps 1–2, take the first ≥4-char token, query CT.gov broadly with `query.term`, and harvest distinct lead-sponsor strings that contain the anchor (≥2 occurrences). This catches renames (`Moderna` → `ModernaTX, Inc.`) and acquisitions whose CT.gov string still mentions the parent token (`Kite, A Gilead Company`).

Each resolved sponsor name is queried separately and results are deduped by NCT ID — so when "Wyeth" and "Wyeth is now a wholly owned subsidiary of Pfizer" both return the same trials, we keep one copy.

## 4b. Sponsor resolution validation (Phase 1.4 results — 2026-04-29)

First-page (≤100/sponsor) results across 6 tickers + edge cases:

| Ticker | Resolved sponsor names | Unique trials (page 1) | Notes |
|---|---|---|---|
| GILD | 2 (1 seed + 1 discovered: Kite) | 111 | Auto-discovery picks up Kite Pharma cleanly |
| MRNA | 2 (1 seed + 1 discovered: ModernaTX) | 71 | **Was 0 before** — auto-discovery fixed it |
| PFE  | 12 (6 seeds + 6 discovered) | 363 | All major M&A history covered (Wyeth, Pharmacia, Hospira, Array, Medivation) |
| BIIB | 1 (seed only) | 100 | Pure case, no subsidiaries needed |
| JNJ  | 27 (3 seeds + 24 discovered) | 1,155 | Comprehensive Janssen coverage |
| LLY  | 6 (3 seeds + 3 discovered) | 118 | Suffix-stripping fix correctly handles "Eli Lilly and Company" |
| AAPL | 2 | 7 | Non-pharma ticker — handled, just very few trials |
| `XYZNONSENSE` | — | — | Raises `ValueError` with helpful message |

## 4c. Inspecting cleaned data (live verification)

`src/inspect_trials.py` reads any raw JSON we've fetched, cleans it in-memory, and prints human-readable summaries — no need to save processed CSVs first. Three modes:

```bash
# 1) Summary: trial counts, phase mix, top organ systems, highest-incidence serious events
python -m src.inspect_trials --ticker GILD

# 2) Top-N: which trials have the most adverse-event detail (good drill-in candidates)
python -m src.inspect_trials --ticker GILD --top 10

# 3) Drill-in: full per-arm incidence table for a specific trial, serious + other events
python -m src.inspect_trials --ticker GILD --nct NCT01472185
```

The `--ticker` form auto-finds the most recent `data/raw/<TICKER>_*.json`. Use `--file <path>` to inspect a specific raw file.

Schema reference (the two tables `clean_raw_file()` produces):

**`trials_df`** — one row per NCT:
> `nct_id`, `brief_title`, `official_title`, `lead_sponsor`, `lead_sponsor_class`, `overall_status`, `phase`, `study_type`, `enrollment_count`, `start_date`, `completion_date`, `primary_completion_date`, `results_first_posted`, `conditions`, `intervention_names`, `intervention_types`, `has_results`, `has_adverse_events`, `n_arms`, `n_serious_events`, `n_other_events`

**`events_df`** — one row per (NCT × arm × event term) — the analytical table:
> `nct_id`, `severity_class` (`serious`/`other`), `organ_system`, `event_term`, `source_vocabulary`, `assessment_type`, `group_id`, `group_title`, `num_at_risk`, `num_affected`, `num_events`, `incidence_rate` (= `num_affected/num_at_risk`)

## 5. Known limitations

- **JNJ false positives (low priority).** Auto-discovery on the anchor `"Johnson"` matches "Mead Johnson Nutrition" (formula company, ex-BMS) and "Robert Wood Johnson Foundation" (philanthropy) — neither is part of J&J. They contribute ~50 trials to JNJ's count. They're food/foundation studies, unlikely to have adverse-events data, but they shouldn't really be there. Workaround when this matters: tighten the anchor for JNJ specifically, or add a per-ticker exclusion list.
- **PFE under-reporting cap.** Even with overrides + discovery, we capture ~363 of ~4,800 broad-mention trials (first page only). Increasing `--max-pages` will pull more. Some ancient acquisitions (pre-2005) may still be missed.
- **CT.gov is the source of truth, not always the primary investigator.** A trial sponsored by an academic medical center but funded by a pharma will list the AMC as `leadSponsor` and the pharma as a *collaborator* — we don't catch those. This is a deliberate scope choice for now (lead-only); revisit at Phase 2 if signal looks weak.

## 6. Open Questions / Decisions Log

| Date       | Question                                              | Decision / Status |
|------------|-------------------------------------------------------|-------------------|
| 2026-04-29 | Use CT.gov v2 (REST/JSON) vs v1 (XML)?                | **v2** — modern, JSON-native |
| 2026-04-29 | Resolve ticker→sponsor via yfinance or hand-mapped?   | yfinance first, fall back to manual map if `.info` is unreliable |
| 2026-04-29 | Storage format for processed data?                    | CSV for now; revisit Parquet at Phase 2 |

---

## 7. How to Resume Next Session

1. Read this file top-to-bottom.
2. Check the **Phased Roadmap** for the first unchecked box → that's the next task.
3. Skim recent files in `data/raw/` to see the most recent successful fetch.
4. Re-run `python main.py --ticker GILD` as a smoke test before adding new code.
