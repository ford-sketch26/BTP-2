"""Production batch builder for the Phase-3 backtest panel.

For each ticker on the watchlist:
  1. Resolve sponsor names (Phase 1.1)
  2. Fetch all completed trials in memory (Phase 1.1)
  3. Flatten + extract arm summaries (Phase 1.2 + 3.0)
  4. Compute safety score per trial (Phase 3.1)
  5. Attach event-window returns vs IBB (Phase 2)
  6. Append to the combined panel

Saves one CSV at the end: ``data/processed/safety_panel.csv``.

Designed to be disk-friendly: raw JSONs are NOT saved (they live only in memory
during their ticker's iteration and are GC'd before the next ticker starts).

Run with:
    python -m src.build_panel
    python -m src.build_panel --tickers GILD,MRNA,PFE --max-pages 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR
from src.data_cleaner import flatten, flatten_arms
from src.data_fetcher import (
    fetch_completed_trials_for_sponsors,
    resolve_sponsor_names,
)
from src.event_study import attach_event_returns
from src.safety_score import compute_safety_score


DEFAULT_WATCHLIST = [
    "GILD", "MRNA", "PFE", "JNJ", "BIIB",
    "LLY",  "ABBV", "BMY", "REGN", "VRTX",
]


def build_for_ticker(ticker: str, max_pages: int | None) -> pd.DataFrame:
    """Run the full pipeline for one ticker. Return a panel slice (one row per NCT)."""
    t0 = time.time()
    names = resolve_sponsor_names(ticker)
    studies = fetch_completed_trials_for_sponsors(names, max_pages=max_pages)
    if not studies:
        return pd.DataFrame()

    trials_df, _ = flatten(studies)
    arms_df = flatten_arms(studies)
    safety_df = compute_safety_score(arms_df)
    returns_df = attach_event_returns(trials_df, ticker=ticker)

    # Combine — start from trials metadata so every NCT is preserved
    keep_trial_cols = [
        "nct_id", "brief_title", "lead_sponsor", "phase", "study_type",
        "enrollment_count", "conditions", "intervention_names",
        "condition_mesh_ids", "condition_mesh_terms",
        "intervention_mesh_ids", "intervention_mesh_terms",
        "completion_date", "results_first_posted",
        "has_adverse_events", "n_arms",
    ]
    panel = trials_df[[c for c in keep_trial_cols if c in trials_df.columns]].copy()
    panel = panel.merge(safety_df, on="nct_id", how="left")
    panel = panel.merge(returns_df, on="nct_id", how="left")
    panel["ticker"] = ticker.upper()  # merge brought one in; overwrite with our canonical value

    elapsed = time.time() - t0
    n_scoreable = panel["safety_score"].notna().sum()
    print(f"  [{ticker}] {len(studies):>5} studies -> {len(panel):>5} panel rows, "
          f"{n_scoreable:>4} with score ({elapsed:.1f}s)")
    return panel


def build_panel(tickers: list[str], max_pages: int | None) -> pd.DataFrame:
    """Iterate the watchlist, concat results."""
    print(f"Building panel for {len(tickers)} tickers (max_pages={max_pages})")
    print()
    parts: list[pd.DataFrame] = []
    for tk in tickers:
        try:
            parts.append(build_for_ticker(tk, max_pages))
        except Exception as exc:  # noqa: BLE001
            print(f"  [{tk}] FAILED: {type(exc).__name__}: {exc}")
    if not parts:
        return pd.DataFrame()
    panel = pd.concat(parts, ignore_index=True)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase-3 backtest panel")
    parser.add_argument(
        "--tickers",
        type=lambda s: [t.strip().upper() for t in s.split(",") if t.strip()],
        default=DEFAULT_WATCHLIST,
        help=f"Comma-separated tickers (default: {','.join(DEFAULT_WATCHLIST)})",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Cap CT.gov pagination per sponsor name (default: no cap)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "safety_panel.csv",
        help="Where to write the combined panel CSV",
    )
    args = parser.parse_args()

    panel = build_panel(args.tickers, args.max_pages)
    if panel.empty:
        print("No data — nothing to save.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output, index=False)

    # Also build the drug-condition index so the deployed app doesn't have to.
    # Building it costs ~40s; ship the CSV instead of recomputing on each cold start.
    try:
        from src.comparator import build_drug_condition_index
        idx = build_drug_condition_index(panel)
        idx_path = args.output.parent / "drug_condition_index.csv"
        idx.to_csv(idx_path, index=False)
        print(f"Drug-condition index saved -> {idx_path} ({len(idx):,} rows)")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not build drug-condition index: {exc}")

    print()
    print("=" * 70)
    print(f"Panel saved -> {args.output}")
    print(f"Total rows: {len(panel):,}")
    print(f"With safety score:           {panel['safety_score'].notna().sum():,}")
    print(f"With abret_0_20:             {panel['abret_0_20'].notna().sum():,}")
    print(f"With BOTH score and return:  {panel.dropna(subset=['safety_score','abret_0_20']).shape[0]:,}")
    print()
    print("Per-ticker breakdown:")
    bd = panel.groupby("ticker").agg(
        n_trials=("nct_id", "count"),
        with_score=("safety_score", lambda s: s.notna().sum()),
        with_return=("abret_0_20", lambda s: s.notna().sum()),
    ).reset_index()
    bd["both"] = panel.dropna(subset=["safety_score", "abret_0_20"]).groupby("ticker").size().reindex(bd["ticker"]).fillna(0).astype(int).values
    print(bd.to_string(index=False))


if __name__ == "__main__":
    main()
