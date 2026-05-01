"""CLI entry point for the Biopharma Quant Strategies pipeline.

Default behavior: fetch -> clean -> save (raw JSON + trials CSV + events CSV).
Flags let you skip stages or skip disk writes when needed.

Examples
--------
# Full pipeline, default (fetch + clean + save 3 files)
python main.py --ticker GILD

# Dev run: cap to first page per sponsor, print summary, don't write to disk
python main.py --ticker GILD --max-pages 1 --inspect --no-save

# Just fetch, skip cleaning (raw JSON only)
python main.py --ticker GILD --no-clean

# Verbose: see every resolved sponsor name and per-sponsor trial count
python main.py --ticker GILD --verbose
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from src.config import RAW_DIR
from src.data_cleaner import flatten, save_processed
from src.data_fetcher import (
    fetch_completed_trials_for_sponsors,
    resolve_sponsor_names,
    save_raw_payload,
)
from src.inspect_trials import print_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Biopharma Quant Strategies pipeline (fetch + clean)"
    )
    parser.add_argument("--ticker", required=True, help="US biopharma ticker, e.g. GILD")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Cap CT.gov pagination per sponsor (useful for dev runs)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Stop after fetching raw JSON; skip the cleaner stage",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run everything in memory, write nothing to disk (useful when disk-constrained)",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print a summary of cleaned data at the end (top organ systems, risk events, etc.)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print resolved sponsors and per-sponsor trial counts during fetch",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    save = not args.no_save

    # ---- Stage 1: fetch ------------------------------------------------------
    print(f"[1/3] Fetching completed trials for {ticker}...")
    sponsor_names = resolve_sponsor_names(ticker, verbose=args.verbose)
    studies = fetch_completed_trials_for_sponsors(
        sponsor_names, max_pages=args.max_pages, verbose=args.verbose
    )
    print(f"      Got {len(studies)} unique trials across {len(sponsor_names)} sponsor name(s).")

    raw_path = None
    if save:
        raw_path = save_raw_payload(ticker, studies)
        print(f"      Saved raw JSON -> {raw_path}")
    else:
        # Still need a stable base name for any downstream save (won't be used
        # since save=False, but keeps the function signature simple).
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        raw_path = RAW_DIR / f"{ticker}_{timestamp}.json"

    if args.no_clean:
        print("[2/3] Skipped (--no-clean)")
        print("[3/3] Skipped")
        return

    # ---- Stage 2: clean ------------------------------------------------------
    print(f"[2/3] Flattening {len(studies)} studies into trials + events tables...")
    trials_df, events_df = flatten(studies)
    n_with_ae = int(trials_df["has_adverse_events"].sum()) if not trials_df.empty else 0
    print(
        f"      trials_df: {len(trials_df):>5} rows  ({n_with_ae} with adverse events)"
    )
    print(f"      events_df: {len(events_df):>5} rows")

    # ---- Stage 3: save -------------------------------------------------------
    if save:
        trials_path, events_path = save_processed(trials_df, events_df, raw_path.stem)
        print(f"[3/3] Saved trials  -> {trials_path}")
        print(f"      Saved events  -> {events_path}")
    else:
        print("[3/3] Skipped (--no-save)")

    if args.inspect:
        print_summary(trials_df, events_df, raw_path)


if __name__ == "__main__":
    main()
