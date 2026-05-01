"""Live inspector for cleaned trial data.

Run this any time you want to verify the cleaner is producing sensible output.

Examples
--------
# Summarize the most recent GILD pull (cleans on the fly from data/raw/)
python -m src.inspect_trials --ticker GILD

# Top 10 trials by adverse-event row count
python -m src.inspect_trials --ticker GILD --top 10

# Drill into a specific trial — see arms side-by-side with serious + other events
python -m src.inspect_trials --ticker GILD --nct NCT01472185

# Inspect any raw file directly
python -m src.inspect_trials --file data/raw/GILD_20260428T210827Z.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR
from src.data_cleaner import clean_raw_file


# ----------------------------- locating files --------------------------------

def latest_raw_file_for(ticker: str) -> Path:
    matches = sorted(RAW_DIR.glob(f"{ticker.upper()}_*.json"))
    if not matches:
        raise FileNotFoundError(
            f"No raw file found for {ticker.upper()} in {RAW_DIR}. "
            f"Run `python main.py --ticker {ticker.upper()}` first."
        )
    return matches[-1]


# ----------------------------- summaries -------------------------------------

def _hr(label: str = "", char: str = "=", width: int = 70) -> None:
    if label:
        pad = max(width - len(label) - 2, 4)
        print(f"\n{char * (pad // 2)} {label} {char * (pad - pad // 2)}")
    else:
        print(char * width)


def print_summary(trials_df: pd.DataFrame, events_df: pd.DataFrame, source: Path) -> None:
    """High-level numbers — what's in the file, what's covered, what looks weird."""
    _hr("TRIALS")
    print(f"Source:                 {source}")
    print(f"Total trials:           {len(trials_df)}")
    print(f"With results posted:    {trials_df['has_results'].sum()}")
    print(f"With adverse events:    {trials_df['has_adverse_events'].sum()}")

    if not trials_df.empty:
        # By phase
        phase_counts = trials_df["phase"].fillna("(none)").value_counts()
        print("\nBy phase:")
        for phase, n in phase_counts.head(10).items():
            print(f"  {n:>5}  {phase}")

        # Sponsors actually present (proves the multi-sponsor fetcher worked)
        spons = trials_df["lead_sponsor"].fillna("(none)").value_counts()
        print("\nLead sponsors represented:")
        for s, n in spons.head(10).items():
            print(f"  {n:>5}  {s}")

        # Date range
        cd = pd.to_datetime(trials_df["completion_date"], errors="coerce")
        if cd.notna().any():
            print(f"\nCompletion date range:  {cd.min().date()}  ...  {cd.max().date()}")

    _hr("ADVERSE EVENTS")
    if events_df.empty:
        print("No adverse-event rows.")
        return

    print(f"Total event rows:       {len(events_df):,}")
    print(f"Distinct trials:        {events_df['nct_id'].nunique()}")
    print(f"Distinct organ systems: {events_df['organ_system'].nunique()}")
    print(f"Serious events:         {(events_df['severity_class']=='serious').sum():,}")
    print(f"Other events:           {(events_df['severity_class']=='other').sum():,}")

    # Top organ systems by row count
    print("\nTop 10 organ systems (by row count):")
    top_organ = events_df["organ_system"].fillna("(none)").value_counts().head(10)
    for organ, n in top_organ.items():
        print(f"  {n:>6,}  {organ}")

    # Highest-incidence rows (only where denominator is meaningful)
    serious = events_df[
        (events_df["severity_class"] == "serious")
        & (events_df["num_at_risk"].fillna(0) >= 30)
        & (events_df["incidence_rate"].notna())
    ].sort_values("incidence_rate", ascending=False)
    if not serious.empty:
        print("\nHighest-incidence SERIOUS events (>=30 at risk):")
        cols = ["nct_id", "group_title", "organ_system", "event_term", "num_affected", "num_at_risk", "incidence_rate"]
        for _, r in serious.head(10).iterrows():
            print(
                f"  {r['nct_id']}  {r['group_title'][:25]:25s}  "
                f"{(r['organ_system'] or '')[:30]:30s}  "
                f"{(r['event_term'] or '')[:35]:35s}  "
                f"{int(r['num_affected']):>4}/{int(r['num_at_risk']):<4}  "
                f"= {r['incidence_rate']*100:5.1f}%"
            )


def print_top_trials(trials_df: pd.DataFrame, events_df: pd.DataFrame, n: int) -> None:
    """Trials with the most adverse-event detail (good drill-in candidates)."""
    if events_df.empty:
        print("No event rows to rank.")
        return
    counts = events_df.groupby("nct_id").size().sort_values(ascending=False).head(n)
    title_lookup = trials_df.set_index("nct_id")["brief_title"].to_dict() if not trials_df.empty else {}
    _hr(f"TOP {n} TRIALS BY ADVERSE-EVENT ROW COUNT")
    print(f"  {'NCT':<14}  {'rows':>6}  title")
    for nct, c in counts.items():
        title = (title_lookup.get(nct) or "")[:80]
        print(f"  {nct:<14}  {c:>6}  {title}")


def print_trial_detail(
    trials_df: pd.DataFrame, events_df: pd.DataFrame, nct: str
) -> None:
    """Pretty-print everything we know about one specific trial."""
    nct = nct.upper()
    trial = trials_df[trials_df["nct_id"] == nct]
    if trial.empty:
        print(f"NCT {nct} not found in this dataset.")
        return
    t = trial.iloc[0]

    _hr(f"{nct}")
    print(f"  Title:           {t['brief_title']}")
    print(f"  Lead sponsor:    {t['lead_sponsor']} ({t['lead_sponsor_class']})")
    print(f"  Phase / type:    {t['phase']} / {t['study_type']}")
    print(f"  Enrollment:      {t['enrollment_count']}")
    print(f"  Start  / Compl.: {t['start_date']}  ->  {t['completion_date']}")
    print(f"  Conditions:      {t['conditions']}")
    print(f"  Interventions:   {t['intervention_names']}")
    print(f"  Has AE module:   {t['has_adverse_events']}  ({t['n_arms']} arms, "
          f"{t['n_serious_events']} serious, {t['n_other_events']} other event terms)")

    sub = events_df[events_df["nct_id"] == nct]
    if sub.empty:
        print("\n  (No adverse-event rows for this trial)")
        return

    arms = sub[["group_id", "group_title"]].drop_duplicates().sort_values("group_id")
    _hr("ARMS")
    for _, a in arms.iterrows():
        print(f"  {a['group_id']}  {a['group_title']}")

    for severity in ("serious", "other"):
        sev = sub[sub["severity_class"] == severity]
        if sev.empty:
            continue
        _hr(f"{severity.upper()} EVENTS")
        # Pivot: one row per (organ system, event), columns per arm showing affected/at-risk
        pivot = sev.pivot_table(
            index=["organ_system", "event_term"],
            columns="group_title",
            values="incidence_rate",
            aggfunc="first",
        )
        # Format as "x.x%"
        pivot_fmt = pivot.map(lambda v: f"{v*100:5.1f}%" if pd.notna(v) else "   .  ")
        # Truncate organ system for readability
        pivot_fmt.index = pd.MultiIndex.from_tuples(
            [(str(o)[:30], str(e)[:40]) for o, e in pivot_fmt.index],
            names=["organ_system", "event_term"],
        )
        with pd.option_context("display.max_rows", 200, "display.max_colwidth", 50, "display.width", 200):
            print(pivot_fmt.to_string())


# ----------------------------- CLI -------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect cleaned trial data live")
    src_grp = parser.add_mutually_exclusive_group(required=True)
    src_grp.add_argument("--ticker", help="Inspect the latest data/raw/<TICKER>_*.json")
    src_grp.add_argument("--file", help="Path to a specific raw JSON file")

    parser.add_argument("--nct", help="Drill into a specific NCT id")
    parser.add_argument("--top", type=int, help="Show the top N trials by event-row count")

    args = parser.parse_args()

    raw_path = Path(args.file) if args.file else latest_raw_file_for(args.ticker)
    trials_df, events_df, _ = clean_raw_file(raw_path, save=False)

    if args.nct:
        print_trial_detail(trials_df, events_df, args.nct)
    elif args.top:
        print_summary(trials_df, events_df, raw_path)
        print_top_trials(trials_df, events_df, args.top)
    else:
        print_summary(trials_df, events_df, raw_path)


if __name__ == "__main__":
    main()
