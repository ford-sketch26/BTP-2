"""Phase 4 — the Trial Completion Watcher.

The "live product" layer: surfaces newly-completed pivotal trials with
high safety scores, so a user can see *as it happens* which trials are
flashing red.

Per Phase 3 findings, the score has signal mainly on **large Phase 3 trials**
(enrollment >= 300). The watcher applies that filter by default.

Two run modes:
  * **Live** (default): finds trials whose `results_first_posted` is within
    the last `--lookback` days from today, ranks them by safety score, prints
    the alert list.
  * **Backtest** (`--asof YYYY-MM-DD`): same logic but uses ``asof`` as "today".
    Lets you replay what the watcher *would have* alerted on at any past date —
    useful for demo screenshots and for testing the alert logic.

Designed to be cron-safe: re-running it doesn't double-alert because state is
kept in ``data/cache/watcher/seen_ncts.txt``. Pass ``--no-state`` to ignore the
state file (re-show everything in the lookback window).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, PROCESSED_DIR


WATCHER_DIR = DATA_DIR / "cache" / "watcher"
SEEN_PATH = WATCHER_DIR / "seen_ncts.txt"


# ---------------------------- alert filter -----------------------------------

def apply_alert_filter(
    panel: pd.DataFrame,
    min_score: float = 0.05,
    min_enrollment: int = 300,
    phases: tuple[str, ...] = ("PHASE3", "PHASE4", "PHASE2|PHASE3"),
) -> pd.DataFrame:
    """Trim ``panel`` to alert-worthy trials.

    Defaults align with Phase 3 findings: large pivotal trials, score above a
    threshold (drug arm at least 5 percentage points worse than placebo).
    """
    df = panel.copy()
    df = df[df["safety_score"].notna()]
    df = df[df["safety_score"] >= min_score]
    df = df[df["enrollment_count"].fillna(0) >= min_enrollment]
    df = df[df["phase"].isin(phases)]
    return df


# ---------------------------- date filter ------------------------------------

def filter_recent_completions(
    panel: pd.DataFrame,
    asof: pd.Timestamp,
    lookback_days: int = 30,
) -> pd.DataFrame:
    """Keep trials whose `results_first_posted` is within ``lookback_days`` of asof.

    Falls back to ``completion_date`` for trials without posted results.
    """
    df = panel.copy()
    rfp = pd.to_datetime(df["results_first_posted"], errors="coerce")
    cd = pd.to_datetime(df["completion_date"], errors="coerce")
    event = rfp.fillna(cd)
    df["_event"] = event

    cutoff = asof - pd.Timedelta(days=lookback_days)
    df = df[(df["_event"] > cutoff) & (df["_event"] <= asof)]
    return df.drop(columns=["_event"])


# ---------------------------- state file -------------------------------------

def load_seen() -> set[str]:
    if not SEEN_PATH.exists():
        return set()
    return {line.strip() for line in SEEN_PATH.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_seen(ncts: list[str]) -> None:
    if not ncts:
        return
    WATCHER_DIR.mkdir(parents=True, exist_ok=True)
    with SEEN_PATH.open("a", encoding="utf-8") as f:
        for n in ncts:
            f.write(f"{n}\n")


# ---------------------------- formatting -------------------------------------

def _format_row(r: pd.Series) -> str:
    title = (str(r.get("brief_title") or "") or "")[:55]
    return (
        f"  [{r['ticker']:<5}] {r['nct_id']:<12}  score={r['safety_score']:+.3f}  "
        f"drug={r['drug_rate']*100:5.1f}%  placebo={r['placebo_rate']*100:5.1f}%  "
        f"phase={r['phase']:<8}  N={int(r['enrollment_count']):<5}  "
        f"results_posted={r.get('results_first_posted','?')}\n"
        f"          \"{title}\"\n"
        f"          https://clinicaltrials.gov/study/{r['nct_id']}"
    )


def format_report(alerts: pd.DataFrame, asof: pd.Timestamp, lookback: int) -> str:
    if alerts.empty:
        return f"=== TRIAL WATCHER ({asof.date()}, looking back {lookback}d) ===\nNo alert-worthy completions in window."
    lines = [f"=== TRIAL WATCHER ({asof.date()}, looking back {lookback}d) ==="]
    lines.append(f"{len(alerts)} alert(s):\n")
    # Sort by score descending so the scariest is first
    for _, r in alerts.sort_values("safety_score", ascending=False).iterrows():
        lines.append(_format_row(r))
        lines.append("")
    return "\n".join(lines)


# ---------------------------- top-level run ----------------------------------

def run_watcher(
    panel: pd.DataFrame,
    asof: pd.Timestamp,
    lookback_days: int = 30,
    use_state: bool = True,
    min_score: float = 0.05,
    min_enrollment: int = 300,
) -> tuple[pd.DataFrame, str]:
    """Compose the full pipeline. Returns (alerts_df, formatted_report)."""
    # Need brief_title for the report — ensure it's there or backfill blank
    if "brief_title" not in panel.columns:
        panel = panel.copy()
        panel["brief_title"] = ""

    recent = filter_recent_completions(panel, asof, lookback_days)
    alerts = apply_alert_filter(recent, min_score=min_score, min_enrollment=min_enrollment)

    if use_state:
        seen = load_seen()
        alerts = alerts[~alerts["nct_id"].isin(seen)]

    report = format_report(alerts, asof, lookback_days)

    if use_state and not alerts.empty:
        append_seen(alerts["nct_id"].tolist())
    return alerts, report


# ---------------------------- CLI --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Trial Completion Watcher (Phase 4)")
    parser.add_argument(
        "--panel",
        default=str(PROCESSED_DIR / "safety_panel.csv"),
        help="Path to the panel CSV produced by `python -m src.build_panel`",
    )
    parser.add_argument(
        "--asof",
        default=None,
        help="Reference date (YYYY-MM-DD). Default: today (live mode).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=30,
        help="Look back this many days from asof (default: 30)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.05,
        help="Alert threshold for safety score (default: 0.05 = drug 5pp worse than placebo)",
    )
    parser.add_argument(
        "--min-enrollment",
        type=int,
        default=300,
        help="Skip trials with enrollment below this (default: 300)",
    )
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="Ignore the seen-NCTs cache and re-show everything in the window",
    )
    args = parser.parse_args()

    panel_path = Path(args.panel)
    if not panel_path.exists():
        sys.exit(
            f"No panel found at {panel_path}. Run `python -m src.build_panel` first."
        )

    panel = pd.read_csv(panel_path)
    # We also want brief_title for the report; pull from trials_df if not in panel
    # build_panel currently doesn't include it — add a quick lookup if missing.
    if "brief_title" not in panel.columns:
        # Best effort: leave blank
        panel["brief_title"] = ""

    asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp(datetime.utcnow().date())

    alerts, report = run_watcher(
        panel,
        asof=asof,
        lookback_days=args.lookback,
        use_state=not args.no_state,
        min_score=args.min_score,
        min_enrollment=args.min_enrollment,
    )
    print(report)


if __name__ == "__main__":
    main()
