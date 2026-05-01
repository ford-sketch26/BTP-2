"""Phase 3 backtest — does the safety score predict forward returns?

Joins:
  * `safety_df` from `safety_score.compute_safety_score`  (one row per NCT)
  * `event_returns_df` from `event_study.attach_event_returns` (one row per NCT)

Outputs:
  * Joined panel (one row per NCT with both score and forward returns)
  * Quantile bucket statistics (e.g., bottom-third score vs top-third score)
  * Information Coefficient (Spearman rank correlation) per horizon

A positive IC at, say, [0,+20] means: trials with high safety score (drug looks
*worse* than placebo) tend to have high abnormal returns. We expect the
*opposite* sign — high score should predict *negative* future returns. So a
*negative* IC is the hypothesis-confirming direction.
"""

from __future__ import annotations

import pandas as pd

from src.event_study import DEFAULT_WINDOWS, _abret_col, _ret_col


def join_panel(
    safety_df: pd.DataFrame,
    event_returns_df: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join safety + returns by nct_id."""
    if safety_df.empty or event_returns_df.empty:
        return pd.DataFrame()
    return safety_df.merge(event_returns_df, on="nct_id", how="inner")


def quantile_buckets(
    panel: pd.DataFrame,
    return_col: str,
    n_buckets: int = 3,
    score_col: str = "safety_score",
) -> pd.DataFrame:
    """Bucket trials into N quantiles by safety score; report mean return per bucket.

    Bucket 1 = lowest score (cleanest safety profile)
    Bucket N = highest score (drug looks scariest)

    If the hypothesis holds, mean return should DECREASE from bucket 1 to bucket N.
    """
    df = panel.dropna(subset=[score_col, return_col]).copy()
    if df.empty:
        return pd.DataFrame()
    df["bucket"] = pd.qcut(
        df[score_col], q=n_buckets, labels=list(range(1, n_buckets + 1)),
        duplicates="drop",
    )
    out = (
        df.groupby("bucket", observed=True)[return_col]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    out["mean_pct"] = out["mean"] * 100
    out["median_pct"] = out["median"] * 100
    return out


def information_coefficient(
    panel: pd.DataFrame,
    return_col: str,
    score_col: str = "safety_score",
) -> dict:
    """Spearman rank correlation between score and forward return.

    Negative IC = hypothesis confirmed (worse safety -> lower returns).
    """
    df = panel.dropna(subset=[score_col, return_col])
    if len(df) < 5:
        return {"n": len(df), "ic_spearman": None, "ic_pearson": None}
    # Spearman = Pearson on ranks. Computing manually avoids the scipy dependency
    # that pandas' method='spearman' triggers.
    ranked_score = df[score_col].rank()
    ranked_ret = df[return_col].rank()
    rho = ranked_score.corr(ranked_ret, method="pearson")
    pearson = df[score_col].corr(df[return_col], method="pearson")
    return {
        "n": int(len(df)),
        "ic_spearman": float(rho) if pd.notna(rho) else None,
        "ic_pearson": float(pearson) if pd.notna(pearson) else None,
    }


def summarise(panel: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-window backtest summary.

    Columns: window, n, ic_spearman, ic_pearson, bucket1_mean, bucket3_mean, spread.
    spread = bucket1_mean - bucketN_mean (positive = hypothesis holds).
    """
    rows = []
    for s, e in DEFAULT_WINDOWS:
        for kind, col_fn in [("raw", _ret_col), ("abnormal", _abret_col)]:
            col = col_fn(s, e)
            if col not in panel.columns:
                continue
            ic = information_coefficient(panel, col)
            buckets = quantile_buckets(panel, col)
            if buckets.empty or len(buckets) < 2:
                row = {
                    "window": f"[{s}, {e}]",
                    "kind": kind,
                    "n": ic["n"],
                    "ic_spearman": ic.get("ic_spearman"),
                    "ic_pearson": ic.get("ic_pearson"),
                    "bucket1_mean_pct": None,
                    "bucketN_mean_pct": None,
                    "spread_pct": None,
                }
            else:
                b1 = buckets.iloc[0]
                bN = buckets.iloc[-1]
                row = {
                    "window": f"[{s}, {e}]",
                    "kind": kind,
                    "n": ic["n"],
                    "ic_spearman": ic.get("ic_spearman"),
                    "ic_pearson": ic.get("ic_pearson"),
                    "bucket1_mean_pct": float(b1["mean_pct"]),
                    "bucketN_mean_pct": float(bN["mean_pct"]),
                    "spread_pct": float(b1["mean_pct"] - bN["mean_pct"]),
                }
            rows.append(row)
    return pd.DataFrame(rows)
