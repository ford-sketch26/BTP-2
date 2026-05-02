"""Drug comparator — Part 2.

The Part-1 unit of analysis is "one trial". The doctor's unit of analysis is
"one drug for one condition" — they want to know "for hypertension, which of
sitagliptin / empagliflozin / saxagliptin has the cleanest safety profile?"

This module pivots the panel:
  * group trials by (intervention_mesh_id, condition_mesh_id)
  * aggregate safety scores across all trials in the group
  * provide search by condition or drug name (raw or MeSH term)

MeSH (Medical Subject Headings) is the National Library of Medicine's
standardised vocabulary. Pulling MeSH IDs from CT.gov dedupes drug names
across pre-market codes (MK-3475), INN (pembrolizumab), and brand (Keytruda),
which would otherwise be impossible to group cleanly.

Trials without MeSH classification (~20-30% of CT.gov) fall back to raw
intervention-name string matching.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd


# Tokens we never want to count as "the drug being tested" — `\b` lets us also
# catch things like "Placebo for PF-06946860" which CT.gov sometimes uses.
_NON_DRUG_TOKENS = re.compile(
    r"^(placebo|sham|standard of care|usual care|control|saline|"
    r"diet|exercise|no treatment|observation|best supportive care|"
    r"vehicle|matching placebo|sugar pill|comparator)\b",
    flags=re.IGNORECASE,
)


def _split_pipe_field(value: object) -> list[str]:
    """Pipe-joined string -> list of cleaned tokens (or empty list)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = str(value).strip()
    if not s:
        return []
    return [t.strip() for t in s.split("|") if t.strip()]


def _filter_drug_names(names: list[str]) -> list[str]:
    """Drop placebo/sham/sugar-pill/etc — keep only candidate drug names."""
    return [n for n in names if not _NON_DRUG_TOKENS.match(n.strip())]


def _explode_pairs(panel: pd.DataFrame) -> pd.DataFrame:
    """One row in the panel can carry many (drug × condition) pairs.

    Explode into one row per pair so we can group cleanly.

    Uses MeSH IDs when available; falls back to lowercased raw strings so
    unclassified trials still participate. The ``key_*`` columns are the
    grouping key; the ``label_*`` columns are the human-readable display value.
    """
    rows: list[dict] = []
    for _, t in panel.iterrows():
        # Drug side -- prefer MeSH terms (canonical) over raw intervention names
        drug_meshes = list(zip(
            _split_pipe_field(t.get("intervention_mesh_ids")),
            _split_pipe_field(t.get("intervention_mesh_terms")),
        ))
        if drug_meshes:
            drug_pairs = [(f"mesh:{mid}", mterm) for mid, mterm in drug_meshes]
        else:
            raw = _filter_drug_names(_split_pipe_field(t.get("intervention_names")))
            drug_pairs = [(f"raw:{n.lower()}", n) for n in raw]

        # Condition side -- same priority
        cond_meshes = list(zip(
            _split_pipe_field(t.get("condition_mesh_ids")),
            _split_pipe_field(t.get("condition_mesh_terms")),
        ))
        if cond_meshes:
            cond_pairs = [(f"mesh:{cid}", cterm) for cid, cterm in cond_meshes]
        else:
            raw = _split_pipe_field(t.get("conditions"))
            cond_pairs = [(f"raw:{n.lower()}", n) for n in raw]

        if not drug_pairs:
            drug_pairs = [(None, None)]
        if not cond_pairs:
            cond_pairs = [(None, None)]

        for dkey, dlabel in drug_pairs:
            for ckey, clabel in cond_pairs:
                rows.append({
                    "nct_id": t.get("nct_id"),
                    "ticker": t.get("ticker"),
                    "lead_sponsor": t.get("lead_sponsor"),
                    "phase": t.get("phase"),
                    "enrollment_count": t.get("enrollment_count"),
                    "completion_date": t.get("completion_date"),
                    "drug_key": dkey,
                    "drug_label": dlabel,
                    "condition_key": ckey,
                    "condition_label": clabel,
                    "safety_score": t.get("safety_score"),
                    # drug-arm-only metrics — computable even without a placebo arm.
                    # These let us include single-arm trials in a separate "limited evidence" view.
                    "drug_rate": t.get("drug_rate"),
                    "drug_at_risk": t.get("drug_at_risk"),
                    "drug_affected": t.get("drug_affected"),
                    "abret_0_20": t.get("abret_0_20"),
                })
    return pd.DataFrame(rows)


def build_drug_condition_index(panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the panel into one row per (drug, condition) pair.

    Output columns:
        drug_key, drug_label, condition_key, condition_label,
        n_trials, n_scored, mean_score, weighted_score, sponsors, tickers
    """
    exploded = _explode_pairs(panel)
    exploded = exploded[exploded["drug_key"].notna() & exploded["condition_key"].notna()]
    if exploded.empty:
        return pd.DataFrame()

    def _agg(g: pd.DataFrame) -> pd.Series:
        scored = g.dropna(subset=["safety_score"])
        with_drug_rate = g.dropna(subset=["drug_rate"])
        drug_label = g["drug_label"].mode().iloc[0] if not g["drug_label"].empty else ""
        cond_label = g["condition_label"].mode().iloc[0] if not g["condition_label"].empty else ""

        # Enrollment-weighted safety score (drug minus placebo) — only when placebo data present
        if not scored.empty:
            weights = scored["enrollment_count"].fillna(1).clip(lower=1)
            weighted_score = float((scored["safety_score"] * weights).sum() / weights.sum())
            mean_score = float(scored["safety_score"].mean())
        else:
            weighted_score = None
            mean_score = None

        # Pooled drug-arm serious-event rate — computable for ANY trial with arm data,
        # including single-arm trials that don't get a safety score. Pool affected+at_risk
        # across all NCTs in the (drug × condition) group so big trials weight more naturally.
        if not with_drug_rate.empty:
            d_at_risk = float(with_drug_rate["drug_at_risk"].fillna(0).sum())
            d_affected = float(with_drug_rate["drug_affected"].fillna(0).sum())
            pooled_drug_rate = d_affected / d_at_risk if d_at_risk > 0 else None
        else:
            pooled_drug_rate = None
            d_at_risk = 0.0
            d_affected = 0.0

        sponsors = sorted({s for s in g["lead_sponsor"].dropna().unique() if s})
        tickers = sorted({t for t in g["ticker"].dropna().unique() if t})

        return pd.Series({
            "drug_label": drug_label,
            "condition_label": cond_label,
            "n_trials": len(g),
            "n_scored": len(scored),
            "n_with_drug_rate": len(with_drug_rate),
            "mean_score": mean_score,
            "weighted_score": weighted_score,
            "pooled_drug_rate": pooled_drug_rate,
            "drug_at_risk_total": int(d_at_risk),
            "drug_affected_total": int(d_affected),
            "sponsors": ", ".join(sponsors),
            "tickers": ", ".join(tickers),
        })

    grouped = (
        exploded.groupby(["drug_key", "condition_key"], dropna=False)
        .apply(_agg)
        .reset_index()
    )
    return grouped


def search(
    index: pd.DataFrame,
    query: str,
    by: str = "auto",
) -> pd.DataFrame:
    """Filter the index by a free-text query.

    Parameters
    ----------
    index : the result of `build_drug_condition_index`.
    query : user-typed text.
    by    : "drug" -> match against drug_label; "condition" -> match against
            condition_label; "auto" -> match against either.
    """
    if index.empty or not query:
        return pd.DataFrame()
    q = query.lower().strip()

    masks = []
    if by in ("drug", "auto"):
        masks.append(index["drug_label"].fillna("").str.lower().str.contains(q, regex=False))
    if by in ("condition", "auto"):
        masks.append(index["condition_label"].fillna("").str.lower().str.contains(q, regex=False))
    combined = masks[0]
    for m in masks[1:]:
        combined = combined | m
    return index[combined].copy()


def alternatives_for_drug(
    index: pd.DataFrame, drug_query: str, min_trials: int = 1,
) -> pd.DataFrame:
    """Given a drug name, find conditions it's tested for and *competing* drugs
    tested for those same conditions, ranked by safety score.
    """
    matches = search(index, drug_query, by="drug")
    if matches.empty:
        return pd.DataFrame()
    relevant_conditions = matches["condition_key"].unique()
    competitors = index[
        (index["condition_key"].isin(relevant_conditions))
        & (index["n_trials"] >= min_trials)
    ].copy()
    return competitors.sort_values(["condition_label", "weighted_score"], na_position="last")


def alternatives_for_condition(
    index: pd.DataFrame,
    condition_query: str,
    min_trials: int = 1,
    exact_label: str | None = None,
) -> pd.DataFrame:
    """Given a condition, return all drugs tested for it, ranked by safety score (cleanest first).

    If ``exact_label`` is provided, restrict to rows whose ``condition_label`` matches
    it exactly (case-insensitive). Useful for the "narrow to specific condition" UI.
    """
    matches = search(index, condition_query, by="condition")
    if matches.empty:
        return pd.DataFrame()
    if exact_label:
        matches = matches[matches["condition_label"].fillna("").str.lower() == exact_label.lower()]
    matches = matches[matches["n_trials"] >= min_trials]
    return matches.sort_values(["weighted_score", "n_trials"], ascending=[True, False], na_position="last")


def list_distinct_conditions(matches: pd.DataFrame) -> pd.DataFrame:
    """For a result set, summarise the distinct conditions present (label + counts).

    Returned columns: ``condition_label``, ``n_drugs``, ``n_trials``.
    Sorted by ``n_trials`` descending so most-evidence-rich conditions surface first.
    """
    if matches.empty:
        return pd.DataFrame(columns=["condition_label", "n_drugs", "n_trials"])
    g = (
        matches.groupby("condition_label", dropna=False)
        .agg(n_drugs=("drug_key", "nunique"), n_trials=("n_trials", "sum"))
        .reset_index()
        .sort_values("n_trials", ascending=False)
    )
    return g
