"""Drug comparator — Part 2.

The Part-1 unit of analysis is "one trial". The doctor's unit of analysis is
"one drug for one condition" — they want to know "for hypertension, which of
sitagliptin / empagliflozin / saxagliptin has the cleanest safety profile?"

This module pivots the panel:
  * group trials by (intervention_mesh_id, condition_mesh_id, phase_scope)
  * aggregate safety scores across trials in the group
  * provide search by condition or drug name (raw or MeSH term)

Two phase scopes are computed in parallel for every drug-condition pair:
  * "pivotal" — only Phase 3 and Phase 4 trials. The default view; matches
    the slice where Part-1's backtest showed IC = -0.09. Cleaner populations,
    larger samples, mostly placebo-controlled designs.
  * "all"     — every phase, including Phase 1 dose-finding. Useful for early
    pipeline drugs where pivotal data doesn't exist yet, but mixes very
    different patient populations and trial designs (Phase 1 oncology trials
    have inherently high raw event rates regardless of drug).

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
                    "is_pivotal": _is_pivotal_phase(t.get("phase")),
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


_PIVOTAL_PHASES = {"PHASE3", "PHASE4", "PHASE2|PHASE3", "PHASE3|PHASE4"}


def _is_pivotal_phase(phase: object) -> bool:
    """A trial counts as 'pivotal' if its phase string contains PHASE3 or PHASE4."""
    if phase is None or (isinstance(phase, float) and pd.isna(phase)):
        return False
    p = str(phase).strip().upper()
    return p in _PIVOTAL_PHASES or "PHASE3" in p or "PHASE4" in p


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

    def _phase_breakdown(g: pd.DataFrame) -> str:
        """Compact human-readable phase distribution: e.g. '5 P3, 2 P2, 1 P1'."""
        if g.empty:
            return ""
        counts: dict[str, int] = {}
        for p in g["phase"].fillna("(none)"):
            label = "P4" if "PHASE4" in p else (
                "P3" if "PHASE3" in p else (
                "P2" if "PHASE2" in p else (
                "P1" if "PHASE1" in p else "?")))
            counts[label] = counts.get(label, 0) + 1
        order = ["P4", "P3", "P2", "P1", "?"]
        return ", ".join(f"{counts[p]} {p}" for p in order if p in counts and counts[p] > 0)

    def _scope_aggregates(g: pd.DataFrame, prefix: str) -> dict:
        """Compute weighted_score, pooled_drug_rate, totals for one phase scope."""
        scored = g.dropna(subset=["safety_score"])
        with_drug_rate = g.dropna(subset=["drug_rate"])

        if not scored.empty:
            weights = scored["enrollment_count"].fillna(1).clip(lower=1)
            ws = float((scored["safety_score"] * weights).sum() / weights.sum())
            ms = float(scored["safety_score"].mean())
        else:
            ws, ms = None, None

        if not with_drug_rate.empty:
            d_at_risk = float(with_drug_rate["drug_at_risk"].fillna(0).sum())
            d_affected = float(with_drug_rate["drug_affected"].fillna(0).sum())
            pdr = d_affected / d_at_risk if d_at_risk > 0 else None
        else:
            pdr = None
            d_at_risk = d_affected = 0.0

        return {
            f"{prefix}_n_trials": len(g),
            f"{prefix}_n_scored": len(scored),
            f"{prefix}_weighted_score": ws,
            f"{prefix}_mean_score": ms,
            f"{prefix}_pooled_drug_rate": pdr,
            f"{prefix}_drug_at_risk_total": int(d_at_risk),
            f"{prefix}_drug_affected_total": int(d_affected),
        }

    def _agg(g: pd.DataFrame) -> pd.Series:
        drug_label = g["drug_label"].mode().iloc[0] if not g["drug_label"].empty else ""
        cond_label = g["condition_label"].mode().iloc[0] if not g["condition_label"].empty else ""

        # Two parallel aggregations: "all" (every phase) and "pivotal" (Phase 3/4 only).
        all_aggs = _scope_aggregates(g, "all")
        pivotal_aggs = _scope_aggregates(g[g["is_pivotal"]], "pivotal")

        sponsors = sorted({s for s in g["lead_sponsor"].dropna().unique() if s})
        tickers = sorted({t for t in g["ticker"].dropna().unique() if t})

        out = {
            "drug_label": drug_label,
            "condition_label": cond_label,
            "n_trials": len(g),
            "phase_breakdown": _phase_breakdown(g),
            "sponsors": ", ".join(sponsors),
            "tickers": ", ".join(tickers),
        }
        out.update(all_aggs)
        out.update(pivotal_aggs)
        return pd.Series(out)

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


def _scope_score_col(scope: str) -> str:
    return "pivotal_weighted_score" if scope == "pivotal" else "all_weighted_score"


def _scope_n_col(scope: str) -> str:
    return "pivotal_n_trials" if scope == "pivotal" else "all_n_trials"


def alternatives_for_drug(
    index: pd.DataFrame, drug_query: str, min_trials: int = 1,
    phase_scope: str = "pivotal",
) -> pd.DataFrame:
    """Given a drug name, find conditions it's tested for and *competing* drugs
    tested for those same conditions, ranked by safety score.

    ``phase_scope`` is "pivotal" (Phase 3/4 only — the default) or "all".
    Drugs without trials in the chosen scope are excluded.
    """
    matches = search(index, drug_query, by="drug")
    if matches.empty:
        return pd.DataFrame()
    n_col = _scope_n_col(phase_scope)
    score_col = _scope_score_col(phase_scope)
    relevant_conditions = matches["condition_key"].unique()
    competitors = index[
        (index["condition_key"].isin(relevant_conditions))
        & (index[n_col] >= min_trials)
    ].copy()
    return competitors.sort_values(["condition_label", score_col], na_position="last")


def alternatives_for_condition(
    index: pd.DataFrame,
    condition_query: str,
    min_trials: int = 1,
    exact_label: str | None = None,
    phase_scope: str = "pivotal",
) -> pd.DataFrame:
    """Given a condition, return drugs tested for it, ranked by safety score.

    ``phase_scope`` is "pivotal" (Phase 3/4 only) or "all". Drugs without
    trials in the chosen scope are excluded — this is the "fairness" guard:
    a drug that only has Phase 1 data won't appear in the pivotal view.
    """
    matches = search(index, condition_query, by="condition")
    if matches.empty:
        return pd.DataFrame()
    if exact_label:
        matches = matches[matches["condition_label"].fillna("").str.lower() == exact_label.lower()]
    n_col = _scope_n_col(phase_scope)
    score_col = _scope_score_col(phase_scope)
    matches = matches[matches[n_col] >= min_trials]
    return matches.sort_values([score_col, n_col], ascending=[True, False], na_position="last")


def list_distinct_conditions(matches: pd.DataFrame, phase_scope: str = "pivotal") -> pd.DataFrame:
    """For a result set, summarise the distinct conditions present (label + counts).

    Returned columns: ``condition_label``, ``n_drugs``, ``n_trials``.
    Sorted by ``n_trials`` descending so most-evidence-rich conditions surface first.
    ``phase_scope`` decides whether to count pivotal-only or all-phase trials.
    """
    if matches.empty:
        return pd.DataFrame(columns=["condition_label", "n_drugs", "n_trials"])
    n_col = _scope_n_col(phase_scope)
    g = (
        matches.groupby("condition_label", dropna=False)
        .agg(n_drugs=("drug_key", "nunique"), n_trials=(n_col, "sum"))
        .reset_index()
        .sort_values("n_trials", ascending=False)
    )
    return g
