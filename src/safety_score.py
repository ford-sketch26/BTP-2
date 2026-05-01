"""Safety Score — Phase 3, Idea 1.

One number per trial summarising "how dangerous does the drug look vs placebo?"

v1 design (deliberately simple, easy to explain and audit):

    score = drug_serious_rate - placebo_serious_rate

  where
    drug_serious_rate    = sum(serious_num_affected over drug arms)    / sum(serious_num_at_risk over drug arms)
    placebo_serious_rate = sum(serious_num_affected over placebo arms) / sum(serious_num_at_risk over placebo arms)

Higher score = drug arm has a higher proportion of patients with serious
adverse events than placebo. Negative score = drug arm looks *safer* than
placebo (rare, but does happen — placebos can carry trial-protocol-related risks).

Trials without an identifiable placebo arm get NaN and are excluded from
backtests. v2 ideas to revisit later:
  * Weight by event severity (deaths >> serious >> other)
  * Compute log risk ratio with Laplace smoothing (handles sparse cells better)
  * Per-organ-system decomposition (some drugs hit one organ very hard)
  * Compare against a *historical baseline* for the drug class instead of just
    the within-trial placebo arm (cf. Idea 2: anomaly detector)
"""

from __future__ import annotations

import pandas as pd


def compute_safety_score(arms_df: pd.DataFrame) -> pd.DataFrame:
    """For each NCT, return one row with `safety_score` plus diagnostics.

    Output columns:
        nct_id, drug_arms_n, placebo_arms_n,
        drug_at_risk, drug_affected, drug_rate,
        placebo_at_risk, placebo_affected, placebo_rate,
        safety_score, score_basis ('drug-minus-placebo' or 'no-placebo')
    """
    if arms_df.empty:
        return pd.DataFrame()

    out_rows: list[dict] = []
    for nct, sub in arms_df.groupby("nct_id"):
        drug = sub[sub["arm_role"] == "drug"]
        placebo = sub[sub["arm_role"] == "placebo"]

        d_at_risk = drug["serious_num_at_risk"].fillna(0).sum()
        d_affected = drug["serious_num_affected"].fillna(0).sum()
        p_at_risk = placebo["serious_num_at_risk"].fillna(0).sum()
        p_affected = placebo["serious_num_affected"].fillna(0).sum()

        d_rate = (d_affected / d_at_risk) if d_at_risk > 0 else None
        p_rate = (p_affected / p_at_risk) if p_at_risk > 0 else None

        if d_rate is not None and p_rate is not None:
            score = d_rate - p_rate
            basis = "drug-minus-placebo"
        else:
            score = None
            basis = "no-placebo" if p_at_risk == 0 else "no-drug"

        out_rows.append({
            "nct_id": nct,
            "drug_arms_n": len(drug),
            "placebo_arms_n": len(placebo),
            "drug_at_risk": int(d_at_risk),
            "drug_affected": int(d_affected),
            "drug_rate": d_rate,
            "placebo_at_risk": int(p_at_risk),
            "placebo_affected": int(p_affected),
            "placebo_rate": p_rate,
            "safety_score": score,
            "score_basis": basis,
        })

    return pd.DataFrame(out_rows)
