"""Flatten clinicaltrials.gov v2 study JSON into tidy DataFrames.

Phase 1.2.

Two outputs per raw file:
  * ``trials``  — one row per NCT (metadata: title, sponsor, phase, dates, flags).
  * ``events``  — one row per (NCT, arm, severity_class, event_term).
                  This is the main analytical table.

We keep both because Phase-2 (market data) joins on the trials table by
``completion_date``, while Phase-3 (signal design) aggregates over the events
table by NCT or organ system.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from src.config import PROCESSED_DIR


# ----------------------------- loading ---------------------------------------

def load_raw_studies(raw_path: str | Path) -> list[dict[str, Any]]:
    """Read a raw CT.gov JSON file written by the fetcher."""
    path = Path(raw_path)
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------- per-trial extractors --------------------------

def _safe_get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Walk ``d`` through nested ``keys``; return ``default`` if any step is missing."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def extract_trial_metadata(study: dict[str, Any]) -> dict[str, Any]:
    """Return one flat dict of headline fields for a single study."""
    ps = study.get("protocolSection", {})
    rs = study.get("resultsSection", {})

    ident = ps.get("identificationModule", {})
    status = ps.get("statusModule", {})
    design = ps.get("designModule", {})
    sponsor_mod = ps.get("sponsorCollaboratorsModule", {})
    cond_mod = ps.get("conditionsModule", {})
    interv_mod = ps.get("armsInterventionsModule", {})

    # Phases come as a list like ["PHASE3"] or ["PHASE2", "PHASE3"]
    phases = design.get("phases") or []
    phase_str = "|".join(phases) if phases else None

    interventions = interv_mod.get("interventions") or []
    intervention_names = "|".join(
        i.get("name", "") for i in interventions if i.get("name")
    ) or None
    intervention_types = "|".join(
        sorted({i.get("type", "") for i in interventions if i.get("type")})
    ) or None

    conditions = cond_mod.get("conditions") or []
    condition_str = "|".join(conditions) if conditions else None

    ae_mod = rs.get("adverseEventsModule") or {}

    return {
        "nct_id": ident.get("nctId"),
        "brief_title": ident.get("briefTitle"),
        "official_title": ident.get("officialTitle"),
        "lead_sponsor": _safe_get(sponsor_mod, "leadSponsor", "name"),
        "lead_sponsor_class": _safe_get(sponsor_mod, "leadSponsor", "class"),
        "overall_status": status.get("overallStatus"),
        "phase": phase_str,
        "study_type": design.get("studyType"),
        "enrollment_count": _safe_get(design, "enrollmentInfo", "count"),
        "start_date": _safe_get(status, "startDateStruct", "date"),
        "completion_date": _safe_get(status, "completionDateStruct", "date"),
        "primary_completion_date": _safe_get(status, "primaryCompletionDateStruct", "date"),
        "results_first_posted": _safe_get(status, "resultsFirstPostDateStruct", "date"),
        "conditions": condition_str,
        "intervention_names": intervention_names,
        "intervention_types": intervention_types,
        "has_results": bool(rs),
        "has_adverse_events": bool(ae_mod),
        "n_arms": len(ae_mod.get("eventGroups") or []),
        "n_serious_events": len(ae_mod.get("seriousEvents") or []),
        "n_other_events": len(ae_mod.get("otherEvents") or []),
    }


_PLACEBO_RE = __import__("re").compile(r"\bplacebo\b|\bsham\b|\bsugar pill\b", flags=__import__("re").IGNORECASE)


def _classify_arm(title: str) -> str:
    """Heuristic: 'placebo' if title mentions placebo/sham/sugar-pill, else 'drug'.

    Active comparators (drug-vs-drug trials) get classified as 'drug' — they'll
    skew the safety score toward 0 (both arms have a real drug), which is the
    right behaviour: the score for those trials is uninformative and we'll
    exclude them at the score level via NaN rather than mislabeling here.
    """
    if not title:
        return "drug"
    if _PLACEBO_RE.search(title):
        return "placebo"
    return "drug"


def iter_arm_summary_rows(study: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield one row per (NCT × arm) — the per-arm rollup CT.gov stores in
    ``eventGroups[]``. This is distinct from per-event rows and gives us the
    *unique* count of patients with any serious / other / fatal event in each
    arm — the right number for a top-line safety score.
    """
    ps = study.get("protocolSection", {})
    nct = _safe_get(ps, "identificationModule", "nctId")
    rs = study.get("resultsSection", {})
    ae = rs.get("adverseEventsModule") or {}
    groups = ae.get("eventGroups") or []
    if not nct or not groups:
        return

    for g in groups:
        title = g.get("title") or ""
        yield {
            "nct_id": nct,
            "group_id": g.get("id"),
            "group_title": title,
            "arm_role": _classify_arm(title),
            "serious_num_at_risk": g.get("seriousNumAtRisk"),
            "serious_num_affected": g.get("seriousNumAffected"),
            "other_num_at_risk": g.get("otherNumAtRisk"),
            "other_num_affected": g.get("otherNumAffected"),
            "deaths_num_at_risk": g.get("deathsNumAtRisk"),
            "deaths_num_affected": g.get("deathsNumAffected"),
        }


def flatten_arms(studies: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the arms DataFrame: one row per (NCT, arm) with rollup counts and role."""
    rows: list[dict[str, Any]] = []
    for s in studies:
        rows.extend(iter_arm_summary_rows(s))
    df = pd.DataFrame(rows)
    if not df.empty:
        # Derived rates — guarded against zero/None denominators
        for sev in ("serious", "other", "deaths"):
            n = df[f"{sev}_num_at_risk"]
            k = df[f"{sev}_num_affected"]
            df[f"{sev}_rate"] = (k / n).where((n.notna()) & (n > 0))
    return df


def iter_adverse_event_rows(study: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield one row per (arm × event) for a single study.

    Rows are emitted both for ``seriousEvents`` and ``otherEvents``, tagged with
    a ``severity_class`` column. Trials without an adverse-events module yield
    nothing — they're tracked via the trials table's ``has_adverse_events`` flag.
    """
    ps = study.get("protocolSection", {})
    nct = _safe_get(ps, "identificationModule", "nctId")
    rs = study.get("resultsSection", {})
    ae = rs.get("adverseEventsModule")
    if not ae or not nct:
        return

    # Build a lookup from groupId -> arm title (so each event-row can name its arm)
    group_titles: dict[str, str] = {
        g.get("id", ""): g.get("title", "") for g in ae.get("eventGroups") or []
    }

    for severity in ("serious", "other"):
        events = ae.get(f"{severity}Events") or []
        for ev in events:
            term = ev.get("term")
            organ = ev.get("organSystem")
            vocab = ev.get("sourceVocabulary")
            assessment = ev.get("assessmentType")
            for stat in ev.get("stats") or []:
                gid = stat.get("groupId", "")
                yield {
                    "nct_id": nct,
                    "severity_class": severity,
                    "organ_system": organ,
                    "event_term": term,
                    "source_vocabulary": vocab,
                    "assessment_type": assessment,
                    "group_id": gid,
                    "group_title": group_titles.get(gid, ""),
                    "num_at_risk": stat.get("numAtRisk"),
                    "num_affected": stat.get("numAffected"),
                    "num_events": stat.get("numEvents"),
                }


# ----------------------------- top-level flattener ---------------------------

def flatten(studies: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn a list of raw studies into ``(trials_df, events_df)``."""
    trial_rows = [extract_trial_metadata(s) for s in studies]
    event_rows: list[dict[str, Any]] = []
    for s in studies:
        event_rows.extend(iter_adverse_event_rows(s))

    trials_df = pd.DataFrame(trial_rows)
    events_df = pd.DataFrame(event_rows)

    # Useful derived field: incidence rate per (NCT, arm, event)
    if not events_df.empty:
        events_df["incidence_rate"] = events_df.apply(
            lambda r: (r["num_affected"] / r["num_at_risk"])
            if (r["num_at_risk"] and r["num_affected"] is not None)
            else None,
            axis=1,
        )

    return trials_df, events_df


def save_processed(
    trials_df: pd.DataFrame,
    events_df: pd.DataFrame,
    base_name: str,
) -> tuple[Path, Path]:
    """Write the two DataFrames as CSVs to ``data/processed/<base>_{trials,events}.csv``."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    trials_path = PROCESSED_DIR / f"{base_name}_trials.csv"
    events_path = PROCESSED_DIR / f"{base_name}_events.csv"
    trials_df.to_csv(trials_path, index=False)
    events_df.to_csv(events_path, index=False)
    return trials_path, events_path


def clean_raw_file(
    raw_path: str | Path,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[Path, Path] | None]:
    """One-call helper: load a raw JSON, flatten, optionally save."""
    raw_path = Path(raw_path)
    studies = load_raw_studies(raw_path)
    trials_df, events_df = flatten(studies)
    paths = None
    if save:
        paths = save_processed(trials_df, events_df, raw_path.stem)
    return trials_df, events_df, paths


# ----------------------------- CLI -------------------------------------------

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Flatten a raw CT.gov JSON into tidy CSVs")
    parser.add_argument("raw_path", help="Path to a raw JSON file in data/raw/")
    parser.add_argument("--no-save", action="store_true", help="Skip writing CSVs")
    args = parser.parse_args()

    trials_df, events_df, paths = clean_raw_file(args.raw_path, save=not args.no_save)
    print(f"Trials:  {len(trials_df):>6} rows  ({trials_df['has_adverse_events'].sum()} with adverse events)")
    print(f"Events:  {len(events_df):>6} rows")
    if paths:
        t, e = paths
        print(f"Saved:   {t}")
        print(f"Saved:   {e}")


if __name__ == "__main__":
    _cli()
