"""Local web UI for the BTP biopharma quant pipeline.

Run with:
    python app.py

Browser tab opens at http://localhost:8000. Hit Ctrl+C to stop.
"""

from __future__ import annotations

import html as html_lib
import json
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

from src.data_cleaner import flatten, flatten_arms
from src.data_fetcher import (
    fetch_completed_trials_for_sponsors,
    resolve_sponsor_names,
)
from src.safety_score import compute_safety_score


import os
PORT = int(os.environ.get("PORT", "8000"))
BIND_HOST = os.environ.get("BIND_HOST", "127.0.0.1")  # set to "0.0.0.0" to allow external connections (ngrok / cloud deploy)
# When set, the app serves from the pre-built safety_panel.csv instead of fetching live.
# Trades freshness for instant response — perfect for a public demo where strangers
# would otherwise hit slow yfinance / CT.gov calls.
DEMO_MODE = os.environ.get("DEMO_MODE", "0") == "1"
DEMO_PANEL_PATH = "data/processed/safety_panel.csv"

# (ticker, max_pages) -> {names, trials, events, arms, safety, panel}
_CACHE: dict[tuple[str, int], dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()


# ============================== HTML / CSS ===================================

PAGE_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, system-ui, Segoe UI, sans-serif;
       max-width: 1400px; margin: 1.5em auto; padding: 0 1.5em; color: #1a202c;
       line-height: 1.5; }
h1 { margin-bottom: 0.2em; }
h2 { margin-top: 1.8em; padding-bottom: 0.3em; border-bottom: 2px solid #e2e8f0; }
h3 { color: #2d3748; margin-top: 1.5em; }
.subtitle { color: #718096; margin-bottom: 2em; }
.help-box { background: #ebf4ff; border-left: 4px solid #3182ce;
            padding: 0.8em 1em; margin: 1em 0; border-radius: 4px;
            font-size: 0.95em; }
.help-box strong { color: #2c5282; }
.warn-box { background: #fffaf0; border-left: 4px solid #dd6b20;
            padding: 0.8em 1em; margin: 1em 0; border-radius: 4px; }
form { display: flex; gap: 1em; align-items: end; flex-wrap: wrap;
       background: #f7fafc; padding: 1.5em; border-radius: 8px; }
form label { display: block; font-size: 0.9em; color: #4a5568; margin-bottom: 0.3em; font-weight: 600; }
form input, form select { padding: 0.6em; border: 1px solid #cbd5e0; border-radius: 4px;
                          font-size: 1em; font-family: inherit; }
form input[type=text] { width: 200px; }
form input[type=number] { width: 100px; }
form button { padding: 0.65em 1.5em; background: #2b6cb0; color: white;
              border: none; border-radius: 4px; font-size: 1em; cursor: pointer; }
form button:hover { background: #2c5282; }
.help { font-size: 0.85em; color: #718096; }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
           gap: 1em; margin: 1.5em 0; }
.metric { background: #edf2f7; padding: 1.2em; border-radius: 8px; }
.metric .num { font-size: 2em; font-weight: 700; color: #2b6cb0; }
.metric .label { font-size: 0.85em; color: #4a5568; text-transform: uppercase;
                 letter-spacing: 0.05em; }
.metric .sub { font-size: 0.8em; color: #718096; margin-top: 0.3em; }
table { width: 100%; border-collapse: collapse; font-size: 0.9em; margin: 1em 0; }
th, td { padding: 0.5em 0.7em; text-align: left; border-bottom: 1px solid #e2e8f0; vertical-align: top; }
th { background: #f7fafc; font-weight: 600; position: sticky; top: 0; z-index: 1; }
tr:hover { background: #fefcbf30; }
.scroll { max-height: 600px; overflow: auto; border: 1px solid #e2e8f0; border-radius: 4px; }
.bar { display: inline-block; height: 14px; background: #4299e1;
       vertical-align: middle; margin-right: 0.5em; }
.error { background: #fed7d7; border: 1px solid #fc8181; padding: 1em;
         border-radius: 6px; color: #742a2a; }
.alert-row { background: #fff5f5 !important; }
a { color: #2b6cb0; }
.tag { display: inline-block; padding: 1px 8px; border-radius: 10px;
       font-size: 0.78em; font-weight: 600; }
.tag.scary { background: #fed7d7; color: #742a2a; }
.tag.warn { background: #feebc8; color: #7b341e; }
.tag.clean { background: #c6f6d5; color: #22543d; }
.tag.unknown { background: #e2e8f0; color: #4a5568; }
.score-pill { display: inline-block; padding: 2px 10px; border-radius: 12px;
              font-weight: 600; font-family: monospace; }
.score-pill.scary { background: #e53e3e; color: white; }
.score-pill.warn { background: #dd6b20; color: white; }
.score-pill.clean { background: #38a169; color: white; }
.score-pill.unknown { background: #a0aec0; color: white; }
.pct-good { color: #38a169; }
.pct-warn { color: #dd6b20; }
.pct-bad  { color: #e53e3e; font-weight: 600; }
details > summary { cursor: pointer; padding: 0.4em; }
.kpi { font-family: monospace; font-size: 1.05em; }
"""


def _esc(s: Any) -> str:
    return html_lib.escape("" if s is None else str(s))


def _page(title: str, body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{_esc(title)}</title>
<style>{PAGE_CSS}</style></head><body>{body_html}</body></html>"""


def _form_html(default_ticker: str = "GILD", default_pages: int = 1) -> str:
    return f"""
<form method="get" action="/run">
  <div>
    <label>Ticker</label>
    <input type="text" name="ticker" value="{_esc(default_ticker)}" required>
    <div class="help">Try GILD, MRNA, PFE, JNJ, BIIB, LLY, ABBV...</div>
  </div>
  <div>
    <label>How much data to pull</label>
    <input type="number" name="max_pages" value="{default_pages}" min="1" max="20">
    <div class="help">Each page = ~100 trials per sponsor. 1 is fast (demo). Higher = more complete (slower).</div>
  </div>
  <button type="submit">Fetch and analyse</button>
</form>
"""


def _score_pill(score: float | None) -> str:
    """Render a safety score with colour coding."""
    if score is None or pd.isna(score):
        return '<span class="score-pill unknown">N/A</span>'
    pct = score * 100
    cls = "scary" if score >= 0.10 else ("warn" if score >= 0.03 else "clean")
    sign = "+" if score >= 0 else ""
    return f'<span class="score-pill {cls}">{sign}{pct:.1f} pp</span>'


def _score_label(score: float | None) -> str:
    """Plain-English label for a score."""
    if score is None or pd.isna(score):
        return "Can't score (no placebo arm)"
    if score >= 0.10:
        return "SCARY — drug looks much worse than placebo"
    if score >= 0.03:
        return "Warning — drug somewhat worse than placebo"
    if score >= -0.03:
        return "Clean — drug roughly similar to placebo"
    return "Drug looks safer than placebo (rare!)"


# ============================== pipeline =====================================

# Cached panel (loaded once in DEMO_MODE)
_DEMO_PANEL: pd.DataFrame | None = None


def _load_demo_panel() -> pd.DataFrame:
    global _DEMO_PANEL
    if _DEMO_PANEL is None:
        _DEMO_PANEL = pd.read_csv(DEMO_PANEL_PATH)
    return _DEMO_PANEL


def _run_pipeline_demo(ticker: str) -> dict[str, Any]:
    """Demo-mode pipeline: filter the pre-built panel CSV. Instant response.

    Trade-offs vs live mode:
      * No live freshness — uses the snapshot from the last `python -m src.build_panel` run.
      * No drill-in side-effect tables (events_df is empty) — drill-in shows
        metadata + score + a link to clinicaltrials.gov for full per-arm detail.
      * Only the 10 watchlist tickers will return data.
    """
    panel_full = _load_demo_panel()
    sub = panel_full[panel_full["ticker"] == ticker.upper()].copy()
    if sub.empty:
        available = sorted(panel_full["ticker"].dropna().unique().tolist())
        raise ValueError(
            f"Ticker {ticker!r} is not in the demo dataset. "
            f"Available: {', '.join(available)}"
        )

    trial_cols = [
        "nct_id", "brief_title", "lead_sponsor", "phase", "study_type",
        "enrollment_count", "conditions", "intervention_names",
        "completion_date", "results_first_posted", "has_adverse_events", "n_arms",
    ]
    trials_df = sub[[c for c in trial_cols if c in sub.columns]].drop_duplicates("nct_id").reset_index(drop=True)

    score_cols = [
        "nct_id", "drug_arms_n", "placebo_arms_n",
        "drug_at_risk", "drug_affected", "drug_rate",
        "placebo_at_risk", "placebo_affected", "placebo_rate",
        "safety_score", "score_basis",
    ]
    safety_df = sub[[c for c in score_cols if c in sub.columns]].drop_duplicates("nct_id").reset_index(drop=True)

    panel_local = trials_df.merge(safety_df, on="nct_id", how="left")

    return {
        "names": [f"(demo data — {len(trials_df)} pre-loaded trials for {ticker.upper()})"],
        "trials_df": trials_df,
        "events_df": pd.DataFrame(),  # empty -> drill-in will show only metadata + score
        "arms_df": pd.DataFrame(),
        "safety_df": safety_df,
        "panel": panel_local,
    }


def _run_pipeline(ticker: str, max_pages: int) -> dict[str, Any]:
    """Resolve, fetch, clean, score, cache. Returns everything the UI needs."""
    if DEMO_MODE:
        # In demo mode we ignore max_pages — there's only one snapshot per ticker.
        key = (ticker, 0)
        with _CACHE_LOCK:
            if key in _CACHE:
                return _CACHE[key]
        bundle = _run_pipeline_demo(ticker)
        with _CACHE_LOCK:
            _CACHE[key] = bundle
        return bundle

    key = (ticker, max_pages)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]

    names = resolve_sponsor_names(ticker)
    studies = fetch_completed_trials_for_sponsors(names, max_pages=max_pages)
    trials_df, events_df = flatten(studies)
    arms_df = flatten_arms(studies)
    safety_df = compute_safety_score(arms_df)

    # Convenience: a per-NCT panel that joins trials metadata + safety score.
    if not trials_df.empty:
        panel = trials_df.merge(safety_df, on="nct_id", how="left")
    else:
        panel = pd.DataFrame()

    bundle = {
        "names": names,
        "trials_df": trials_df,
        "events_df": events_df,
        "arms_df": arms_df,
        "safety_df": safety_df,
        "panel": panel,
    }
    with _CACHE_LOCK:
        _CACHE[key] = bundle
    return bundle


# ============================== views ========================================

def _home_view() -> str:
    quick_links_html = ""
    demo_note = ""
    if DEMO_MODE:
        try:
            tickers = sorted(_load_demo_panel()["ticker"].dropna().unique().tolist())
        except Exception:
            tickers = []
        if tickers:
            buttons = "".join(
                f'<a href="/run?ticker={t}" '
                f'style="display:inline-block;padding:0.5em 1em;margin:0.25em;'
                f'background:#2b6cb0;color:white;border-radius:6px;'
                f'text-decoration:none;font-weight:600">{t}</a>'
                for t in tickers
            )
            quick_links_html = f"""
<h3 style="margin-top: 1em">Or click any pre-loaded ticker for an instant view:</h3>
<div>{buttons}</div>
"""
        demo_note = """
<div class="warn-box">
  <strong>Demo mode:</strong> serving from a snapshot of clinicaltrials.gov data
  (last updated 2026-04-30). Only the watchlist tickers below have data. For a
  fully-live version (any ticker, freshly fetched), run the project locally —
  see the README on the GitHub repo.
</div>
"""

    body = f"""
<h1>Biopharma Trial Safety Explorer</h1>
<p class="subtitle">Type a US biopharma ticker. We'll show every completed clinical trial,
score how scary each drug looks compared to placebo, and surface the alarming ones.</p>

{demo_note}

{_form_html()}

{quick_links_html}

<div class="help-box" style="margin-top: 1.5em">
  <strong>What you'll see after fetching:</strong>
  <ol style="margin: 0.5em 0 0 1.5em">
    <li><strong>Headline metrics</strong> — how many trials we found, how many had side-effect data, how many had a placebo arm we could compare against</li>
    <li><strong>Alerts</strong> — the trials whose drug arm looks much worse than placebo (the "you should look at this" list)</li>
    <li><strong>Safety-score table</strong> — every trial we could score, sortable</li>
    <li><strong>Drill-in</strong> — pick any trial, see metadata + score, with a verification link to clinicaltrials.gov</li>
  </ol>
</div>

<h2>What this is doing under the hood</h2>
<ol>
  <li>Resolve the ticker to all CT.gov sponsor names (yfinance + manual overrides + auto-discovery — handles renames like Moderna→ModernaTX, acquisitions like Wyeth→Pfizer).</li>
  <li>Hit clinicaltrials.gov v2 once per resolved name; dedupe by trial ID.</li>
  <li>Flatten the deeply-nested JSON into tidy tables (trials + events + arm rollups).</li>
  <li>For each trial with both a drug arm and a placebo arm, compute:
      <span class="kpi">safety_score = (% of drug-arm patients with serious side effects) − (% of placebo-arm patients with serious side effects)</span></li>
  <li>Higher score = drug looks scarier than placebo. Trials without a placebo arm get N/A.</li>
</ol>
"""
    return _page("Biopharma Trial Explorer", body)


def _alerts_section_html(panel: pd.DataFrame) -> str:
    """Top-10 highest safety-score trials — the 'these stand out' section."""
    scored = panel.dropna(subset=["safety_score"]).sort_values("safety_score", ascending=False)
    if scored.empty:
        return ""

    top = scored.head(10)
    rows_html = []
    for _, r in top.iterrows():
        cls = "alert-row" if r["safety_score"] >= 0.05 else ""
        rows_html.append(f"""
<tr class="{cls}">
  <td>{_score_pill(r['safety_score'])}</td>
  <td><a href="https://clinicaltrials.gov/study/{_esc(r['nct_id'])}" target="_blank">{_esc(r['nct_id'])}</a></td>
  <td>{_esc(r.get('phase'))}</td>
  <td>{_esc(r.get('enrollment_count'))}</td>
  <td><span class="pct-bad">{r['drug_rate']*100:.1f}%</span></td>
  <td><span class="pct-good">{r['placebo_rate']*100:.1f}%</span></td>
  <td>{_esc(str(r.get('brief_title') or '')[:80])}</td>
</tr>""")

    n_alerts = (scored["safety_score"] >= 0.05).sum()
    return f"""
<h2>Alerts — trials with the biggest drug-vs-placebo safety gap</h2>
<div class="help-box">
  <strong>How to read this:</strong> Each row is one trial. The <em>Safety Score</em>
  is the percentage-point difference between the drug arm's serious-side-effect rate
  and the placebo arm's rate. A score of <span class="score-pill scary">+10.0 pp</span>
  means 10% MORE drug-arm patients had a serious event than placebo-arm patients.
  Rows shaded pink have crossed our 5pp alert threshold ({n_alerts} of {len(scored)} scoreable trials).
  Click an NCT to verify on clinicaltrials.gov.
</div>
<div class="scroll">
<table>
<tr><th>Score</th><th>Trial</th><th>Phase</th><th>Enrolled</th>
    <th>Drug-arm serious rate</th><th>Placebo-arm serious rate</th><th>Title</th></tr>
{''.join(rows_html)}
</table>
</div>
"""


def _safety_table_html(panel: pd.DataFrame) -> str:
    """Full sortable table of all trials with safety scores."""
    scored = panel.dropna(subset=["safety_score"]).sort_values("safety_score", ascending=False)
    if scored.empty:
        return '<p><em>No trials in this dataset have both a drug arm and a placebo arm — nothing to score.</em></p>'

    rows_html = []
    for _, r in scored.iterrows():
        rows_html.append(f"""
<tr>
  <td>{_score_pill(r['safety_score'])}</td>
  <td><a href="https://clinicaltrials.gov/study/{_esc(r['nct_id'])}" target="_blank">{_esc(r['nct_id'])}</a></td>
  <td>{_esc(r.get('phase'))}</td>
  <td>{_esc(r.get('enrollment_count'))}</td>
  <td>{r['drug_rate']*100:.1f}%</td>
  <td>{r['placebo_rate']*100:.1f}%</td>
  <td>{_esc(r.get('completion_date'))}</td>
  <td>{_esc(str(r.get('brief_title') or '')[:80])}</td>
</tr>""")

    return f"""
<h2>All scoreable trials, ranked</h2>
<div class="help-box">
  Every trial below has both a drug arm AND a placebo arm, so we could compute the
  comparison. Sorted scariest first. Higher score = drug arm caused noticeably more
  serious side effects than placebo did.
</div>
<div class="scroll">
<table>
<tr><th>Score</th><th>Trial</th><th>Phase</th><th>Enrolled</th>
    <th>Drug serious rate</th><th>Placebo serious rate</th>
    <th>Completion</th><th>Title</th></tr>
{''.join(rows_html)}
</table>
</div>
"""


def _bar_chart(series: pd.Series, max_bars: int = 15) -> str:
    if series.empty:
        return ""
    top = series.head(max_bars)
    max_val = top.max()
    rows = []
    for label, val in top.items():
        width = int(400 * val / max_val) if max_val else 0
        rows.append(
            f"<tr><td style='white-space:nowrap'>{_esc(label)}</td>"
            f"<td><span class='bar' style='width:{width}px'></span> {int(val):,}</td></tr>"
        )
    return f"<table>{''.join(rows)}</table>"


def _drill_in_html(
    nct: str, trials_df: pd.DataFrame, events_df: pd.DataFrame,
    arms_df: pd.DataFrame, safety_df: pd.DataFrame,
) -> str:
    trow = trials_df[trials_df["nct_id"] == nct]
    if trow.empty:
        return f'<div class="error">Trial {_esc(nct)} not found in this dataset.</div>'
    t = trow.iloc[0]

    score_row = safety_df[safety_df["nct_id"] == nct]
    if not score_row.empty and pd.notna(score_row.iloc[0]["safety_score"]):
        s = score_row.iloc[0]
        score = float(s["safety_score"])
        score_html = f"""
<div class="help-box" style="background: #f0fff4; border-left-color: #38a169">
  <h3 style="margin-top: 0">Safety Score: {_score_pill(score)}
    <span style="font-weight: normal; color: #4a5568">— {_score_label(score)}</span>
  </h3>
  <p style="margin: 0.5em 0">
    In this trial's drug arm(s), <strong>{s['drug_rate']*100:.1f}%</strong> of patients
    ({int(s['drug_affected'])}/{int(s['drug_at_risk'])}) had at least one serious side effect.<br>
    In the placebo arm(s), <strong>{s['placebo_rate']*100:.1f}%</strong>
    ({int(s['placebo_affected'])}/{int(s['placebo_at_risk'])}) did.<br>
    <em>Difference: {(s['drug_rate']-s['placebo_rate'])*100:+.1f} percentage points.</em>
  </p>
</div>
"""
    else:
        basis = score_row.iloc[0]["score_basis"] if not score_row.empty else "no-data"
        score_html = f"""
<div class="warn-box">
  <strong>No safety score for this trial.</strong> Reason: <code>{_esc(basis)}</code>.
  Most likely a single-arm trial (everyone got the drug, no placebo to compare to —
  common in Phase 1 studies).
</div>
"""

    sub = events_df[events_df["nct_id"] == nct]

    # Per-arm pivot tables
    pivots_html = ""
    for severity, label in [("serious", "Serious side effects"), ("other", "Mild side effects")]:
        sev = sub[sub["severity_class"] == severity]
        if sev.empty:
            continue
        pivot = sev.pivot_table(
            index=["organ_system", "event_term"],
            columns="group_title",
            values="incidence_rate",
            aggfunc="first",
        )
        rows = []
        rows.append("<tr><th>Body system</th><th>Side effect</th>" +
                    "".join(f"<th>{_esc(c)}</th>" for c in pivot.columns) + "</tr>")
        for (organ, term), row in pivot.iterrows():
            cells = [f"<td>{_esc(organ)}</td>", f"<td>{_esc(term)}</td>"]
            for c in pivot.columns:
                v = row[c]
                if pd.isna(v):
                    cells.append("<td></td>")
                else:
                    pct = v * 100
                    cls = "pct-bad" if pct >= 10 else ("pct-warn" if pct >= 5 else "pct-good")
                    cells.append(f'<td class="{cls}">{pct:.1f}%</td>')
            rows.append("<tr>" + "".join(cells) + "</tr>")
        intro = (
            "<strong>How to read:</strong> each row is one specific side effect. "
            "Each column is one arm of the trial. The cell is the percentage of patients "
            "in that arm who had that side effect. Compare columns to see if the drug arm "
            "is worse than placebo." if severity == "serious" else
            "Same idea as the serious-events table above, but for milder side effects."
        )
        pivots_html += f"""
<h3>{label} — % of patients in each arm</h3>
<div class="help-box">{intro}</div>
<div class="scroll"><table>{''.join(rows)}</table></div>
"""

    if not pivots_html:
        pivots_html = "<p><em>No adverse-event rows for this trial.</em></p>"

    return f"""
<h2>{_esc(nct)} — {_esc(t['brief_title'])}</h2>
<p>
  <strong>Sponsor:</strong> {_esc(t['lead_sponsor'])} &nbsp;|&nbsp;
  <strong>Phase:</strong> {_esc(t['phase'])} &nbsp;|&nbsp;
  <strong>Patients:</strong> {_esc(t['enrollment_count'])} &nbsp;|&nbsp;
  <strong>Completed:</strong> {_esc(t['completion_date'])}
</p>
<p><strong>Conditions:</strong> {_esc(t['conditions'])}</p>
<p><strong>Drugs being tested:</strong> {_esc(t['intervention_names'])}</p>
<p><a href="https://clinicaltrials.gov/study/{_esc(nct)}" target="_blank">
   → Verify on clinicaltrials.gov (open the official "Adverse Events" section
   and check that our percentages match)</a></p>

{score_html}
{pivots_html}
"""


def _run_view(ticker: str, max_pages: int, drill_nct: str | None) -> str:
    try:
        bundle = _run_pipeline(ticker, max_pages)
    except Exception as exc:  # noqa: BLE001
        return _page("Error", f"""
<a href="/">&larr; back</a>
<h1>Failed to fetch {_esc(ticker)}</h1>
<div class="error">
  <strong>{_esc(type(exc).__name__)}:</strong> {_esc(exc)}
  <pre>{_esc(traceback.format_exc())}</pre>
</div>
""")

    names = bundle["names"]
    trials_df: pd.DataFrame = bundle["trials_df"]
    events_df: pd.DataFrame = bundle["events_df"]
    arms_df: pd.DataFrame = bundle["arms_df"]
    safety_df: pd.DataFrame = bundle["safety_df"]
    panel: pd.DataFrame = bundle["panel"]

    n_with_ae = int(trials_df["has_adverse_events"].sum()) if not trials_df.empty else 0
    n_scored = int(safety_df["safety_score"].notna().sum()) if not safety_df.empty else 0
    n_alerts = int((safety_df["safety_score"] >= 0.05).sum()) if not safety_df.empty else 0

    intro = f"""
<a href="/">&larr; New search</a>
<h1>{_esc(ticker)} — Trial Safety Dashboard</h1>

<div class="help-box">
  <strong>How to read this page:</strong>
  <ol style="margin: 0.5em 0 0 1.5em">
    <li>The <strong>Alerts section</strong> shows trials where the drug looks
        meaningfully scarier than the placebo. Start there.</li>
    <li>The <strong>Safety Score</strong> for each trial is one number: how many
        more percentage points of drug-arm patients had a serious side effect
        compared to placebo-arm patients. <span class="score-pill scary">+10 pp</span>
        means 10% more drug patients than placebo patients got hospitalized
        (or worse).</li>
    <li>Click any <strong>NCT id</strong> to drill in — see the trial's arms
        side-by-side, and follow the verification link to clinicaltrials.gov to
        confirm our numbers match the official ones.</li>
    <li>Trials without a placebo arm (typical of Phase 1) can't be scored —
        they're not in the alerts list.</li>
  </ol>
</div>

<details><summary>{len(names)} sponsor name(s) used to query CT.gov for {_esc(ticker)}</summary>
<div class="help" style="background: #f7fafc; padding: 0.8em; border-radius: 6px; margin: 0.5em 0">
  {"<br>".join(_esc(n) for n in names)}
</div></details>

<div class="metrics">
  <div class="metric">
    <div class="num">{len(trials_df)}</div>
    <div class="label">Total trials</div>
    <div class="sub">Completed studies pulled from CT.gov</div>
  </div>
  <div class="metric">
    <div class="num">{n_with_ae}</div>
    <div class="label">With side-effect data</div>
    <div class="sub">Trials that posted adverse events</div>
  </div>
  <div class="metric">
    <div class="num">{n_scored}</div>
    <div class="label">Scoreable</div>
    <div class="sub">Had both drug and placebo arms</div>
  </div>
  <div class="metric" style="background: {'#fed7d7' if n_alerts else '#edf2f7'}">
    <div class="num" style="color: {'#742a2a' if n_alerts else '#2b6cb0'}">{n_alerts}</div>
    <div class="label">Alerts</div>
    <div class="sub">Score ≥ +5 pp (drug noticeably worse)</div>
  </div>
</div>
"""

    alerts = _alerts_section_html(panel) if not panel.empty else ""
    safety_table = _safety_table_html(panel) if not panel.empty else ""

    # Drill-in selector
    drill_form = ""
    drill_html = ""
    if not events_df.empty:
        ncts = sorted(events_df["nct_id"].dropna().unique())
        # Default selection: pick the highest-score one if no nct chosen
        default_nct = drill_nct
        if not default_nct and not safety_df.empty:
            scored = safety_df.dropna(subset=["safety_score"]).sort_values("safety_score", ascending=False)
            if not scored.empty:
                default_nct = scored.iloc[0]["nct_id"]
        options = "".join(
            f'<option value="{_esc(n)}"{ " selected" if n == default_nct else ""}>{_esc(n)}</option>'
            for n in ncts
        )
        drill_form = f"""
<h2>Drill into a single trial</h2>
<div class="help-box">
  Pick any NCT id below. You'll get the trial's metadata, its safety score
  with plain-English interpretation, and a side-by-side per-arm table of every
  side effect with the verification link to clinicaltrials.gov.
</div>
<form method="get" action="/run">
  <input type="hidden" name="ticker" value="{_esc(ticker)}">
  <input type="hidden" name="max_pages" value="{max_pages}">
  <div><label>Trial</label>
    <select name="nct">{options}</select>
  </div>
  <button type="submit">Show details</button>
</form>
"""
        if drill_nct:
            drill_html = _drill_in_html(drill_nct, trials_df, events_df, arms_df, safety_df)

    # Chart of organ systems
    organ_chart = ""
    if not events_df.empty:
        organ_counts = events_df["organ_system"].fillna("(none)").value_counts()
        organ_chart = f"""
<h2>Most-reported body systems across all trials</h2>
<div class="help-box">
  Each bar is the total count of side-effect rows reported for that body system,
  across all the trials in this dataset. Tells you what kinds of side effects
  this company's drugs tend to produce.
</div>
{_bar_chart(organ_counts)}
"""

    body = intro + alerts + safety_table + organ_chart + drill_form + drill_html
    return _page(f"{ticker} - Trial Safety Dashboard", body)


# ============================== HTTP =========================================

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"  [{self.command}] {self.path}\n")

    def _send(self, body: str, status: int = 200, content_type: str = "text/html; charset=utf-8") -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if url.path == "/":
            self._send(_home_view())
            return
        if url.path == "/run":
            params = parse_qs(url.query)
            ticker = (params.get("ticker", [""])[0] or "").strip().upper()
            try:
                max_pages = int(params.get("max_pages", ["1"])[0])
            except ValueError:
                max_pages = 1
            max_pages = max(1, min(20, max_pages))
            nct = (params.get("nct", [""])[0] or "").strip().upper() or None
            if not ticker:
                self._send(_page("Error", '<a href="/">back</a><h1>Missing ticker</h1>'))
                return
            self._send(_run_view(ticker, max_pages, nct))
            return
        if url.path == "/health":
            self._send(json.dumps({"status": "ok"}), content_type="application/json")
            return
        self._send(_page("Not Found", '<h1>404</h1><a href="/">home</a>'), status=404)


def main() -> None:
    server = ThreadingHTTPServer((BIND_HOST, PORT), Handler)
    visible_host = "localhost" if BIND_HOST in ("127.0.0.1", "0.0.0.0") else BIND_HOST
    url = f"http://{visible_host}:{PORT}"
    print(f"Biopharma Trial Safety Dashboard running at {url}")
    print(f"Bound to {BIND_HOST}:{PORT}")
    if BIND_HOST == "0.0.0.0":
        print("(External connections allowed — exposing via ngrok or cloud deploy.)")
    print("Press Ctrl+C to stop.")
    # Only auto-open the browser when running locally — skip when deployed/exposed.
    if os.environ.get("AUTO_OPEN_BROWSER", "1") == "1" and BIND_HOST != "0.0.0.0":
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.server_close()


if __name__ == "__main__":
    main()
