"""Fetchers for clinicaltrials.gov v2 and yfinance.

Phase 1.1 — ticker -> sponsor name resolution -> completed trials -> raw JSON.

Sponsor name resolution combines three strategies so this works for any ticker:

1. **yfinance lookup** + corporate-suffix stripping (handles `Gilead Sciences, Inc.`)
2. **Auto-discovery** of sponsor variants from CT.gov (handles `Moderna -> ModernaTX, Inc.`
   and acquired subs whose CT.gov name still contains the parent token, e.g.
   `Kite, A Gilead Company`).
3. **Manual override map** in `config.SPONSOR_OVERRIDES` for acquisitions whose
   CT.gov sponsor name shares no token with the parent (e.g. `Janssen` for JNJ).
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yfinance as yf

from src.config import (
    CTGOV_API_BASE,
    DEFAULT_PAGE_SIZE,
    RAW_DIR,
    REQUEST_TIMEOUT_SECONDS,
    SPONSOR_OVERRIDES,
)

_CORPORATE_SUFFIX_RE = re.compile(
    r",?\s*\b(?:Inc|Incorporated|Corp|Corporation|Co|Company|Ltd|Limited|LLC|PLC|"
    r"S\.?A\.?|N\.?V\.?|AG|GmbH|Holdings?|Group)\.?\s*$",
    flags=re.IGNORECASE,
)
_TRAILING_CONJUNCTION_RE = re.compile(r"\s+(?:and|&|of|the)\s*$", flags=re.IGNORECASE)


def _clean_company_name(name: str) -> str:
    """Strip trailing corporate suffixes and stray conjunctions.

    Examples:
        "Gilead Sciences, Inc."   -> "Gilead Sciences"
        "Eli Lilly and Company"   -> "Eli Lilly"
        "Foo Holdings, Inc."      -> "Foo"
    """
    cleaned = name.strip()
    for _ in range(3):
        new = _CORPORATE_SUFFIX_RE.sub("", cleaned).strip().rstrip(",").strip()
        new = _TRAILING_CONJUNCTION_RE.sub("", new).strip()
        if new == cleaned:
            break
        cleaned = new
    return cleaned


def _discovery_anchor(name: str) -> str | None:
    """Pick a distinctive token from a company name for broad CT.gov search.

    Returns the first token of length >= 4 (skipping `Eli`, `the`, `&`, etc).
    """
    for token in name.split():
        token = token.strip(",.&-")
        if len(token) >= 4:
            return token
    return None


def discover_sponsor_variants(
    anchor: str,
    sample_size: int = 200,
    min_count: int = 2,
) -> list[str]:
    """Probe CT.gov broadly and return distinct lead-sponsor names containing ``anchor``.

    Uses ``query.term`` (matches the anchor anywhere in the study) to surface
    sponsor strings that ``query.lead`` alone would miss — e.g. a search for
    ``Moderna`` returns trials whose lead sponsor is literally ``ModernaTX, Inc.``.

    The ``min_count`` filter drops sponsors that appear only once in the sample,
    which is a cheap way to avoid one-off false positives (a trial title that
    happens to mention the anchor in passing).
    """
    if len(anchor) < 4:
        return []
    resp = requests.get(
        f"{CTGOV_API_BASE}/studies",
        params={
            "query.term": anchor,
            "filter.overallStatus": "COMPLETED",
            "pageSize": sample_size,
            "format": "json",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    counts: dict[str, int] = {}
    for study in resp.json().get("studies", []):
        ls = (
            study.get("protocolSection", {})
            .get("sponsorCollaboratorsModule", {})
            .get("leadSponsor", {})
            .get("name", "")
        )
        if ls and anchor.lower() in ls.lower():
            counts[ls] = counts.get(ls, 0) + 1
    return sorted(name for name, count in counts.items() if count >= min_count)


def resolve_sponsor_names(ticker: str, verbose: bool = False) -> list[str]:
    """Return all CT.gov lead-sponsor strings to query for ``ticker``.

    Combines:
      1. yfinance ``.info`` longName (suffix-stripped)
      2. Manual ``SPONSOR_OVERRIDES`` for known acquisitions
      3. Auto-discovered variants from a broad CT.gov term-search

    Raises ``ValueError`` if no name can be resolved at all.
    """
    seeds: list[str] = []

    # 1. yfinance primary
    yf_name: str | None = None
    try:
        info = yf.Ticker(ticker).info
        raw = info.get("longName") or info.get("shortName")
        if raw:
            yf_name = _clean_company_name(raw)
            if yf_name:
                seeds.append(yf_name)
    except Exception as exc:  # yfinance is flaky — log and continue
        if verbose:
            print(f"  [warn] yfinance lookup failed for {ticker!r}: {exc}", file=sys.stderr)

    # 2. Manual overrides
    for alias in SPONSOR_OVERRIDES.get(ticker.upper(), []):
        if alias not in seeds:
            seeds.append(alias)

    if not seeds:
        raise ValueError(
            f"Could not resolve any sponsor name for ticker {ticker!r}. "
            f"Add an entry to SPONSOR_OVERRIDES in src/config.py."
        )

    # 3. Auto-discovery from each seed
    discovered: list[str] = []
    anchors_tried: set[str] = set()
    for seed in list(seeds):
        anchor = _discovery_anchor(seed)
        if not anchor or anchor.lower() in anchors_tried:
            continue
        anchors_tried.add(anchor.lower())
        for variant in discover_sponsor_variants(anchor):
            if variant not in seeds and variant not in discovered:
                discovered.append(variant)

    resolved = seeds + discovered

    if verbose:
        print(f"  Resolved sponsor names for {ticker.upper()}:")
        for n in resolved:
            tag = "seed" if n in seeds else "discovered"
            print(f"    [{tag:10s}] {n}")

    return resolved


def fetch_completed_trials(
    sponsor_name: str,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_pages: int | None = None,
) -> list[dict[str, Any]]:
    """Pull all 'Completed' studies for a single sponsor from CT.gov v2."""
    url = f"{CTGOV_API_BASE}/studies"
    params: dict[str, Any] = {
        "query.lead": sponsor_name,
        "filter.overallStatus": "COMPLETED",
        "pageSize": page_size,
        "format": "json",
        "countTotal": "true",
    }

    studies: list[dict[str, Any]] = []
    pages_fetched = 0
    while True:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        payload = resp.json()
        studies.extend(payload.get("studies", []))
        pages_fetched += 1

        token = payload.get("nextPageToken")
        if not token:
            break
        if max_pages is not None and pages_fetched >= max_pages:
            break
        params["pageToken"] = token

    return studies


def fetch_completed_trials_for_sponsors(
    sponsor_names: list[str],
    page_size: int = DEFAULT_PAGE_SIZE,
    max_pages: int | None = None,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Fetch trials for every sponsor name; merge and dedupe by NCT ID."""
    seen: dict[str, dict[str, Any]] = {}
    for name in sponsor_names:
        studies = fetch_completed_trials(name, page_size=page_size, max_pages=max_pages)
        added = 0
        for study in studies:
            nct = (
                study.get("protocolSection", {})
                .get("identificationModule", {})
                .get("nctId")
            )
            if nct and nct not in seen:
                seen[nct] = study
                added += 1
        if verbose:
            print(f"    {name!r:50s} -> {len(studies):>5} trials ({added} new, {len(studies)-added} dupes)")
    return list(seen.values())


def save_raw_payload(ticker: str, studies: list[dict[str, Any]]) -> Path:
    """Persist the raw study list under ``data/raw/<ticker>_<timestamp>.json``."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RAW_DIR / f"{ticker.upper()}_{timestamp}.json"
    path.write_text(json.dumps(studies, indent=2), encoding="utf-8")
    return path


def fetch_trials_for_ticker(
    ticker: str,
    max_pages: int | None = None,
    verbose: bool = False,
    save: bool = True,
) -> tuple[list[dict[str, Any]], Path | None]:
    """End-to-end: ticker -> resolved sponsors -> completed trials -> (optionally) JSON.

    Returns ``(studies, path_or_none)``. ``save=False`` skips disk write — useful
    when validating or when the disk is constrained.
    """
    sponsor_names = resolve_sponsor_names(ticker, verbose=verbose)
    studies = fetch_completed_trials_for_sponsors(
        sponsor_names, max_pages=max_pages, verbose=verbose
    )
    path = save_raw_payload(ticker, studies) if save else None
    return studies, path
