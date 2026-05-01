"""Market-side data layer — yfinance OHLCV with on-disk cache.

Phase 2.0.

We fetch full historical daily OHLCV for each ticker once and store it as a
CSV in ``data/cache/market/<TICKER>.csv``. Subsequent calls read from the
cache. yfinance is unauthenticated and rate-limited, so caching is essential.

Phase-3 backtest code never calls yfinance directly — it goes through
``get_ohlcv()`` and gets a deterministic DataFrame.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from src.config import DATA_DIR


CACHE_DIR = DATA_DIR / "cache" / "market"
DEFAULT_BENCHMARK = "IBB"  # iShares Biotech ETF — the canonical biopharma benchmark


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}.csv"


def _is_stale(path: Path, max_age_days: int = 7) -> bool:
    """A cache is stale if it's older than `max_age_days` (so end-of-month re-runs pick up new prices)."""
    if not path.exists():
        return True
    age = datetime.now().timestamp() - path.stat().st_mtime
    return age > max_age_days * 86400


def get_ohlcv(
    ticker: str,
    force_refresh: bool = False,
    max_age_days: int = 7,
) -> pd.DataFrame:
    """Return full daily OHLCV for ``ticker``, indexed by date (tz-naive).

    Columns: Open, High, Low, Close, Volume (Adj Close becomes Close).
    Empty DataFrame if yfinance returns nothing (e.g., delisted, never listed).
    """
    ticker = ticker.upper()
    path = _cache_path(ticker)

    if not force_refresh and not _is_stale(path, max_age_days):
        df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
        return df

    # Fetch fresh from yfinance
    raw = yf.Ticker(ticker).history(period="max", auto_adjust=True)
    if raw.empty:
        # Cache the emptiness so we don't keep retrying — but write a tiny marker
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        empty.index.name = "Date"
        empty.to_csv(path)
        return empty

    # yfinance returns tz-aware datetimes — strip the tz so date comparisons are simple
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    raw.index.name = "Date"

    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    df = raw[keep].copy()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def get_ohlcv_many(
    tickers: list[str], force_refresh: bool = False
) -> dict[str, pd.DataFrame]:
    """Convenience: fetch many tickers, return dict keyed by uppercase ticker."""
    return {t.upper(): get_ohlcv(t, force_refresh=force_refresh) for t in tickers}


def parse_event_date(date_str: Optional[str]) -> Optional[pd.Timestamp]:
    """CT.gov dates can be 'YYYY', 'YYYY-MM', or 'YYYY-MM-DD'. Coerce to a Timestamp.

    For partial dates, anchor to the first day of the month/year — conservative
    (events fire as early as possible, biasing returns toward the unknown side).
    """
    if not date_str or pd.isna(date_str):
        return None
    s = str(date_str).strip()
    # Try most-specific first
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            continue
    # Last resort — pandas' flexible parser
    try:
        return pd.Timestamp(pd.to_datetime(s))
    except (ValueError, TypeError):
        return None


def first_trading_day_at_or_after(
    prices: pd.DataFrame, target: pd.Timestamp
) -> Optional[pd.Timestamp]:
    """Find the first row in ``prices`` whose index is >= ``target``.

    Necessary because trial dates land on weekends/holidays roughly 30% of the time.
    Returns ``None`` if ``target`` is past the end of the price series.
    """
    if prices.empty:
        return None
    later = prices.index[prices.index >= target]
    return later[0] if len(later) else None
