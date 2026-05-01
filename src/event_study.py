"""Event-study layer — join trials to market data and compute event-window returns.

Phase 2.1.

Inputs:
  * `trials_df` from Phase 1.2 (one row per NCT)
  * `ticker` whose stock should be measured
  * benchmark ticker (default IBB)

Output:
  A DataFrame with one row per trial, columns:
    nct_id, ticker, event_date_used, event_date_source,
    ret_<-N>_0, ret_0_<+N> for each window,
    abnormal_ret_<...> = stock minus benchmark over the same window

Phase-3 backtest code joins this to a per-trial safety score and looks for
correlation between score and forward return.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.market_data import (
    DEFAULT_BENCHMARK,
    first_trading_day_at_or_after,
    get_ohlcv,
    parse_event_date,
)


# Trading-day windows. (start, end) is inclusive of both endpoints.
DEFAULT_WINDOWS: list[tuple[int, int]] = [
    (-5, 0),   # pre-event leakage check
    (0, 5),    # immediate reaction (~1 week)
    (0, 20),   # ~1 month
    (0, 60),   # ~3 months
]


# ---------------------------------------------------------------------------

def pick_event_date(trial: pd.Series) -> tuple[Optional[pd.Timestamp], Optional[str]]:
    """Return ``(date, source_label)`` for a trial row.

    Prefer ``results_first_posted`` (when CT.gov made the data public — closest
    to when the market could react). Fall back to ``completion_date`` (when the
    trial ended; may precede market awareness by months).
    """
    rfp = parse_event_date(trial.get("results_first_posted"))
    if rfp is not None:
        return rfp, "results_first_posted"
    cd = parse_event_date(trial.get("completion_date"))
    if cd is not None:
        return cd, "completion_date"
    return None, None


def windowed_return(
    prices: pd.DataFrame,
    anchor_date: pd.Timestamp,
    start_offset: int,
    end_offset: int,
) -> Optional[float]:
    """Compounded close-to-close return between ``anchor + start_offset`` trading
    days and ``anchor + end_offset`` trading days. Returns ``None`` if either
    endpoint falls outside the available price series.
    """
    if prices.empty:
        return None
    anchor = first_trading_day_at_or_after(prices, anchor_date)
    if anchor is None:
        return None
    try:
        anchor_idx = prices.index.get_loc(anchor)
    except KeyError:
        return None

    start_idx = anchor_idx + start_offset
    end_idx = anchor_idx + end_offset
    if start_idx < 0 or end_idx >= len(prices) or start_idx >= len(prices):
        return None
    p_start = prices["Close"].iloc[start_idx]
    p_end = prices["Close"].iloc[end_idx]
    if pd.isna(p_start) or pd.isna(p_end) or p_start == 0:
        return None
    return float(p_end / p_start - 1.0)


def attach_event_returns(
    trials_df: pd.DataFrame,
    ticker: str,
    benchmark: str = DEFAULT_BENCHMARK,
    windows: list[tuple[int, int]] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """For every trial in ``trials_df``, compute event-window returns on
    ``ticker`` and abnormal returns vs ``benchmark`` over the same windows.

    Trials with no resolvable event date or with prices outside the available
    history are kept (so you can audit them) but their return columns are NaN.
    """
    if trials_df.empty:
        return pd.DataFrame()

    stock = get_ohlcv(ticker)
    bench = get_ohlcv(benchmark)

    rows: list[dict] = []
    for _, t in trials_df.iterrows():
        event_date, source = pick_event_date(t)
        row: dict = {
            "nct_id": t.get("nct_id"),
            "ticker": ticker.upper(),
            "benchmark": benchmark.upper(),
            "event_date_used": event_date,
            "event_date_source": source,
        }
        if event_date is None:
            for s, e in windows:
                row[_ret_col(s, e)] = None
                row[_abret_col(s, e)] = None
            rows.append(row)
            continue

        for s, e in windows:
            r_stock = windowed_return(stock, event_date, s, e)
            r_bench = windowed_return(bench, event_date, s, e)
            row[_ret_col(s, e)] = r_stock
            row[_abret_col(s, e)] = (
                r_stock - r_bench if (r_stock is not None and r_bench is not None) else None
            )
        rows.append(row)

    return pd.DataFrame(rows)


def _ret_col(s: int, e: int) -> str:
    return f"ret_{s}_{e}"


def _abret_col(s: int, e: int) -> str:
    return f"abret_{s}_{e}"
