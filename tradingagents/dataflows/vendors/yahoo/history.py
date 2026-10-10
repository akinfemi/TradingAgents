"""Split-only closes and the split list, for the fact sheet's valuation
history (REPORT_QUALITY_PLAN R8).

The run's OHLCV frame is adjusted for dividends as well as splits (Yahoo's
``auto_adjust``, Tiingo's ``adjClose``), which reads a dividend payer's past
multiples low: Realty Income's close at 2021-12-31 was $72.25, its adjusted
close $56.30. A multiple needs the price as it traded, adjusted only for
splits, beside a share count put on the same split basis.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.symbols import normalize_symbol, safe_ticker_component
from tradingagents.dataflows.files import replace_file
from tradingagents.dataflows.vendors.yahoo.common import raise_for_empty, yf_retry
from tradingagents.dataflows.vendors.yahoo.ohlcv import _cache_is_fresh, _ensure_date_column

logger = logging.getLogger(__name__)

YEARS = 6


def split_history(symbol: str, as_of_date: str) -> tuple[pd.DataFrame, list[tuple[str, float]]]:
    """(frame of Date and split-only Close up to ``as_of_date``, [(split date, ratio)]
    to today: the basis the closes are on). One Yahoo request per symbol per
    day, cached."""
    canonical = normalize_symbol(symbol)
    safe_symbol = safe_ticker_component(canonical)
    config = get_config()
    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(config["data_cache_dir"], f"{safe_symbol}-YFin-splitonly.csv")
    now = pd.Timestamp.today()
    as_of_dt = pd.to_datetime(as_of_date).normalize()

    data = None
    if os.path.exists(data_file):
        cached = pd.read_csv(data_file, on_bad_lines="skip", encoding="utf-8")
        if not cached.empty and "Close" in cached.columns and _cache_is_fresh(data_file, as_of_dt, now):
            data = cached
    if data is None:
        downloaded = yf_retry(lambda: yf.Ticker(canonical).history(
            start=(now - pd.DateOffset(years=YEARS)).strftime("%Y-%m-%d"),
            end=(now + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False,
            actions=True,
        ))
        if downloaded is None or downloaded.empty:
            raise_for_empty(symbol, canonical, "price rows")
        downloaded = _ensure_date_column(downloaded.reset_index())
        keep = [c for c in ("Date", "Close", "Stock Splits") if c in downloaded.columns]
        downloaded = downloaded[keep]
        replace_file(data_file, lambda temp: downloaded.to_csv(temp, index=False, encoding="utf-8"))
        data = downloaded

    dates = pd.to_datetime(data["Date"], utc=True).dt.tz_localize(None).dt.normalize()
    frame = pd.DataFrame({"Date": dates.dt.strftime("%Y-%m-%d"), "Close": data["Close"].astype(float)})
    mask = dates <= as_of_dt
    frame = frame[mask].reset_index(drop=True)
    # Every split to today, not only to the as-of date: the closes are on
    # today's split basis, so a later split still scales an earlier count
    # (NVDA as of 2024-06-07: closes on the post-split basis, counts not).
    splits: list[tuple[str, float]] = []
    if "Stock Splits" in data.columns:
        all_days = dates.dt.strftime("%Y-%m-%d")
        for day, ratio in zip(all_days, data["Stock Splits"].fillna(0).astype(float)):
            if ratio and ratio > 0 and ratio != 1:
                splits.append((day, ratio))
    return frame, splits
