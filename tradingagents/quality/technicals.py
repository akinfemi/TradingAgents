"""Price facts computed in code from the run's own OHLCV (REPORT_QUALITY_PLAN R0).

Agents interpret indicators; they never compute them. These facts come from
the same cached frame the analysts' tools read (``load_ohlcv``), with the
same stockstats indicators, so the digest can be anchored to them: a
"bearish MACD cross" when MACD sits above its signal, or a "$1.68 gap" that
is really $2.20, gets corrected instead of repeated.

R4's fact sheet reuses ``price_facts``; only price-derived facts live here.
"""

from __future__ import annotations

import glob
import logging
import os

import pandas as pd
from stockstats import wrap

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.stockstats_utils import load_ohlcv
from tradingagents.dataflows.symbol_utils import normalize_symbol
from tradingagents.dataflows.utils import safe_ticker_component

logger = logging.getLogger(__name__)

_YEAR_ROWS = 252


def _price_source(symbol: str) -> str:
    """Which vendor's cache the frame came from (the cache file names it)."""
    safe = safe_ticker_component(normalize_symbol(symbol))
    cache_dir = get_config()["data_cache_dir"]
    if glob.glob(os.path.join(cache_dir, f"{safe}-Tiingo-data-*.csv")):
        return "Tiingo"
    return "Yahoo Finance"


def _money(v: float) -> str:
    return f"${v:,.2f}"


def _pct(v: float) -> str:
    return f"{v:+.1f}%"


def price_facts(symbol: str, trade_date: str) -> list[tuple[str, str]]:
    """[(label, value)] computed from the run's OHLCV up to trade_date.
    Empty when there is no usable price history."""
    frame = load_ohlcv(symbol, trade_date)
    if frame is None or frame.empty or len(frame) < 2:
        return []
    df = wrap(frame.copy())
    for col in ("close_10_ema", "close_50_sma", "close_200_sma", "macd", "macds", "rsi_14"):
        df[col]  # stockstats computes on access
    dates = pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d").tolist()
    close = float(frame["Close"].iloc[-1])
    asof = dates[-1]

    facts: list[tuple[str, str]] = [
        (f"Share price (last close used, {asof})", _money(close)),
        ("Price source", _price_source(symbol)),
    ]

    for col, label in (
        ("close_10_ema", "10-day EMA"),
        ("close_50_sma", "50-day SMA"),
        ("close_200_sma", "200-day SMA"),
    ):
        window = int(col.split("_")[1])
        if len(frame) < window:
            continue
        avg = float(df[col].iloc[-1])
        gap = close - avg
        side = "above" if gap >= 0 else "below"
        facts.append(
            (label, f"{_money(avg)}; price is {_money(abs(gap))} ({_pct(gap / avg * 100)}) {side} it")
        )

    macd = df["macd"].astype(float).tolist()
    signal = df["macds"].astype(float).tolist()
    if len(macd) >= 35:
        state = "above" if macd[-1] > signal[-1] else "below"
        cross_date, cross_kind = None, None
        for i in range(len(macd) - 1, 0, -1):
            prev = macd[i - 1] - signal[i - 1]
            curr = macd[i] - signal[i]
            if prev == 0 or curr == 0 or (prev > 0) != (curr > 0):
                cross_date = dates[i]
                cross_kind = "bullish" if curr > 0 else "bearish"
                break
        line = (
            f"MACD {macd[-1]:.3f} is {state} its signal line {signal[-1]:.3f} "
            f"(histogram {macd[-1] - signal[-1]:+.3f})"
        )
        if cross_date:
            line += f"; last crossover was {cross_kind}, on {cross_date}"
        facts.append(("MACD", line))

    if len(frame) >= 15:
        facts.append(("RSI (14-day)", f"{float(df['rsi_14'].iloc[-1]):.1f}"))

    year = frame.tail(_YEAR_ROWS)
    year_dates = dates[-len(year):]
    closes = year["Close"].astype(float).tolist()
    hi, lo = max(closes), min(closes)
    hi_date, lo_date = year_dates[closes.index(hi)], year_dates[closes.index(lo)]
    span = "52-week" if len(year) >= _YEAR_ROWS else f"{len(year)}-session"
    facts.append(
        (f"{span} high", f"{_money(hi)} on {hi_date}; price is {_pct((close / hi - 1) * 100)} from it")
    )
    facts.append(
        (f"{span} low", f"{_money(lo)} on {lo_date}; price is {_pct((close / lo - 1) * 100)} from it")
    )
    return facts


def computed_context(symbol: str, trade_date: str) -> str | None:
    """The digest's authoritative-figures block, or None. Never raises:
    the digest runs without it rather than failing the run."""
    try:
        facts = price_facts(symbol, trade_date)
    except Exception as exc:  # noqa: BLE001 — decoration input, never fatal
        logger.warning("price facts for %s unavailable (%s)", symbol, exc)
        return None
    if not facts:
        return None
    return "\n".join(f"- {label}: {value}" for label, value in facts)
