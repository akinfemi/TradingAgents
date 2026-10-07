"""Tiingo end-of-day prices (licensed price source).

Selected with ``config["price_vendor"] = "tiingo"`` (or the
``TRADINGAGENTS_PRICE_VENDOR`` env var); the key is ``TIINGO_API_KEY``.
Returns the same frame shape ``load_ohlcv`` caches for yfinance
(Date, Open, High, Low, Close, Volume, split- and dividend-adjusted), so
every consumer — indicators, the verified snapshot, ``get_stock_data`` —
reads one price source per run.

Tiingo's redistribution terms require the attribution "Data sourced by
Tiingo" wherever its data is displayed; the server adds it on report
surfaces.
"""

import os

import pandas as pd
import requests

from .symbol_utils import NoMarketDataError

TIINGO_BASE_URL = "https://api.tiingo.com/tiingo/daily"
_TIMEOUT_SECONDS = 30

# Adjusted columns match yfinance's auto_adjust=True frames.
_COLUMNS = {
    "adjOpen": "Open",
    "adjHigh": "High",
    "adjLow": "Low",
    "adjClose": "Close",
    "adjVolume": "Volume",
}


def price_vendor(config: dict | None = None) -> str:
    """The configured OHLCV vendor: env wins over config, default yfinance."""
    env = (os.getenv("TRADINGAGENTS_PRICE_VENDOR") or "").strip().lower()
    if env:
        return env
    return str((config or {}).get("price_vendor") or "yfinance").lower()


def tiingo_symbol(canonical: str) -> str | None:
    """Tiingo's ticker for a Yahoo-canonical symbol, or None when Tiingo
    doesn't list that kind of instrument (indices, futures, FX, crypto and
    non-US exchange suffixes) — the caller falls back to yfinance."""
    sym = canonical.upper()
    if sym.startswith("^") or "=" in sym or "." in sym:
        return None
    if sym.endswith("-USD"):  # crypto pairs live on a different Tiingo API
        return None
    return sym  # Tiingo uses the same dash form for share classes (BRK-B)


def fetch_daily(symbol: str, start: str, end: str, api_key: str | None = None) -> pd.DataFrame:
    """Adjusted daily OHLCV for ``symbol`` between start and end (inclusive).

    Raises NoMarketDataError when Tiingo has no rows or doesn't list the
    symbol, so the router's existing no-data handling applies.
    """
    key = api_key or os.getenv("TIINGO_API_KEY") or ""
    if not key:
        raise NoMarketDataError(symbol, symbol, "TIINGO_API_KEY is not set")
    tsym = tiingo_symbol(symbol)
    if tsym is None:
        raise NoMarketDataError(symbol, symbol, "Tiingo does not list this instrument")
    resp = requests.get(
        f"{TIINGO_BASE_URL}/{tsym}/prices",
        params={"startDate": start, "endDate": end, "resampleFreq": "daily"},
        headers={"Authorization": f"Token {key}", "Content-Type": "application/json"},
        timeout=_TIMEOUT_SECONDS,
    )
    if resp.status_code == 404:
        raise NoMarketDataError(symbol, tsym, "Tiingo does not list this ticker")
    resp.raise_for_status()
    rows = resp.json() or []
    if not rows:
        raise NoMarketDataError(symbol, tsym, f"Tiingo returned no rows {start}..{end}")
    frame = pd.DataFrame(rows)
    missing = [c for c in ("date", *_COLUMNS) if c not in frame.columns]
    if missing:
        raise NoMarketDataError(symbol, tsym, f"Tiingo response missing {missing}")
    out = frame[["date", *_COLUMNS]].rename(columns={"date": "Date", **_COLUMNS})
    out["Date"] = pd.to_datetime(out["Date"], utc=True).dt.tz_localize(None).dt.normalize()
    return out.sort_values("Date").reset_index(drop=True)
