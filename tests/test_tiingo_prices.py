"""Tiingo as the licensed OHLCV source: parsing, routing, Yahoo fallback."""

import pandas as pd
import pytest

import tradingagents.dataflows.stockstats_utils as su
import tradingagents.dataflows.tiingo as tiingo
import tradingagents.dataflows.y_finance as yfin
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.symbol_utils import NoMarketDataError


def _rows(days):
    return [
        {
            "date": f"{d}T00:00:00.000Z",
            "adjOpen": 10.0 + i,
            "adjHigh": 11.0 + i,
            "adjLow": 9.0 + i,
            "adjClose": 10.5 + i,
            "adjVolume": 1000 + i,
            "close": 99.0,  # raw columns must be ignored in favour of adjusted
        }
        for i, d in enumerate(days)
    ]


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


@pytest.fixture
def tiingo_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TIINGO_API_KEY", "test-key")
    monkeypatch.delenv("TRADINGAGENTS_PRICE_VENDOR", raising=False)
    set_config({"data_cache_dir": str(tmp_path), "price_vendor": "tiingo"})
    yield tmp_path
    set_config({"price_vendor": "yfinance"})


def test_fetch_daily_uses_adjusted_columns_and_token_header(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "k")
    seen = {}

    def fake_get(url, params, headers, timeout):
        seen.update(url=url, params=params, headers=headers)
        return _Resp(_rows(["2026-10-01", "2026-10-02"]))

    monkeypatch.setattr(tiingo.requests, "get", fake_get)
    frame = tiingo.fetch_daily("ONDS", "2026-09-01", "2026-10-02")
    assert list(frame.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert frame["Close"].tolist() == [10.5, 11.5]
    assert seen["url"].endswith("/ONDS/prices")
    assert seen["headers"]["Authorization"] == "Token k"
    assert "token" not in seen["params"]  # key travels in the header, not the URL


@pytest.mark.parametrize("sym", ["^GSPC", "RELIANCE.NS", "GC=F", "BTC-USD"])
def test_unlisted_instruments_are_not_sent_to_tiingo(sym):
    assert tiingo.tiingo_symbol(sym) is None


def test_missing_key_is_no_data(monkeypatch):
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    with pytest.raises(NoMarketDataError):
        tiingo.fetch_daily("AAPL", "2026-01-01", "2026-02-01")


def test_404_is_no_data(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "k")
    monkeypatch.setattr(tiingo.requests, "get", lambda *a, **k: _Resp({}, status=404))
    with pytest.raises(NoMarketDataError):
        tiingo.fetch_daily("ZZZZ", "2026-01-01", "2026-02-01")


def test_env_overrides_config(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_PRICE_VENDOR", "Tiingo")
    assert tiingo.price_vendor({"price_vendor": "yfinance"}) == "tiingo"
    monkeypatch.delenv("TRADINGAGENTS_PRICE_VENDOR")
    assert tiingo.price_vendor({}) == "yfinance"


def test_load_ohlcv_routes_to_tiingo_and_caches_by_vendor(monkeypatch, tiingo_env):
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    calls = []

    def fake_fetch(symbol, start, end, api_key=None):
        calls.append(symbol)
        return pd.DataFrame(
            {
                "Date": pd.to_datetime([today]),
                "Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [10],
            }
        )

    def no_yahoo(*a, **k):
        raise AssertionError("Yahoo must not be called when Tiingo has the symbol")

    monkeypatch.setattr(su, "fetch_tiingo_daily", fake_fetch)
    monkeypatch.setattr(su.yf, "download", no_yahoo)

    data = su.load_ohlcv("AAPL", today)
    assert data["Close"].iloc[-1] == 1.5
    assert list(tiingo_env.glob("AAPL-Tiingo-data-*.csv"))
    assert not list(tiingo_env.glob("AAPL-YFin-data-*.csv"))

    su.load_ohlcv("AAPL", today)  # second read comes from the Tiingo cache
    assert calls == ["AAPL"]


def test_load_ohlcv_falls_back_to_yahoo_when_tiingo_has_nothing(monkeypatch, tiingo_env):
    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    def empty_tiingo(symbol, start, end, api_key=None):
        raise NoMarketDataError(symbol, symbol, "none")

    def fake_download(*a, **k):
        return pd.DataFrame(
            {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [7.0], "Volume": [5]},
            index=pd.DatetimeIndex([pd.Timestamp(today)], name="Date"),
        )

    monkeypatch.setattr(su, "fetch_tiingo_daily", empty_tiingo)
    monkeypatch.setattr(su.yf, "download", fake_download)
    data = su.load_ohlcv("AAPL", today)
    assert data["Close"].iloc[-1] == 7.0
    assert list(tiingo_env.glob("AAPL-YFin-data-*.csv"))
    assert not list(tiingo_env.glob("AAPL-Tiingo-data-*.csv"))


def test_get_stock_data_reads_tiingo_and_names_the_source(monkeypatch, tiingo_env):
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-09-30", "2026-10-01", "2026-10-02"]),
            "Open": [1.0, 2.0, 3.0], "High": [1.0, 2.0, 3.0], "Low": [1.0, 2.0, 3.0],
            "Close": [1.0, 2.0, 3.0], "Volume": [1, 2, 3],
        }
    )
    monkeypatch.setattr(yfin, "load_ohlcv", lambda sym, d: frame)
    monkeypatch.setattr(
        yfin.yf, "Ticker", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no Yahoo"))
    )
    out = yfin.get_YFin_data_online("ONDS", "2026-10-01", "2026-10-02")
    assert "# Source: Tiingo" in out
    assert "2026-09-30" not in out and "2026-10-02" in out
