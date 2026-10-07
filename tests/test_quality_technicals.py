"""Price facts computed in code anchor the digest (REPORT_QUALITY_PLAN R0)."""

import numpy as np
import pandas as pd
import pytest

import tradingagents.quality.technicals as tech
from tradingagents.graph.digest import build_digest_prompt


def _frame(closes):
    dates = pd.bdate_range("2025-01-01", periods=len(closes))
    return pd.DataFrame(
        {"Date": dates, "Open": closes, "High": closes, "Low": closes, "Close": closes,
         "Volume": [1000] * len(closes)}
    )


@pytest.fixture
def stub_prices(monkeypatch, tmp_path):
    from tradingagents.dataflows.config import set_config

    set_config({"data_cache_dir": str(tmp_path)})

    def use(closes):
        monkeypatch.setattr(tech, "load_ohlcv", lambda s, d, fill_gaps=True: _frame(closes))

    return use


def test_gaps_name_their_side_and_size(stub_prices):
    # Long decline: price ends below every average.
    stub_prices(list(np.linspace(20, 10, 260)))
    facts = dict(tech.price_facts("TEST", "2025-12-31"))
    assert "below it" in facts["200-day SMA"]
    last = _frame([0] * 260)["Date"].iloc[-1].strftime("%Y-%m-%d")
    assert facts[f"Share price (last close used, {last})"] == "$10.00"
    assert facts["Price source"] == "Yahoo Finance"


def test_macd_direction_and_last_cross_are_computed(stub_prices):
    # Down, then a sharp turn up: MACD crosses above its signal (bullish).
    closes = list(np.linspace(30, 10, 200)) + list(np.linspace(10, 16, 40))
    stub_prices(closes)
    macd = dict(tech.price_facts("TEST", "2025-12-31"))["MACD"]
    assert "is above its signal line" in macd
    assert "last crossover was bullish" in macd


def test_short_history_skips_long_averages(stub_prices):
    stub_prices([10.0 + i * 0.1 for i in range(30)])
    facts = dict(tech.price_facts("TEST", "2025-02-11"))
    assert "200-day SMA" not in facts and "50-day SMA" not in facts
    assert any(k.endswith("-session high") for k in facts)


def test_tiingo_cache_is_named_as_the_source(stub_prices, tmp_path):
    stub_prices([10.0 + i * 0.1 for i in range(30)])
    (tmp_path / "TEST-Tiingo-data.csv").write_text("x")
    assert dict(tech.price_facts("TEST", "2025-02-11"))["Price source"] == "Tiingo"


def test_context_never_raises(monkeypatch):
    def boom(s, d, fill_gaps=True):
        raise RuntimeError("vendor down")

    monkeypatch.setattr(tech, "load_ohlcv", boom)
    assert tech.computed_context("TEST", "2025-02-11") is None


def test_digest_prompt_puts_computed_indicators_above_transcripts():
    prompt = build_digest_prompt(
        {"company_of_interest": "ONDS", "trade_date": "2026-10-05",
         "market_report": "MACD bearish cross (Signal -0.182 above MACD -0.169)"},
        computed_context="- MACD: MACD -0.169 is above its signal line -0.182",
    )
    assert "computed figures win" in prompt
    assert "Never name the pipeline's internal roles" in prompt
    assert "at most 200" in prompt
    assert prompt.index("## Computed figures") < prompt.index("## Market analyst report")
