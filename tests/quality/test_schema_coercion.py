"""Model output the typed stages must survive (release review 2026-10-09):
one malformed field must not drop a whole structured decision, and a failed
sentiment parse must not leave a typed block without a score."""

from unittest.mock import MagicMock

import pytest

import tradingagents.agents.analysts.sentiment_analyst as sentiment
from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    ResearchPlan,
    TraderAction,
    TraderProposal,
)

from .test_social import RD, REPOST, ST


def _decision(**fields):
    return PortfolioDecision(**{"rating": "Hold", "executive_summary": "s", "investment_thesis": "t", **fields})


@pytest.mark.unit
@pytest.mark.parametrize(("raw", "expected"), [
    (None, []),
    ("ev_sales.ttm, ttm_revenue.2026Q2; shares.cover", ["ev_sales.ttm", "ttm_revenue.2026Q2", "shares.cover"]),
    ("['ev_sales.ttm', 'shares.cover']", ["ev_sales.ttm", "shares.cover"]),
    ("", []),
    (["ev_sales.ttm", None, 3], ["ev_sales.ttm", "3"]),
])
def test_valuation_inputs_take_none_and_a_comma_separated_string(raw, expected):
    assert _decision(valuation_inputs=raw).valuation_inputs == expected


@pytest.mark.unit
def test_ratings_and_actions_match_without_regard_to_case_or_markdown():
    assert _decision(rating="**overweight**").rating is PortfolioRating.OVERWEIGHT
    assert ResearchPlan(recommendation="SELL.", rationale="r", strategic_actions="a").recommendation is PortfolioRating.SELL
    assert TraderProposal(action=" buy ", reasoning="r").action is TraderAction.BUY
    with pytest.raises(ValueError):
        _decision(rating="Strong Buy")


@pytest.mark.unit
def test_text_fields_given_as_numbers_or_lists_are_kept():
    decision = _decision(target_math=["12x × $174.1M = $2.09B", "÷ 570.6M = $3.66"], execution_timing=3)
    assert decision.target_math == "12x × $174.1M = $2.09B; ÷ 570.6M = $3.66" and decision.execution_timing == "3"
    assert TraderProposal(action="Hold", reasoning="r", position_sizing=5).position_sizing == "5"


def _run_sentiment(monkeypatch, stocktwits: str, reddit: str, news: str):
    monkeypatch.setattr(sentiment, "fetch_stocktwits_messages", lambda *a, **k: stocktwits)
    monkeypatch.setattr(sentiment, "fetch_reddit_posts", lambda *a, **k: reddit)
    monkeypatch.setattr(sentiment.get_news, "func", lambda *a, **k: news, raising=False)
    monkeypatch.setattr(sentiment, "jev_screen", lambda *_a: None)
    monkeypatch.setattr(sentiment, "get_config", lambda: {"sentiment_rules": True})
    structured = MagicMock()
    structured.invoke.side_effect = ValueError("malformed JSON")
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    llm.invoke.return_value.content = "**Overall Sentiment:** Bullish (Score: 7/10)\nPosts are upbeat."
    return sentiment.create_sentiment_analyst(llm)({
        "company_of_interest": "ONDS", "trade_date": "2026-10-05", "asset_type": "stock", "messages": []})


@pytest.mark.unit
def test_a_failed_parse_on_a_scoreable_sample_leaves_no_typed_block(monkeypatch):
    out = _run_sentiment(monkeypatch, ST + "\n" + REPOST, RD, "Ondas wins an order")
    assert out["sentiment_structured"] is None
    assert out["sentiment_report"].startswith("**Sample:**")


@pytest.mark.unit
def test_an_insufficient_sample_states_its_reason_once(monkeypatch):
    out = _run_sentiment(monkeypatch, ST, "", "")
    assert out["sentiment_structured"]["overall_score"] is None
    header = out["sentiment_report"].split("\n")[1]
    assert header.startswith("**Overall Sentiment:** insufficient data: ")
    assert "insufficient data (insufficient data" not in out["sentiment_report"]
