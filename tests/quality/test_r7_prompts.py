"""R7 prompt changes reach their stages."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from tradingagents.quality.prompts import ADJUDICATION, RISK_SCOPE


def _state():
    return {
        "company_of_interest": "ONDS", "instrument_context": "ONDS is Ondas Inc.", "trade_date": "2026-10-05",
        "market_report": "m", "sentiment_report": "s", "news_report": "n", "fundamentals_report": "f",
        "trader_investment_plan": "plan", "investment_plan": "rm",
        "investment_debate_state": {"history": "h", "bull_history": "b", "bear_history": "r",
                                    "current_response": "", "count": 2},
        "risk_debate_state": {"history": "", "aggressive_history": "", "conservative_history": "",
                              "neutral_history": "", "latest_speaker": "", "current_aggressive_response": "",
                              "current_conservative_response": "", "current_neutral_response": "", "count": 0},
    }


@pytest.mark.unit
@pytest.mark.parametrize("module, factory", [
    ("tradingagents.agents.risk_mgmt.aggressive_debator", "create_aggressive_debator"),
    ("tradingagents.agents.risk_mgmt.conservative_debator", "create_conservative_debator"),
    ("tradingagents.agents.risk_mgmt.neutral_debator", "create_neutral_debator"),
])
def test_risk_turns_carry_the_scope_and_cap(module, factory):
    import importlib

    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="turn")
    getattr(importlib.import_module(module), factory)(llm)(_state())
    prompt = llm.invoke.call_args[0][0]
    assert RISK_SCOPE in prompt and "under 500 words" in prompt


@pytest.mark.unit
def test_research_manager_adjudicates_the_numbers():
    from tradingagents.agents.managers.research_manager import create_research_manager

    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError
    llm.invoke.return_value = AIMessage(content="**Recommendation**: Hold\n\n**Rationale**: r")
    create_research_manager(llm)(_state())
    prompt = str(llm.invoke.call_args[0][0])
    assert "Adjudicate the numbers first" in prompt and "Disputed figures" in prompt
    assert ADJUDICATION[:40] in prompt


@pytest.mark.unit
def test_the_digest_reads_the_research_managers_ruling():
    """Upstream moved the ruling to investment_plan (cf960d6); the digest read
    the old key and lost it (code review, 2026-10-09)."""
    from tradingagents.graph.digest import _sources_from_state
    sources = dict(_sources_from_state({"investment_plan": "RULING: the bear case holds",
                                        "investment_debate_state": {"bull_history": "b"}}))
    assert "the bear case holds" in (sources["Research manager ruling"] or "")


@pytest.mark.unit
def test_pm_prompt_carries_the_rating_bands_anchor_and_catalyst_rules():
    """AMD review (2026-10-10): R7 fixes and R8 rules reach the PM."""
    from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager

    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError
    llm.invoke.return_value = AIMessage(content="**Rating**: Hold\n\n**Executive Summary**: s\n\n**Investment Thesis**: t")
    create_portfolio_manager(llm)(_state())
    prompt = str(llm.invoke.call_args[0][0])
    assert "target move between -12% and -4%" in prompt          # Underweight's band
    assert 'never "I"' in prompt                                  # voice
    assert "target_anchor" in prompt and "pe_hist.median" in prompt and "P/E" in prompt  # anchor menu, cross-check
    assert "valuation alone, with no catalyst view, is a Hold" in prompt


@pytest.mark.unit
def test_news_analyst_covers_competitors_and_cites_the_yield():
    from tradingagents.agents.analysts.news_analyst import create_news_analyst

    llm = MagicMock()
    bound = MagicMock()
    llm.bind_tools.return_value = bound
    captured = {}

    def fake_invoke(messages, *a, **k):
        captured["text"] = str(messages)
        return AIMessage(content="report")
    bound.invoke.side_effect = fake_invoke
    state = {**_state(), "messages": [], "asset_type": "stock"}
    try:
        create_news_analyst(llm)(state)
    except Exception:  # noqa: BLE001 — the prompt is all this test reads
        pass
    text = captured.get("text") or str(llm.bind_tools.call_args) + str(bound.mock_calls)
    assert "competitive picture" in text and "[F:macro.ust10y]" in text
