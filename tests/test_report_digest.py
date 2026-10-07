"""Tests for the post-run report digest (tradingagents/graph/digest.py).

The digest is optional decoration: it must extract cleanly from a full
state, clip oversized transcripts instead of shipping them whole, and
return None — never raise — on any provider failure.
"""

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.schemas import DigestPoint, ReportDigest, RiskLens
from tradingagents.graph.digest import (
    _SECTION_CHAR_BUDGET,
    build_digest_prompt,
    generate_report_digest,
)


def _sample_digest() -> ReportDigest:
    return ReportDigest(
        headline="Overweight: operating strength outweighs capex risk",
        bull_thesis="Growth is already in the numbers.",
        bull_points=[DigestPoint(title="Revenue compounding", detail="~18% YoY on a $400B base.")],
        bear_thesis="No margin of safety at 28x forward.",
        bear_points=[DigestPoint(title="Valuation", detail="28x forward P/E, 11x book.")],
        ruling="The bull side won on current operating evidence, but valuation caps conviction.",
        debate_winner="bull",
        risk_neutral=RiskLens(stance="Own it, size with discipline.", summary="Compounding, but capex payback is unproven."),
        conviction=68,
        conviction_note="Measured — bull won, not decisively",
        exit_triggers=[DigestPoint(title="Margin compression", detail="Capex eroding operating margin for 2+ quarters.")],
    )


def _state(**overrides) -> dict:
    state = {
        "company_of_interest": "GOOG",
        "trade_date": "2026-05-01",
        "market_report": "Price above the 50- and 200-day averages.",
        "sentiment_report": "Retail sentiment net-positive.",
        "news_report": "Cloud momentum dominates coverage.",
        "fundamentals_report": "Margins elite; debt stepped up.",
        "investment_debate_state": {
            "bull_history": "Bull: growth is real.",
            "bear_history": "Bear: too expensive.",
            "judge_decision": "Bull wins, gradually.",
        },
        "trader_investment_plan": "BUY in 3-4 tranches.",
        "risk_debate_state": {
            "aggressive_history": "Lean in.",
            "neutral_history": "Own it, capped.",
            "conservative_history": "Fragile setup.",
            "judge_decision": "Side with neutral.",
        },
        "final_trade_decision": "BUY, overweight gradually.",
    }
    state.update(overrides)
    return state


@pytest.mark.unit
class TestBuildDigestPrompt:
    def test_includes_every_present_section(self):
        prompt = build_digest_prompt(_state())
        for label in (
            "Market analyst report",
            "Bull researcher argument",
            "Bear researcher argument",
            "Research manager ruling",
            "Trader plan",
            "Conservative risk analyst",
            "Final portfolio decision",
        ):
            assert label in prompt
        assert "GOOG" in prompt and "2026-05-01" in prompt

    def test_omits_missing_sections(self):
        prompt = build_digest_prompt(_state(market_report=None, risk_debate_state={}))
        assert "Market analyst report" not in prompt
        assert "Aggressive risk analyst" not in prompt

    def test_clips_oversized_sections_keeping_head_and_tail(self):
        long_report = "HEAD " + ("x" * (_SECTION_CHAR_BUDGET * 3)) + " TAIL"
        prompt = build_digest_prompt(_state(market_report=long_report))
        assert "HEAD" in prompt and "TAIL" in prompt
        assert "elided for length" in prompt
        assert len(prompt) < _SECTION_CHAR_BUDGET * 3


@pytest.mark.unit
class TestGenerateReportDigest:
    def test_returns_plain_dict_on_success(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = _sample_digest()
        out = generate_report_digest(llm, _state())
        assert isinstance(out, dict)
        assert out["debate_winner"] == "bull"
        assert out["conviction"] == 68
        assert out["bull_points"][0]["title"] == "Revenue compounding"
        # None-able lenses serialize as None, not missing keys.
        assert out["risk_aggressive"] is None

    def test_forwards_callbacks_to_invocation(self):
        llm = MagicMock()
        structured = llm.with_structured_output.return_value
        structured.invoke.return_value = _sample_digest()
        cb = [object()]
        generate_report_digest(llm, _state(), callbacks=cb)
        assert structured.invoke.call_args.kwargs["config"] == {"callbacks": cb}

    def test_none_on_invocation_failure(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("boom")
        assert generate_report_digest(llm, _state()) is None

    def test_none_on_null_parse(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = None
        assert generate_report_digest(llm, _state()) is None

    def test_none_when_provider_lacks_structured_output(self):
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError
        assert generate_report_digest(llm, _state()) is None


@pytest.mark.unit
class TestComputedContext:
    def test_prompt_includes_authoritative_block_and_basis_rules(self):
        ctx = "- Share price (close 2026-04-30): $382.99\n- P/E (price/FY25 EPS): 35.4x"
        prompt = build_digest_prompt(_state(), computed_context=ctx)
        assert "Computed figures (authoritative" in prompt
        assert "P/E (price/FY25 EPS): 35.4x" in prompt
        assert "THESE are correct" in prompt
        # Consistency rules present regardless of context.
        assert "Label the basis" in prompt
        assert "sanity-checking" in prompt

    def test_prompt_omits_block_without_context(self):
        prompt = build_digest_prompt(_state())
        assert "Computed figures" not in prompt
        assert "Label the basis" in prompt

    def test_generate_forwards_computed_context(self):
        llm = MagicMock()
        structured = llm.with_structured_output.return_value
        structured.invoke.return_value = _sample_digest()
        generate_report_digest(llm, _state(), computed_context="- P/E: 35.4x")
        prompt = structured.invoke.call_args.args[0]
        assert "- P/E: 35.4x" in prompt
