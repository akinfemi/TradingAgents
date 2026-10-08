"""Stage gates (R5): lint, one fix-up turn, errata forward."""

import json
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from tradingagents.quality import edgar_ext, facts, gates

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def state():
    st = edgar_ext.from_json("0001646188", json.loads((FIX / "onds_companyfacts.json").read_text()),
                             json.loads((FIX / "onds_submissions.json").read_text()), "2026-10-05")
    sheet = facts.build("ONDS", "2026-10-05", "2026-10-05T10:33:00Z", statements=st,
                        ohlcv=pd.read_csv(FIX / "onds_ohlcv.csv"), offline=True)
    return {"fact_sheet": sheet.model_dump(mode="json"), "fact_sheet_text": facts.render(sheet),
            "company_of_interest": "ONDS", "instrument_context": "ONDS is Ondas Inc."}


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def invoke(self, prompt, config=None):
        self.prompts.append(prompt)
        return AIMessage(content=self.reply)


WRONG = "The MACD shows a bearish cross, and R&D was $23.4M [F:rnd.2026Q2] last quarter. " + ("Context. " * 20).strip()
RIGHT = "The MACD crossed bullish on 2026-09-21, and R&D was $31.0M [F:rnd.2026Q2] last quarter. " + ("Context. " * 20).strip()


@pytest.mark.unit
def test_market_report_is_fixed_before_the_debate_reads_it(state):
    llm = FakeLLM(RIGHT)
    out = gates.analyst_check("market_report", "market_analyst", llm, "market analyst")({**state, "market_report": WRONG})
    assert out["market_report"] == RIGHT
    assert "bearish cross" in llm.prompts[0] and "R&D was $23.4M" in llm.prompts[0]
    assert "open_errata" not in out
    assert out["quality_gates"][0]["flags_before"] == 2 and out["quality_gates"][0]["flags_after"] == 0


@pytest.mark.unit
def test_an_unfixed_error_goes_forward_as_errata(state):
    out = gates.analyst_check("market_report", "market_analyst", FakeLLM(WRONG), "")({**state, "market_report": WRONG})
    assert "market_report" not in out
    assert {e["kind"] for e in out["open_errata"]} == {"direction", "cited_mismatch"}
    from tradingagents.agents.context import get_instrument_context_from_state

    later = get_instrument_context_from_state({**state, "open_errata": out["open_errata"]})
    assert "Open errata" in later and "bearish cross" in later


@pytest.mark.unit
def test_a_truncated_rewrite_is_rejected(state):
    out = gates.analyst_check("market_report", "market_analyst", FakeLLM("Fixed."), "")({**state, "market_report": WRONG})
    assert "market_report" not in out and out["quality_gates"][0]["rewrite_rejected"]


@pytest.mark.unit
def test_a_clean_report_costs_no_call(state):
    llm = FakeLLM("unused")
    out = gates.analyst_check("market_report", "market_analyst", llm, "")({**state, "market_report": RIGHT})
    assert llm.prompts == [] and "market_report" not in out


@pytest.mark.unit
def test_no_fact_sheet_no_gate():
    llm = FakeLLM("unused")
    out = gates.analyst_check("market_report", "market_analyst", llm, "")({"market_report": WRONG})
    assert llm.prompts == [] and out["quality_gates"][0]["checked"] is False


@pytest.mark.unit
def test_a_debate_turn_is_fixed_in_every_history(state):
    def bear(_state):
        turn = f"Bear Analyst: {WRONG}"
        return {"investment_debate_state": {"history": "Bull Analyst: hi\n" + turn, "bear_history": "\n" + turn,
                                            "bull_history": "Bull Analyst: hi", "current_response": turn, "count": 2}}

    out = gates.debate_turn(bear, "bear_researcher", "investment_debate_state", "current_response", "bear_history",
                            FakeLLM(RIGHT), "bear researcher")(state)
    debate = out["investment_debate_state"]
    assert debate["current_response"] == f"Bear Analyst: {RIGHT}"
    assert debate["history"].endswith(f"Bear Analyst: {RIGHT}") and debate["history"].startswith("Bull Analyst: hi")
    assert WRONG not in debate["bear_history"]


@pytest.mark.unit
def test_managers_are_linted_not_rewritten(state):
    def pm(_state):
        return {"final_trade_decision": WRONG}

    llm = FakeLLM(RIGHT)
    out = gates.text_stage(pm, "portfolio_manager", "final_trade_decision", llm, "pm", fix=False)(state)
    assert out["final_trade_decision"] == WRONG and llm.prompts == []
    assert out["open_errata"]


@pytest.mark.unit
@pytest.mark.parametrize("gated", [False, True])
def test_the_graph_compiles_with_and_without_gates(gated):
    from unittest.mock import MagicMock

    from tradingagents.graph.conditional_logic import ConditionalLogic
    from tradingagents.graph.setup import GraphSetup

    setup = GraphSetup(MagicMock(), MagicMock(), ConditionalLogic(), 3, quality_gates=gated)
    graph = setup.setup_graph(("market", "news")).compile()
    nodes = set(graph.get_graph().nodes)
    assert ("Market Analyst Check" in nodes) is gated
    assert ("News Analyst Check" in nodes) is gated
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    exit_ = "Market Analyst Check" if gated else "Market Analyst"
    assert (exit_, "Bull Researcher") in edges


@pytest.mark.unit
def test_streaming_keeps_every_gate_record():
    """Staging, 2026-10-08: the progress stream's merge kept only the last
    node's gate record; the graph's reducer appends them all."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    graph = object.__new__(TradingAgentsGraph)

    class FakeCompiled:
        def stream(self, _input, **_args):
            yield {"Market Analyst Check": {"quality_gates": [{"stage": "market"}], "open_errata": [{"stage": "market"}]}}
            yield {"Bear Researcher": {"quality_gates": [{"stage": "bear"}]}}
            yield {"Portfolio Manager": {"quality_gates": [{"stage": "pm"}], "final_trade_decision": "x"}}

    graph.graph = FakeCompiled()
    final = graph._stream_with_progress({}, {"quality_gates": [], "open_errata": []}, {}, lambda *a: None)
    assert [g["stage"] for g in final["quality_gates"]] == ["market", "bear", "pm"]
    assert [e["stage"] for e in final["open_errata"]] == ["market"]
    assert final["final_trade_decision"] == "x"
