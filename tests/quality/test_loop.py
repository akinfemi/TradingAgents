"""The quality loop (R6) with a fake graph and a fake editor."""

import pytest

from tradingagents.quality import editor, errata, loop

SHEET = {"facts": [{"key": "revenue.2026Q2", "value": 83.8e6, "unit": "usd"},
                   {"key": "price.close", "value": 7.19, "unit": "usd"}]}
WRONG_MARKET = "Revenue was $50.0M [F:revenue.2026Q2] last quarter."
RIGHT_MARKET = "Revenue was $83.8M [F:revenue.2026Q2] last quarter."


def state(market=RIGHT_MARKET, pm="Rating: Hold"):
    return {
        "fact_sheet": SHEET, "fact_sheet_text": "sheet", "company_of_interest": "ONDS",
        "market_report": market, "sentiment_report": "s", "news_report": "n", "fundamentals_report": "f",
        "investment_debate_state": {"bull_history": "b", "bear_history": "r", "history": "x", "count": 2},
        "investment_plan": "plan", "trader_investment_plan": "trade", "risk_debate_state": {"history": "h", "count": 3},
        "final_trade_decision": pm, "report_digest": {"headline": "Hold: fine", "bull_thesis": "ok"},
    }


class FakeGraph:
    def __init__(self, states):
        self.states, self.calls, self.logged, self.recorded = list(states), [], 0, 0

    def propagate(self, ticker, trade_date, revision=None, record=True, **_):
        self.calls.append({"revision": revision, "record": record})
        return self.states.pop(0), "Hold"

    def _log_state(self, *_):
        self.logged += 1

    def record_decision(self, *_):
        self.recorded += 1


def review_with(*reviews):
    reviews = list(reviews)

    def fake(_llm, _state, _lint, _errata, callbacks=None):
        r = reviews.pop(0)
        if isinstance(r, Exception):
            raise r
        return {"findings": [], "digest_patch": [], "decision_flags": [], "editor_note": "", "hold_reason": "", **r}

    return fake


@pytest.mark.unit
def test_a_clean_first_pass_publishes():
    graph = FakeGraph([state()])
    final, _ = loop.run_with_quality(graph, "ONDS", "2026-10-05", None, review_fn=review_with({}))
    assert final["quality"]["status"] == "clean"
    assert len(graph.calls) == 1 and graph.calls[0]["record"] is False and graph.recorded == 1


@pytest.mark.unit
def test_a_decision_flag_revises_from_the_research_stage_keeping_the_analysts():
    graph = FakeGraph([state(), state()])
    final, _ = loop.run_with_quality(
        graph, "ONDS", "2026-10-05", None,
        review_fn=review_with({"decision_flags": ["price target not derived"]}, {"editor_note": "Target re-derived."}))
    assert final["quality"]["status"] == "revised"
    revision = graph.calls[1]["revision"]
    assert set(revision["kept"]) == {"market", "social", "news", "fundamentals"}
    assert "price target not derived" in revision["review_errata"]
    assert final["quality"]["passes"][0]["restart"]["from"] == "research"
    assert final["quality"]["editor_note"] == "Target re-derived."


@pytest.mark.unit
def test_an_analyst_error_reruns_that_analyst_only():
    graph = FakeGraph([state(), state()])
    finding = {"severity": "load_bearing", "location": {"stage": "market_analyst", "field": "market_report",
                                                       "quote": WRONG_MARKET}, "problem": "revenue misread"}
    loop.run_with_quality(graph, "ONDS", "2026-10-05", None, review_fn=review_with({"findings": [finding]}, {}))
    kept = graph.calls[1]["revision"]["kept"]
    assert "market" not in kept and {"social", "news", "fundamentals"} <= set(kept)
    assert "research" not in kept  # everything after the analysts re-runs


@pytest.mark.unit
def test_still_wrong_after_two_revisions_is_held():
    graph = FakeGraph([state(), state(), state()])
    flagged = {"decision_flags": ["rating not supported"], "hold_reason": "Rating unsupported by the evidence."}
    final, _ = loop.run_with_quality(graph, "ONDS", "2026-10-05", None,
                                     review_fn=review_with(flagged, flagged, flagged))
    assert final["quality"]["status"] == "held"
    assert final["quality"]["hold_reason"] == "Rating unsupported by the evidence."
    assert len(graph.calls) == 3 and graph.recorded == 0 and graph.logged == 1
    assert final["quality"]["editor_note"] == ""


@pytest.mark.unit
def test_an_unavailable_editor_holds_the_run():
    graph = FakeGraph([state()])
    final, _ = loop.run_with_quality(graph, "ONDS", "2026-10-05", None,
                                     review_fn=review_with(editor.EditorUnavailable("down")))
    assert final["quality"]["status"] == "held" and final["quality"]["hold_reason"] == "editor_unavailable"


@pytest.mark.unit
def test_a_digest_patch_is_applied_and_relinted():
    graph = FakeGraph([state()])
    patch = [{"field": "headline", "action": "replace", "value": "Hold: revenue of $83.8M [F:revenue.2026Q2]"},
             {"field": "bull_thesis", "action": "replace", "value": "Revenue of $99.0M [F:revenue.2026Q2]"}]
    final, _ = loop.run_with_quality(graph, "ONDS", "2026-10-05", None, review_fn=review_with({"digest_patch": patch}))
    assert final["report_digest"]["headline"].startswith("Hold: revenue of $83.8M")
    assert final["report_digest"]["bull_thesis"] == ""  # a replacement that fails lint becomes a deletion


@pytest.mark.unit
def test_errata_merge_dedupes_and_render():
    a = errata.from_editor([{"severity": "load_bearing", "location": {"stage": "bear_researcher", "quote": "SG&A $179.8M"},
                             "problem": "total opex, not SG&A", "correction": "[F:ga.2026Q2]"}])
    b = errata.from_editor([{"severity": "minor", "location": {"stage": "research_manager", "quote": "SG&A $179.8M"},
                             "problem": "repeated"}])
    merged = errata.merge(a, b)
    assert len(merged) == 1 and merged[0]["id"] == "E1" and merged[0]["also_in"] == ["research_manager."]
    text = errata.render(merged)
    assert "[E1] load_bearing" in text and "SG&A $179.8M" in text


@pytest.mark.unit
def test_calc_is_arithmetic_only():
    assert editor.calc("(83.8 - 6.3) / 6.3 * 100") == "1230.16"
    assert editor.calc("__import__('os')").startswith("error")


class ToolLLM:
    """Answers with a fact lookup, a calculation, then the review."""

    def __init__(self):
        self.step, self.seen = 0, []

    def bind_tools(self, tools):
        assert {t["name"] for t in tools} == {"calc", "fact", "submit_review"}
        return self

    def invoke(self, messages, config=None):
        from langchain_core.messages import AIMessage

        self.seen.append(messages[-1])
        self.step += 1
        calls = {
            1: [{"name": "fact", "args": {"key": "revenue.2026Q2"}, "id": "t1"}],
            2: [{"name": "calc", "args": {"expression": "83.8/6.3"}, "id": "t2"}],
            3: [{"name": "submit_review", "args": {"findings": [], "digest_patch": [], "decision_flags": [],
                                                   "editor_note": "Checked.", "hold_reason": ""}, "id": "t3"}],
        }[self.step]
        return AIMessage(content="", tool_calls=calls)


@pytest.mark.unit
def test_editor_uses_its_tools_then_submits():
    llm = ToolLLM()
    out = editor.review(llm, state(), {"flags": []})
    assert out["editor_note"] == "Checked."
    assert '"value": 83800000.0' in llm.seen[1].content   # the fact tool's answer
    assert llm.seen[2].content == "13.3016"                # the calc tool's answer


@pytest.mark.unit
def test_editor_retries_then_gives_up():
    class Down:
        def bind_tools(self, _):
            return self

        def invoke(self, *_a, **_k):
            raise ConnectionError("overloaded")

    with pytest.raises(editor.EditorUnavailable):
        editor.review(Down(), state(), {"flags": []}, budget_seconds=12, sleep=lambda _s: None)


@pytest.mark.unit
def test_kept_stages_return_the_previous_output_without_running():
    from tradingagents.quality import gates

    def boom(_state):
        raise AssertionError("a kept stage ran")

    debate = {"history": "x", "count": 2}
    kept = {"research": {"investment_debate_state": debate, "investment_plan": "plan"}, "trader": {"trader_investment_plan": "t"}}
    st = {"kept": kept}
    assert gates.kept_or_run("research", boom, lambda k: {"investment_debate_state": k["investment_debate_state"]})(st) == {
        "investment_debate_state": debate}
    assert gates.kept_or_run("research", boom)(st)["investment_plan"] == "plan"
    assert gates.kept_or_run("risk", lambda s: {"ran": True})(st) == {"ran": True}

    class Sub:
        def invoke(self, s, config=None):
            return {"market_report": "fresh"}

    assert gates.kept_analyst("market", Sub())({"kept": {"market": {"market_report": "old"}}}) == {"market_report": "old"}
    assert gates.kept_analyst("market", Sub())({"kept": {}}) == {"market_report": "fresh"}


@pytest.mark.unit
def test_the_graph_compiles_with_the_loop():
    from unittest.mock import MagicMock

    from tradingagents.graph.conditional_logic import ConditionalLogic
    from tradingagents.graph.setup import GraphSetup

    graph = GraphSetup(MagicMock(), MagicMock(), ConditionalLogic(), 3, quality_gates=True,
                       quality_loop=True).setup_graph(("market", "news")).compile()
    assert "Market Analyst Check" in graph.get_graph().nodes


@pytest.mark.unit
def test_the_staging_hook_holds_only_its_ticker():
    graph = FakeGraph([state()])
    graph.config = {"quality_force_hold_ticker": "ONDS"}
    final, _ = loop.run_with_quality(graph, "onds", "2026-10-05", None, review_fn=review_with({}))
    assert final["quality"]["status"] == "held" and "forced hold" in final["quality"]["hold_reason"]
    other = FakeGraph([state()])
    other.config = {"quality_force_hold_ticker": "KO"}
    assert loop.run_with_quality(other, "ONDS", "2026-10-05", None, review_fn=review_with({}))[0]["quality"]["status"] == "clean"
