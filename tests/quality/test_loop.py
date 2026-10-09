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

    def fake(_llm, _state, _lint, _errata, callbacks=None, on_step=None):
        r = reviews.pop(0)
        if on_step and not isinstance(r, Exception):
            on_step({"kind": "fact", "detail": "revenue.2026Q2"})
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
    # A target is a levels issue: the trader and PM re-run, the debate is kept.
    assert set(revision["kept"]) == {"market", "social", "news", "fundamentals", "research"}
    assert "price target not derived" in revision["review_errata"]
    assert final["quality"]["passes"][0]["restart"]["from"] == "trader"
    assert final["quality"]["editor_note"] == "Target re-derived."


@pytest.mark.unit
def test_an_analyst_only_finding_does_not_force_a_revision():
    """R7 calibration: problems a reader never sees (an analyst's report the
    ruling and decision don't rely on) are minor, not a revision."""
    graph = FakeGraph([state()])
    finding = {"severity": "load_bearing", "location": {"stage": "market_analyst", "field": "market_report",
                                                       "quote": WRONG_MARKET}, "problem": "revenue misread"}
    final, _ = loop.run_with_quality(graph, "ONDS", "2026-10-05", None, review_fn=review_with({"findings": [finding]}))
    assert final["quality"]["status"] == "clean" and len(graph.calls) == 1


@pytest.mark.unit
def test_an_erratum_sourced_in_one_analyst_reruns_only_that_analyst():
    errs = [{"severity": "load_bearing", "source": "market_analyst", "also_in": []}]
    rerun, first_later = errata.restart_group(errs)
    assert rerun == {"market"} and first_later == "research"
    kept = errata.kept_outputs(state(), rerun, first_later, loop.ANALYST_REPORT_KEYS)
    assert "market" not in kept and {"social", "news", "fundamentals"} <= set(kept) and "research" not in kept


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
    original = state()["report_digest"]["bull_thesis"]
    final, _ = loop.run_with_quality(graph, "ONDS", "2026-10-05", None, review_fn=review_with({"digest_patch": patch}))
    assert final["report_digest"]["headline"].startswith("Hold: revenue of $83.8M")
    # A text replacement that fails lint is not applied: the original stays for the relint.
    assert final["report_digest"]["bull_thesis"] == original


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


@pytest.mark.unit
def test_a_billing_error_is_not_retried():
    class Broke:
        calls = 0

        def bind_tools(self, _):
            return self

        def invoke(self, *_a, **_k):
            Broke.calls += 1
            raise RuntimeError("Error code: 400 - Your credit balance is too low to access the Anthropic API.")

    with pytest.raises(RuntimeError, match="credit balance"):
        editor.review(Broke(), state(), {"flags": []}, budget_seconds=600, sleep=lambda _s: None)
    assert Broke.calls == 1


@pytest.mark.unit
def test_a_levels_flag_reruns_from_the_trader_keeping_the_debate():
    graph = FakeGraph([state(), state()])
    loop.run_with_quality(graph, "ONDS", "2026-10-05", None,
                          review_fn=review_with({"decision_flags": ["stop sits inside the entry zone"]}, {}))
    kept = graph.calls[1]["revision"]["kept"]
    assert "research" in kept and "trader" not in kept


@pytest.mark.unit
def test_a_rating_flag_reruns_the_debate():
    graph = FakeGraph([state(), state()])
    loop.run_with_quality(graph, "ONDS", "2026-10-05", None,
                          review_fn=review_with({"decision_flags": ["rating not supported by the evidence"]}, {}))
    assert "research" not in graph.calls[1]["revision"]["kept"]



@pytest.mark.unit
def test_a_kept_sentiment_analyst_keeps_its_structured_block():
    st = {**state(), "sentiment_structured": {"overall_score": 6, "coverage": "30 StockTwits messages"}}
    kept = errata.kept_outputs(st, set(), "research", loop.ANALYST_REPORT_KEYS)
    assert kept["social"]["sentiment_structured"]["coverage"] == "30 StockTwits messages"


UNSUPPORTED_PM = "Rating: Hold. Backlog of $250.0M supports the call."


@pytest.mark.unit
def test_a_lint_flag_the_editor_dismissed_does_not_hold_the_run():
    """Staging ONDS, 2026-10-09: held on lint flags the editor had checked
    and dismissed."""
    lb = [f for f in loop.lint_state(state(pm=UNSUPPORTED_PM))["flags"] if f["severity"] == "load_bearing"]
    assert lb, "the fixture needs a load-bearing lint flag"
    dismissed = [{k: f[k] for k in ("kind", "stage", "field", "quote")} | {"reason": "derived"} for f in lb]
    graph = FakeGraph([state(pm=UNSUPPORTED_PM)])
    final, _ = loop.run_with_quality(graph, "ONDS", "2026-10-05", None,
                                     review_fn=review_with({"lint_dismissed": dismissed}))
    assert final["quality"]["status"] == "clean" and len(graph.calls) == 1
    # Not dismissed, the same flag forces a revision.
    graph = FakeGraph([state(pm=UNSUPPORTED_PM), state()])
    final, _ = loop.run_with_quality(graph, "ONDS", "2026-10-05", None, review_fn=review_with({}, {}))
    assert final["quality"]["status"] == "revised"


@pytest.mark.unit
def test_dismissals_resolve_by_lint_id():
    flags = [{"kind": "unsupported", "stage": "rm", "field": "rm", "quote": "a", "severity": "load_bearing"},
             {"kind": "cited_mismatch", "stage": "pm", "field": "pm", "quote": "b", "severity": "minor"}]
    out = editor._resolve_dismissals([{"id": "L2", "reason": "rounding"}, {"id": "L9", "reason": "x"}, {"id": "1"}], flags)
    assert [d["quote"] for d in out] == ["b", "a"] and out[0]["reason"] == "rounding"
    prompt = editor.build_prompt({"fact_sheet_text": "s"}, {"flags": flags})
    assert "- L1 [load_bearing · unsupported]" in prompt and "lint_dismissed" in prompt
    assert "do not mention the errata" in errata.render([{"id": "E1", "severity": "minor", "kind": "k", "quote": "q"}])


@pytest.mark.unit
def test_the_editor_can_run_on_another_provider(monkeypatch):
    seen = {}

    class Client:
        def get_llm(self):
            return "llm"

    def fake_create(provider, model, base_url=None, **kwargs):
        seen.update(provider=provider, model=model, base_url=base_url, **kwargs)
        return Client()

    import tradingagents.llm_clients.factory as factory
    monkeypatch.setattr(factory, "create_llm_client", fake_create)
    config = {"llm_provider": "anthropic", "deep_think_llm": "claude-sonnet-5", "backend_url": "https://proxy",
              "editor_llm": "gpt-6.1-sol", "editor_provider": "openai", "editor_effort": "medium"}
    editor.create_editor_llm(config)
    assert seen["provider"] == "openai" and seen["model"] == "gpt-6.1-sol"
    assert seen["base_url"] is None and seen["reasoning_effort"] == "medium" and "effort" not in seen
    editor.create_editor_llm({**config, "editor_provider": None, "editor_llm": "claude-sonnet-5-5"})
    assert seen["provider"] == "anthropic" and seen["base_url"] == "https://proxy" and seen["effort"] == "medium"


@pytest.mark.unit
def test_a_review_that_breaks_the_schema_is_retried():
    replies = [{"findings": ["a string, not a finding"], "digest_patch": [], "decision_flags": []},
               {"findings": '[{"severity": "minor", "location": {"stage": "pm", "quote": "q"}, "problem": "p"}]',
                "digest_patch": [], "decision_flags": [], "editor_note": "", "hold_reason": ""}]
    import tradingagents.quality.editor as ed
    orig = ed._run_once
    ed._run_once = lambda *a, **k: replies.pop(0)
    try:
        out = ed.review(None, {"fact_sheet": SHEET}, {"flags": []}, sleep=lambda _s: None)
    finally:
        ed._run_once = orig
    assert out["findings"][0]["problem"] == "p" and not replies


@pytest.mark.unit
def test_a_digest_patch_with_review_vocabulary_is_rejected():
    digest = {"headline": "Underweight: quality verified, but spending is outrunning cash conversion",
              "bull_points": [{"title": "Cash", "detail": "FCF covers capex."}]}
    patch = [{"field": "headline", "action": "replace",
              "value": "Underweight: verified quality, but spending outruns cash conversion"},
             {"field": "bull_points[0].detail", "action": "replace", "value": "FCF covers capex (per errata E2)."}]
    out, applied = editor.apply_patch(digest, patch, editor.Facts(SHEET))
    # The headline keeps its text (the relint still flags it); the point is deleted.
    assert out["headline"] == digest["headline"] and out["bull_points"] == []
    assert applied == [{"field": "bull_points[0].detail", "action": "delete"}]
    fixed, _ = editor.apply_patch(digest, [{"field": "headline", "action": "replace",
                                             "value": "Underweight: spending is outrunning cash conversion"}],
                                  editor.Facts(SHEET))
    assert fixed["headline"] == "Underweight: spending is outrunning cash conversion"


@pytest.mark.unit
def test_effort_reaches_models_behind_openrouter(monkeypatch):
    """All testing runs through OpenRouter (2026-10-09): the editor's medium
    effort must reach a Claude or GPT model there as OpenRouter's reasoning."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    gpt = editor.create_editor_llm({"llm_provider": "anthropic", "deep_think_llm": "x", "editor_provider": "openrouter",
                                    "editor_llm": "openai/gpt-6.1-sol", "editor_effort": "medium"})
    assert gpt.extra_body == {"reasoning": {"effort": "medium"}}
    # Claude behind OpenRouter: no reasoning (it would switch on thinking and
    # break the tool loop), but the prompt is cached.
    claude = editor.create_editor_llm({"llm_provider": "anthropic", "deep_think_llm": "x", "editor_provider": "openrouter",
                                       "editor_llm": "anthropic/claude-sonnet-5.5", "editor_effort": "medium"})
    assert not (claude.extra_body or {}).get("reasoning")
    seen = {}

    class Bound:
        def invoke(self, messages, config=None):
            seen["block"] = messages[0].content[0]
            from langchain_core.messages import AIMessage
            return AIMessage(content="", tool_calls=[{"name": "submit_review", "args": {"findings": []}, "id": "1"}])

    class Fake:
        model_name = "anthropic/claude-sonnet-5.5"

        def bind_tools(self, _tools):
            return Bound()

    editor._run_once(Fake(), "prompt", editor.Facts({}))
    assert seen["block"]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.unit
def test_the_review_reports_its_progress():
    """The run page showed nothing for the review's minutes ("it just looks
    stuck", 2026-10-08): the loop reports checking, each tool call, the
    verdict and what a revision re-runs."""
    events = []
    graph = FakeGraph([state(), state()])
    loop.run_with_quality(graph, "ONDS", "2026-10-05", None,
                          on_progress=lambda node, delta, _s: events.append((node, (delta or {}).get("_review"))),
                          review_fn=review_with({"decision_flags": ["price target not derived"]}, {}))
    steps = [r["step"] for node, r in events if node == "Quality Review" and r]
    assert steps == ["checking", "tool", "checked", "revising", "checking", "tool", "checked", "passed"]
    assert next(r for _n, r in events if r and r["step"] == "checked")["facts"] == 1
    revising = next(r for _n, r in events if r and r["step"] == "revising")
    assert revising["restart_from"] == "trader" and revising["corrections"] == 1
    assert next(r for _n, r in events if r and r["step"] == "passed")["status"] == "revised"
