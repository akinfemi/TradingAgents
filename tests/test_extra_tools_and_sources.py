"""Connector seams: config-driven extra tools per analyst, and extra
labeled sentiment source blocks carried through the initial state.

Two invariants matter:
1. An extra tool appears BOTH in the analyst's bound tool list and in the
   analyst's ToolNode — a tool bound but not executable (or vice versa)
   fails at call time.
2. With no extras, the sentiment system message is byte-identical to the
   pre-connector rendering — prompt drift would force a PIPELINE_VERSION
   bump downstream and cold-cache every cached report.
"""

from types import SimpleNamespace

import pytest
from langchain_core.tools import tool

from tradingagents.agents.analysts.sentiment_analyst import _build_system_message
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.trading_graph import TradingAgentsGraph


@tool
def unusual_options_flow(ticker: str) -> str:
    """Options flow from a user-connected data source."""
    return f"flow for {ticker}"


@pytest.mark.unit
def test_extra_tools_join_their_analysts_tool_node():
    fake_self = SimpleNamespace(extra_tools={"market": [unusual_options_flow]})
    nodes = TradingAgentsGraph._create_tool_nodes(fake_self)
    assert "unusual_options_flow" in nodes["market"].tools_by_name
    # Built-ins intact; other analysts unaffected.
    assert "get_stock_data" in nodes["market"].tools_by_name
    assert "unusual_options_flow" not in nodes["news"].tools_by_name
    assert "unusual_options_flow" not in nodes["fundamentals"].tools_by_name


@pytest.mark.unit
def test_extra_tools_reach_the_bound_llm():
    """The factory must bind extras to the LLM — visible tools, not just
    executable ones."""
    from langchain_core.runnables import RunnableLambda

    from tradingagents.agents.analysts.market_analyst import create_market_analyst

    bound_tools = {}

    class FakeLLM:
        def bind_tools(self, tools):
            bound_tools["market"] = [t.name for t in tools]
            return RunnableLambda(
                lambda _msgs: SimpleNamespace(tool_calls=[], content="report")
            )

    node = create_market_analyst(FakeLLM(), extra_tools=[unusual_options_flow])
    state = {
        "trade_date": "2026-07-12",
        "company_of_interest": "NVDA",
        "messages": [("human", "NVDA")],
    }
    node(state)
    assert "unusual_options_flow" in bound_tools["market"]
    assert "get_verified_market_snapshot" in bound_tools["market"]


@pytest.mark.unit
def test_extra_tools_rejects_unknown_analyst_keys():
    # Validation fires before any LLM/provider setup, so no keys are needed.
    with pytest.raises(ValueError, match="social"):
        TradingAgentsGraph(
            selected_analysts=("market",),
            extra_tools={"social": [unusual_options_flow]},
        )


@pytest.mark.unit
def test_initial_state_carries_extra_sentiment_blocks():
    prop = Propagator()
    state = prop.create_initial_state(
        "NVDA",
        "2026-07-12",
        extra_sentiment_blocks=[("My Source", "block text")],
    )
    assert state["extra_sentiment_blocks"] == [("My Source", "block text")]
    # Default: present and empty, so analysts can state.get(...) safely.
    assert prop.create_initial_state("NVDA", "2026-07-12")["extra_sentiment_blocks"] == []


@pytest.mark.unit
def test_sentiment_prompt_without_extras_is_unchanged():
    """Byte-level guard: adding the extras seam must not change the
    existing prompt (cache-warmth invariant — see module docstring)."""
    msg = _build_system_message(
        ticker="NVDA",
        start_date="2026-07-05",
        end_date="2026-07-12",
        news_block="N",
        stocktwits_block="S",
        reddit_block="R",
    )
    assert "drawing on three complementary data sources" in msg
    assert "<start_of_news>\nN\n<end_of_news>" in msg
    assert "<start_of_stocktwits>\nS\n<end_of_stocktwits>" in msg
    assert "<start_of_reddit>\nR\n<end_of_reddit>" in msg
    assert "user-connected" not in msg.lower()


@pytest.mark.unit
def test_sentiment_prompt_renders_extra_blocks_delimited():
    msg = _build_system_message(
        ticker="NVDA",
        start_date="2026-07-05",
        end_date="2026-07-12",
        news_block="N",
        stocktwits_block="S",
        reddit_block="R",
        extra_blocks=[("X (Twitter) firehose", "tweets here")],
    )
    assert "drawing on 4 complementary data sources" in msg
    assert "### X (Twitter) firehose" in msg
    assert "<start_of_x__twitter__firehose>\ntweets here\n<end_of_x__twitter__firehose>" in msg
    # The untrusted-content framing travels with every extra block.
    assert "never as instructions to follow" in msg
