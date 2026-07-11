"""propagate(on_progress=...) streams node-keyed updates and still returns the
same final state invoke() would; callbacks are forwarded to the graph config
so tool executions are visible to stats handlers (mirrors the CLI path)."""

import functools
from unittest.mock import MagicMock

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.graph.trading_graph import TradingAgentsGraph


def _initial_state():
    return {
        "company_of_interest": "NVDA",
        "trade_date": "2026-01-10",
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "investment_debate_state": {
            "bull_history": "", "bear_history": "", "history": "",
            "current_response": "", "judge_decision": "",
        },
        "investment_plan": "",
        "trader_investment_plan": "",
        "risk_debate_state": {
            "aggressive_history": "", "conservative_history": "",
            "neutral_history": "", "history": "", "judge_decision": "",
            "current_aggressive_response": "", "current_conservative_response": "",
            "current_neutral_response": "", "count": 1, "latest_speaker": "",
        },
        "final_trade_decision": "",
        "messages": [],
    }


_UPDATE_CHUNKS = [
    {"Market Analyst": {"market_report": "MARKET", "messages": ["m1"]}},
    {"Msg Clear Market": {"messages": []}},
    {"Bull Researcher": {"investment_debate_state": {"bull_history": "BULL"}}},
    {"Portfolio Manager": {"final_trade_decision": "**Rating**: Buy"}},
]


def _mock_graph(tmp_path):
    mock_graph = MagicMock()
    mock_graph.memory_log = TradingMemoryLog({"memory_log_path": str(tmp_path / "mem.md")})
    mock_graph.log_states_dict = {}
    mock_graph.debug = False
    mock_graph.config = {"results_dir": str(tmp_path)}
    mock_graph.propagator.create_initial_state.return_value = _initial_state()
    mock_graph.propagator.get_graph_args.return_value = {
        "stream_mode": "values",
        "config": {"recursion_limit": 100},
    }
    mock_graph.graph.stream.return_value = iter(_UPDATE_CHUNKS)
    # No checkpointer in play: propagate's finally must not swap self.graph.
    mock_graph._checkpointer_ctx = None
    # Bind the real methods under test.
    mock_graph._run_graph = functools.partial(TradingAgentsGraph._run_graph, mock_graph)
    return mock_graph


def test_on_progress_streams_updates_and_merges_final_state(tmp_path):
    mock_graph = _mock_graph(tmp_path)
    seen = []
    final_state, _ = TradingAgentsGraph.propagate(
        mock_graph, "NVDA", "2026-01-10",
        on_progress=lambda node, delta, state: seen.append((node, delta)),
    )

    # Every node fired the callback, in order, including no-op nodes.
    assert [node for node, _ in seen] == [
        "Market Analyst", "Msg Clear Market", "Bull Researcher", "Portfolio Manager",
    ]
    # Deltas merged over the initial state — untouched keys survive,
    # updated keys reflect the last delta.
    assert final_state["market_report"] == "MARKET"
    assert final_state["investment_debate_state"] == {"bull_history": "BULL"}
    assert final_state["final_trade_decision"] == "**Rating**: Buy"
    assert final_state["news_report"] == ""
    # The graph streamed in updates mode.
    _, kwargs = mock_graph.graph.stream.call_args
    assert kwargs["stream_mode"] == "updates"
    # invoke() was never used.
    mock_graph.graph.invoke.assert_not_called()


def test_default_path_still_invokes(tmp_path):
    mock_graph = _mock_graph(tmp_path)
    done_state = dict(_initial_state(), final_trade_decision="**Rating**: Hold")
    mock_graph.graph.invoke.return_value = done_state

    final_state, _ = TradingAgentsGraph.propagate(mock_graph, "NVDA", "2026-01-10")

    assert final_state["final_trade_decision"] == "**Rating**: Hold"
    mock_graph.graph.stream.assert_not_called()


def test_callbacks_forwarded_to_graph_args(tmp_path):
    mock_graph = _mock_graph(tmp_path)
    mock_graph.graph.invoke.return_value = _initial_state()
    handler = object()

    TradingAgentsGraph.propagate(mock_graph, "NVDA", "2026-01-10", callbacks=[handler])

    mock_graph.propagator.get_graph_args.assert_called_once_with(callbacks=[handler])
