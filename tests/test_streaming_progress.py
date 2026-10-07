"""tickeragent.ai hooks on propagate(): on_progress streams each top-level step
and still returns the state invoke() would; callbacks reach the graph config so
tool executions are counted; extra sentiment blocks reach the initial state."""

from tradingagents.graph.propagation import Propagator
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.memory import TradingMemoryLog

_UPDATE_CHUNKS = [
    # Analysts run side by side as subgraphs; each reports once, with its report.
    {"Market Analyst": {"market_report": "MARKET"}},
    {"Memory Log": {"past_context": "", "memory_note": ""}},
    {"Bull Researcher": {"investment_debate_state": {"bull_history": "BULL"}}},
    {"__metadata__": {"ignored": True}},
    {"Portfolio Manager": [{"final_trade_decision": "Rating: Buy\n\nx"}, {"final_rating": "Buy"}]},
]


class _Graph:
    """Stands in for the compiled LangGraph: records how it was driven."""

    def __init__(self, invoke_result=None):
        self.stream_calls, self.invoke_calls = [], []
        self._invoke_result = invoke_result

    def stream(self, graph_input, **kwargs):
        self.stream_calls.append(kwargs)
        return iter(_UPDATE_CHUNKS)

    def invoke(self, graph_input, **kwargs):
        self.invoke_calls.append(kwargs)
        return self._invoke_result or dict(graph_input, final_trade_decision="Rating: Hold\n\nx",
                                           final_rating="Hold")


def _bare_graph(tmp_path, compiled):
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {"memory_log_path": None, "max_debate_rounds": 1, "max_risk_discuss_rounds": 1,
                    "results_dir": str(tmp_path), "data_cache_dir": str(tmp_path),
                    "report_digest": False}
    graph.memory_log = TradingMemoryLog(graph.config)
    graph.propagator = Propagator()
    graph.selected_analysts = ("market",)
    graph.debug = False
    graph._resuming = False
    graph._checkpointer_ctx = None
    graph.graph = compiled
    graph.resolve_instrument_context = lambda t, a="stock", d=None: ""
    graph._log_state = lambda *a, **k: None
    return graph


def test_on_progress_streams_steps_and_merges_final_state(tmp_path):
    compiled = _Graph()
    graph = _bare_graph(tmp_path, compiled)
    seen = []
    final_state, rating = graph.propagate(
        "NVDA", "2026-01-09", on_progress=lambda node, delta, state: seen.append((node, delta)),
    )
    # Every step fired the callback in order; LangGraph metadata entries did not.
    assert [node for node, _ in seen] == [
        "Market Analyst", "Memory Log", "Bull Researcher", "Portfolio Manager",
    ]
    # A node that wrote a channel twice arrives as a list; its deltas merge.
    assert seen[-1][1] == {"final_trade_decision": "Rating: Buy\n\nx", "final_rating": "Buy"}
    assert final_state["market_report"] == "MARKET"
    assert final_state["news_report"] == ""  # untouched keys keep their initial value
    assert rating == "Buy"
    assert compiled.stream_calls[0]["stream_mode"] == "updates"
    assert compiled.invoke_calls == []


def test_default_path_still_invokes(tmp_path):
    compiled = _Graph()
    graph = _bare_graph(tmp_path, compiled)
    final_state, rating = graph.propagate("NVDA", "2026-01-09")
    assert rating == "Hold"
    assert compiled.stream_calls == [] and len(compiled.invoke_calls) == 1


def test_callbacks_reach_the_graph_config(tmp_path):
    compiled = _Graph()
    graph = _bare_graph(tmp_path, compiled)
    handler = object()
    graph.propagate("NVDA", "2026-01-09", callbacks=[handler])
    assert compiled.invoke_calls[0]["config"]["callbacks"] == [handler]


def test_extra_sentiment_blocks_reach_the_initial_state(tmp_path):
    compiled = _Graph()
    graph = _bare_graph(tmp_path, compiled)
    final_state, _ = graph.propagate("NVDA", "2026-01-09", extra_sentiment_blocks=[("Feed", "text")])
    assert final_state["extra_sentiment_blocks"] == [("Feed", "text")]
