"""StatsCallbackHandler: cached prompt tokens are counted per model and in
total, apart from tokens_in, which stays the raw prompt count."""

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from cli.stats_handler import StatsCallbackHandler


def _response(model, tokens_in, tokens_out, details=None):
    usage = {"input_tokens": tokens_in, "output_tokens": tokens_out, "total_tokens": tokens_in + tokens_out}
    if details is not None:
        usage["input_token_details"] = details
    msg = AIMessage(content="x", usage_metadata=usage)
    return LLMResult(generations=[[ChatGeneration(message=msg)]], llm_output={"model_name": model})


def test_cache_reads_and_writes_are_counted_per_model_and_in_total():
    stats = StatsCallbackHandler()
    # langchain-openai / OpenRouter shape.
    stats.on_llm_end(_response("anthropic/claude-sonnet-5.5", 42_000, 2_000,
                               {"cache_read": 0, "cache_creation": 40_000}))
    stats.on_llm_end(_response("anthropic/claude-sonnet-5.5", 45_000, 2_000,
                               {"cache_read": 40_000}))
    # langchain-anthropic with a TTL breakdown: cache_creation zeroed.
    stats.on_llm_end(_response("claude-sonnet-5-5", 30_000, 1_000,
                               {"cache_read": 5_000, "cache_creation": 0,
                                "ephemeral_5m_input_tokens": 20_000, "ephemeral_1h_input_tokens": 0}))
    # No details at all (most quick-tier calls).
    stats.on_llm_end(_response("openai/gpt-6-luna", 10_000, 500))

    by_model = stats.usage_by_model
    assert by_model["anthropic/claude-sonnet-5.5"] == {
        "llm_calls": 2, "tokens_in": 87_000, "tokens_out": 4_000, "cache_read": 40_000, "cache_write": 40_000}
    assert by_model["claude-sonnet-5-5"]["cache_read"] == 5_000
    assert by_model["claude-sonnet-5-5"]["cache_write"] == 20_000
    assert by_model["openai/gpt-6-luna"] == {
        "llm_calls": 1, "tokens_in": 10_000, "tokens_out": 500, "cache_read": 0, "cache_write": 0}

    assert stats.tokens_in == 127_000  # raw, cached tokens included
    assert stats.cache_read == 45_000 and stats.cache_write == 60_000
    got = stats.get_stats()
    assert got["cache_read"] == 45_000 and got["cache_write"] == 60_000
    assert got["usage_by_model"]["claude-sonnet-5-5"]["cache_write"] == 20_000


def test_malformed_details_count_as_no_cache():
    stats = StatsCallbackHandler()
    response = _response("m", 100, 10)
    # Past LangChain's validation, as a provider adapter's raw dict could be.
    response.generations[0][0].message.usage_metadata["input_token_details"] = {"cache_read": "n/a"}
    stats.on_llm_end(response)
    assert stats.tokens_in == 100 and stats.cache_read == 0
    assert stats.usage_by_model["m"]["cache_read"] == 0
