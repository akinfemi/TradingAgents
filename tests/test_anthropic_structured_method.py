"""tickeragent.ai: models that reject a forced tool choice get native JSON-schema
output, with an unforced tool call as the fallback for schemas Anthropic finds
too complex; other models keep LangChain's default."""

import pytest

from tradingagents.agents.schemas import PortfolioDecision
from tradingagents.llm_clients.anthropic_client import (
    NormalizedChatAnthropic,
    rejects_forced_tool_choice,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "model,rejects",
    [
        ("claude-opus-5-5", True),
        ("claude-sonnet-5-5", True),
        ("claude-fable-5-1", True),
        ("claude-sonnet-5", False),
        ("claude-haiku-4-5", False),
        ("claude-opus-5", False),
        ("claude-sonnet-5-50", False),
    ],
)
def test_which_models_reject_a_forced_tool_choice(model, rejects):
    assert rejects_forced_tool_choice(model) is rejects


@pytest.mark.unit
def test_5_5_models_use_json_schema_with_an_unforced_tool_fallback(monkeypatch):
    llm = NormalizedChatAnthropic(model="claude-sonnet-5-5", api_key="placeholder")
    runnable = llm.with_structured_output(PortfolioDecision)
    # A RunnableWithFallbacks: native JSON schema first, then the auto tool call.
    assert type(runnable).__name__ == "RunnableWithFallbacks"
    assert len(runnable.fallbacks) == 1


@pytest.mark.unit
def test_other_models_keep_function_calling():
    llm = NormalizedChatAnthropic(model="claude-sonnet-5", api_key="placeholder")
    runnable = llm.with_structured_output(PortfolioDecision)
    assert type(runnable).__name__ != "RunnableWithFallbacks"
