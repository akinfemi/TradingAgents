"""Waiting out per-minute rate limits (eval, 2026-10-10: runs failed on
OpenRouter's new-account cap of 20 requests a minute per model)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tradingagents.llm_clients import openai_client
from tradingagents.llm_clients.openai_client import RATE_LIMIT_MAX_WAIT, rate_limit_wait


def _err(body=None, retry_after=None):
    return SimpleNamespace(body=body, response=SimpleNamespace(headers={"retry-after": retry_after} if retry_after else {}))


@pytest.mark.unit
def test_wait_follows_openrouters_reset():
    body = {"message": "Rate limit exceeded", "metadata": {"headers": {"X-RateLimit-Reset": "1000030000"}}}
    wait = rate_limit_wait(_err(body), now=1_000_000.0)            # reset 30s away
    assert 30.5 <= wait <= 33.0
    assert 30.5 <= rate_limit_wait(_err({"error": body}), now=1_000_000.0) <= 33.0


@pytest.mark.unit
def test_wait_falls_back_and_is_clamped():
    assert 5.5 <= rate_limit_wait(_err(retry_after="5"), now=0) <= 8.0
    assert 60.5 <= rate_limit_wait(_err(), now=0) <= 63.0
    far = {"metadata": {"headers": {"X-RateLimit-Reset": str(10**15)}}}
    assert rate_limit_wait(_err(far), now=0) <= RATE_LIMIT_MAX_WAIT + 3.0


@pytest.mark.unit
def test_generate_retries_a_rate_limit_then_succeeds():
    import httpx
    import openai
    from langchain_core.messages import HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.messages import AIMessage

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    limited = openai.RateLimitError("429", response=httpx.Response(429, request=request), body={})
    ok = ChatResult(generations=[ChatGeneration(message=AIMessage(content="done"))])
    calls = {"n": 0}

    def fake(self, messages, stop=None, run_manager=None, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise limited
        return ok

    llm = openai_client.NormalizedChatOpenAI(model="gpt-6-luna", api_key="x", base_url="https://openrouter.ai/api/v1")
    with patch.object(openai_client.ChatOpenAI, "_generate", fake), patch.object(openai_client.time, "sleep") as sleep:
        result = llm._generate([HumanMessage(content="hi")])
    assert result is ok and calls["n"] == 3 and sleep.call_count == 2
