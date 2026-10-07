import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult


class StatsCallbackHandler(BaseCallbackHandler):
    """Callback handler that tracks LLM calls, tool calls, and token usage."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.tool_calls_by_name: dict[str, int] = {}
        self.usage_by_model: dict[str, dict[str, int]] = {}

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when an LLM starts."""
        with self._lock:
            self.llm_calls += 1

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when a chat model starts."""
        with self._lock:
            self.llm_calls += 1

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Extract token usage from LLM response."""
        try:
            generation = response.generations[0][0]
        except (IndexError, TypeError):
            return

        usage_metadata = None
        if hasattr(generation, "message"):
            message = generation.message
            if isinstance(message, AIMessage) and hasattr(message, "usage_metadata"):
                usage_metadata = message.usage_metadata

        model_name = self._model_name(response, generation)

        with self._lock:
            per_model = self.usage_by_model.setdefault(
                model_name, {"llm_calls": 0, "tokens_in": 0, "tokens_out": 0}
            )
            per_model["llm_calls"] += 1
            if usage_metadata:
                tokens_in = usage_metadata.get("input_tokens", 0)
                tokens_out = usage_metadata.get("output_tokens", 0)
                self.tokens_in += tokens_in
                self.tokens_out += tokens_out
                per_model["tokens_in"] += tokens_in
                per_model["tokens_out"] += tokens_out

    @staticmethod
    def _model_name(response: LLMResult, generation: Any) -> str:
        """Best-effort model name from the response, across provider shapes.

        OpenAI puts it in ``llm_output["model_name"]`` and the message's
        ``response_metadata["model_name"]``; Anthropic uses
        ``response_metadata["model"]`` (newer langchain-anthropic also sets
        ``"model_name"``). Fall back to ``"unknown"`` when neither is present.
        """
        llm_output = response.llm_output or {}
        name = llm_output.get("model_name") or llm_output.get("model")
        if name:
            return name
        message = getattr(generation, "message", None)
        metadata = getattr(message, "response_metadata", None) or {}
        return metadata.get("model_name") or metadata.get("model") or "unknown"

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Increment tool call counter when a tool starts."""
        name = serialized.get("name") if isinstance(serialized, dict) else None
        with self._lock:
            self.tool_calls += 1
            if name:
                self.tool_calls_by_name[name] = self.tool_calls_by_name.get(name, 0) + 1

    def get_stats(self) -> dict[str, Any]:
        """Return current statistics."""
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "tool_calls_by_name": dict(self.tool_calls_by_name),
                "usage_by_model": {
                    model: dict(usage) for model, usage in self.usage_by_model.items()
                },
            }
