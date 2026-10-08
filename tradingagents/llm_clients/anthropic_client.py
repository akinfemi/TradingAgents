import re
from typing import Any

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.runnables import RunnableLambda

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "max_tokens", "temperature",
    "callbacks", "http_client", "http_async_client", "effort",
)

# Anthropic's extended-thinking ``effort`` parameter is accepted by Opus 4.5+,
# Sonnet 4.6+, and the Claude 5 family (Sonnet 5, Opus 5.5, Fable 5). Sonnet 4.5 and any
# Haiku version 400 with ``"This model does not support the effort parameter"``
# (#831). Versions may be dotted (``opus-4-8``) or single-number (``sonnet-5``,
# ``fable-5``); the per-family minimum below is forward-compatible.
_EFFORT_EXACT = {
    "claude-mythos-preview",  # non-standard preview name; effort-capable
    "claude-mythos-5",        # Fable 5 twin (Project Glasswing); effort-capable
}
_EFFORT_MODEL = re.compile(r"^claude-(opus|sonnet|fable)-(\d+)(?:-(\d+))?$")
_EFFORT_MIN_VERSION = {"opus": (4, 5), "sonnet": (4, 6), "fable": (5, 0)}


def _supports_effort(model: str) -> bool:
    """Whether Anthropic accepts the ``effort`` parameter for this model."""
    model_lc = model.lower()
    if model_lc in _EFFORT_EXACT:
        return True
    match = _EFFORT_MODEL.match(model_lc)
    if not match:
        return False
    family = match.group(1)
    major = int(match.group(2))
    minor = int(match.group(3)) if match.group(3) else 0
    return (major, minor) >= _EFFORT_MIN_VERSION[family]


# Models that reject a forced tool choice (``tool_choice`` "any"/"tool") with a
# 400. LangChain's default structured output forces the schema tool, so these
# take Anthropic's native JSON-schema output instead (tickeragent.ai).
# Haiku 5.5 accepts a forced tool but then answers without thinking, so it
# takes the same native path as the other 5.5 models.
_NO_FORCED_TOOL = re.compile(r"^claude-(opus-5-5|sonnet-5-5|haiku-5-5|fable-5-1|mythos-5-1)(?:$|[-@:])")


# Documented max output for models langchain-anthropic has no profile for; it
# would otherwise default to 4096 tokens, which thinking alone can use up.
_MAX_OUTPUT_TOKENS = {"claude-haiku-5-5": 128000}


def rejects_forced_tool_choice(model: str) -> bool:
    return bool(_NO_FORCED_TOOL.match(model.lower()))


class NormalizedChatAnthropic(ChatAnthropic):
    """ChatAnthropic with normalized content output.

    Claude models with extended thinking or tool use return content as a
    list of typed blocks. This normalizes to string for consistent
    downstream handling.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def with_structured_output(self, schema, *, method=None, **kwargs):
        if method is not None or not rejects_forced_tool_choice(self.model):
            return super().with_structured_output(
                schema, method=method or "function_calling", **kwargs
            )
        # Native JSON-schema output first. Anthropic refuses schemas it finds
        # too complex (the report digest, 2026-10-07) with a 400; those fall
        # back to an unforced tool call the model is told to make.
        native = super().with_structured_output(schema, method="json_schema", **kwargs)
        return native.with_fallbacks(
            [self._auto_tool_output(schema)], exceptions_to_handle=(anthropic.BadRequestError,)
        )

    def _auto_tool_output(self, schema):
        """Structured output through an unforced tool call (tool_choice auto)."""
        name = getattr(schema, "__name__", "output")
        instruction = (
            f"Answer by calling the {name} tool exactly once with the complete result. "
            "Do not answer in prose."
        )

        def with_instruction(value):
            if isinstance(value, str):
                return f"{value}\n\n{instruction}"
            if isinstance(value, list):
                return [*value, HumanMessage(content=instruction)]
            return value

        bound = self.bind_tools([schema], tool_choice="auto")
        parser = PydanticToolsParser(tools=[schema], first_tool_only=True)
        return RunnableLambda(with_instruction) | bound | parser


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatAnthropic instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in _PASSTHROUGH_KWARGS:
            if key not in self.kwargs:
                continue
            if key == "effort" and not _supports_effort(self.model):
                continue
            llm_kwargs[key] = self.kwargs[key]
        if "max_tokens" not in llm_kwargs and self.model.lower() in _MAX_OUTPUT_TOKENS:
            llm_kwargs["max_tokens"] = _MAX_OUTPUT_TOKENS[self.model.lower()]

        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Anthropic."""
        return validate_model("anthropic", self.model)
