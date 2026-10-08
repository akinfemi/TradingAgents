"""Stage gates (REPORT_QUALITY_PLAN R5, Layer 2).

Each stage's output is linted (checks 1–5 and the data rule) before the next
stage reads it. On blocking flags the same model gets one fix-up turn on its
own text ("these figures don't match the fact sheet: …; correct them or
remove them"), and the corrected text replaces the original. A flag that
survives goes to ``state["open_errata"]``, which every later prompt shows,
so a wrong figure is not argued over as if it were true.

Gates are a no-op without a fact sheet. The research manager and portfolio
manager are linted but not rewritten here: their typed decisions would fall
out of step with an edited text; the editor (R6) patches those.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from tradingagents.quality.lint import Facts, LintFlag, lint_text

logger = logging.getLogger(__name__)

# A rewrite shorter than this share of the original lost content: keep the original.
MIN_REWRITE_RATIO = 0.6
MAX_FLAGS_IN_PROMPT = 12

FIXUP_PROMPT = """You wrote the text below as the {role}. A check against the run's fact sheet found these problems:

{problems}

Return the COMPLETE text again with each problem corrected: replace a wrong figure with the fact sheet's value (keep or add its [F:key] citation), fix a wrong period, concept or direction, remove a quote that its source does not contain, and remove data the platform may not use. Change nothing else: same structure, same argument, same length. Return only the corrected text.

{fact_sheet}

--- YOUR TEXT ---
{text}"""


def describe(flags: list[LintFlag]) -> str:
    lines = []
    for f in flags[:MAX_FLAGS_IN_PROMPT]:
        lines.append(f"- [{f.kind}] \"{f.quote[:240]}\"" + (f" (fact sheet: {f.expected})" if f.expected else ""))
    if len(flags) > MAX_FLAGS_IN_PROMPT:
        lines.append(f"- … and {len(flags) - MAX_FLAGS_IN_PROMPT} more of the same kinds")
    return "\n".join(lines)


def errata_entries(flags: list[LintFlag]) -> list[dict]:
    return [{"stage": f.stage, "kind": f.kind, "quote": f.quote[:240], "expected": f.expected} for f in flags]


def render_errata(errata: list[dict] | None) -> str:
    """The block later prompts show while earlier errors are still open."""
    if not errata:
        return ""
    lines = ["**Open errata** (statements in earlier stages that do not match the fact sheet; do not repeat "
             "or build on them, use the fact sheet):"]
    for e in errata[-20:]:
        lines.append(f"- {e['stage']}: \"{e['quote'][:200]}\"" + (f" (fact sheet: {e['expected']})" if e.get("expected") else ""))
    return "\n".join(lines)


def sources_of(state: dict, own: str) -> dict[str, str]:
    """The analyst reports a stage may quote (not its own)."""
    reports = {"market": "market_report", "sentiment": "sentiment_report", "news": "news_report",
               "fundamentals": "fundamentals_report"}
    return {k: state.get(v) or "" for k, v in reports.items() if not own.startswith(k)}


def check_and_fix(text: str, state: dict, stage: str, field_name: str, llm=None, role: str = "") -> tuple[str, list[dict], dict]:
    """(text to keep, errata to add, gate record). Never raises."""
    record = {"stage": stage, "field": field_name, "checked": False, "flags_before": 0, "flags_after": 0,
              "fixed": False}
    facts = Facts(state.get("fact_sheet"))
    if not facts or not text:
        return text, [], record
    try:
        sources = sources_of(state, stage)
        flags = [f for f in lint_text(text, facts, stage, field_name, sources) if f.blocking]
        record.update(checked=True, flags_before=len(flags), kinds=sorted({f.kind for f in flags}))
        if not flags:
            return text, [], record
        if llm is None:
            record["flags_after"] = len(flags)
            return text, errata_entries(flags), record
        prompt = FIXUP_PROMPT.format(role=role or stage.replace("_", " "), problems=describe(flags),
                                     fact_sheet=state.get("fact_sheet_text") or "", text=text)
        reply = llm.invoke(prompt)
        fixed = (reply.text if hasattr(reply, "text") else str(reply)).strip()
        if len(fixed) < MIN_REWRITE_RATIO * len(text):
            logger.warning("gate %s: fix-up rewrite too short (%d of %d chars); keeping the original",
                           stage, len(fixed), len(text))
            record["flags_after"] = len(flags)
            record["rewrite_rejected"] = True
            return text, errata_entries(flags), record
        remaining = [f for f in lint_text(fixed, facts, stage, field_name, sources) if f.blocking]
        record.update(fixed=True, flags_after=len(remaining))
        return fixed, errata_entries(remaining), record
    except Exception as exc:  # noqa: BLE001 — a gate never fails the run
        logger.warning("gate %s failed: %s", stage, exc, exc_info=True)
        record["error"] = type(exc).__name__
        return text, [], record


# ---- wrappers for the graph's nodes -------------------------------------------------------


def analyst_check(report_key: str, stage: str, llm, role: str) -> Callable:
    """A node after an analyst subgraph: checks and fixes its report."""

    def node(state: dict) -> dict:
        text = state.get(report_key) or ""
        kept, errata, record = check_and_fix(text, state, stage, report_key, llm, role)
        out: dict = {"quality_gates": [record]}
        if kept != text:
            out[report_key] = kept
        if errata:
            out["open_errata"] = errata
        return out

    return node


def _replace_last(haystack: str, old: str, new: str) -> str:
    i = haystack.rfind(old)
    return haystack if i < 0 else haystack[:i] + new + haystack[i + len(old):]


def debate_turn(node: Callable, stage: str, state_key: str, response_key: str, own_history_key: str,
                llm, role: str) -> Callable:
    """Wraps a debate node: checks the turn it just added and swaps in the fix
    everywhere the turn was written (its own history, the shared history and
    the current response)."""

    def wrapped(state: dict) -> dict:
        delta = node(state)
        debate = dict(delta.get(state_key) or {})
        turn = debate.get(response_key) or ""
        prefix, _, body = turn.partition(": ")
        view = {**state, **{k: v for k, v in delta.items() if k != state_key}}
        kept, errata, record = check_and_fix(body or turn, view, stage, own_history_key, llm, role)
        out = dict(delta)
        if kept != (body or turn):
            new_turn = f"{prefix}: {kept}" if body else kept
            for key in ("history", own_history_key, response_key):
                if isinstance(debate.get(key), str):
                    debate[key] = _replace_last(debate[key], turn, new_turn) if key != response_key else new_turn
            out[state_key] = debate
        out["quality_gates"] = [record]
        if errata:
            out["open_errata"] = errata
        return out

    return wrapped


def text_stage(node: Callable, stage: str, field_name: str, llm, role: str, fix: bool = True) -> Callable:
    """Wraps a node whose output is one text field (trader plan); with
    ``fix=False`` (research manager, portfolio manager) it only lints."""

    def wrapped(state: dict) -> dict:
        delta = node(state)
        text = delta.get(field_name) or ""
        view = {**state, **delta}
        kept, errata, record = check_and_fix(text, view, stage, field_name, llm if fix else None, role)
        out = dict(delta)
        if kept != text:
            out[field_name] = kept
            if out.get("messages"):  # the trader also posts its plan as a message
                from langchain_core.messages import AIMessage

                out["messages"] = [AIMessage(content=kept)]
        out["quality_gates"] = [record]
        if errata:
            out["open_errata"] = errata
        return out

    return wrapped
