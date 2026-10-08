"""The editor (REPORT_QUALITY_PLAN R6, Layer 3).

One reviewer model reads the whole record (the fact sheet, the digest, the
lint report and every stage's text) the way a broker's supervisory analyst
reads a draft, with two tools: ``calc`` for arithmetic and ``fact`` for the
fact sheet. It may not state a number it didn't get from one of them. It
returns findings, field-level digest patches, decision flags it may not fix
itself (the rating, the target), a reader-facing note and a hold reason.

The editor never rewrites the call: a decision flag triggers a revision,
which re-runs the deciding stages.
"""

from __future__ import annotations

import ast
import json
import logging
import operator
import re
import time

from tradingagents.quality.lint import Facts, lint_text, stage_texts

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 16
RETRY_BUDGET_SECONDS = 600  # plan: retry with backoff for up to 10 minutes, then hold


class EditorUnavailable(RuntimeError):
    """The editor could not complete a review within the retry budget."""


# Failures that retrying can't fix: billing, authentication, a malformed
# request. They propagate at once so the run fails with its real cause
# (no_credit / bad_key, refunded) instead of waiting out the retry budget
# (staging, 2026-10-08: "credit balance is too low" retried for 10 minutes).
_FATAL_MARKERS = ("credit balance", "insufficient_quota", "billing", "invalid x-api-key", "invalid api key")
_FATAL_STATUS = {400, 401, 402, 403, 404}


def is_fatal(exc: Exception) -> bool:
    message = str(exc).lower()
    if any(m in message for m in _FATAL_MARKERS):
        return True
    if type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError", "BadRequestError", "NotFoundError"):
        return True
    return getattr(exc, "status_code", None) in _FATAL_STATUS


# ---- tools ----------------------------------------------------------------------------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def calc(expression: str) -> str:
    """Arithmetic only: numbers, + − × ÷, powers and parentheses."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only numbers and + - * / ** ( ) are allowed")

    try:
        cleaned = expression.replace(",", "").replace("×", "*").replace("÷", "/").replace("−", "-")
        value = ev(ast.parse(cleaned, mode="eval").body)
        return f"{value:.6g}"
    except Exception as exc:  # noqa: BLE001 — the model sees the error and retries
        return f"error: {exc}"


def fact(facts: Facts, key: str) -> str:
    f = facts.get(key.strip().strip("[]").removeprefix("F:"))
    if f is None:
        near = [k for k in facts.by_key if key.split(".")[0] in k][:12]
        return f"no fact {key}" + (f"; keys starting the same: {', '.join(near)}" if near else "")
    return json.dumps({k: f.get(k) for k in ("key", "value", "unit", "period", "derivation", "source")})


TOOLS = [
    {"name": "calc", "description": "Evaluate an arithmetic expression (numbers, + - * / ** and parentheses).",
     "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}},
                      "required": ["expression"]}},
    {"name": "fact", "description": "Look up one fact-sheet key, e.g. revenue.2026Q2, and get its value, unit and period.",
     "input_schema": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}},
    {"name": "submit_review", "description": "Submit the finished review. Call exactly once, last.",
     "input_schema": {
         "type": "object",
         "properties": {
             "findings": {"type": "array", "items": {"type": "object", "properties": {
                 "severity": {"type": "string", "enum": ["load_bearing", "minor"]},
                 "kind": {"type": "string"},
                 "location": {"type": "object", "properties": {
                     "stage": {"type": "string"}, "field": {"type": "string"}, "quote": {"type": "string"}},
                     "required": ["stage", "quote"]},
                 "problem": {"type": "string"},
                 "correction": {"type": "string"}},
                 "required": ["severity", "location", "problem"]}},
             "digest_patch": {"type": "array", "items": {"type": "object", "properties": {
                 "field": {"type": "string", "description": "digest field, e.g. headline, bull_thesis, bear_points[1].detail"},
                 "action": {"type": "string", "enum": ["replace", "delete"]},
                 "value": {"type": "string"}},
                 "required": ["field", "action"]}},
             "decision_flags": {"type": "array", "items": {"type": "string"}},
             "editor_note": {"type": "string"},
             "hold_reason": {"type": "string"},
         },
         "required": ["findings", "digest_patch", "decision_flags", "editor_note", "hold_reason"],
     }},
]

BRIEF = """You are the supervisory analyst reviewing an AI-written equity research report before publication. Review it the way a broker's compliance and research supervisor would.

You have the run's fact sheet (figures computed in code from the company's SEC filings and its prices), the report's digest (what readers see first), the automated lint report, and the full transcript of every stage. Use the tools: `fact` to read any fact-sheet key, `calc` for any arithmetic. You may not state a number in a finding, correction, patch or note unless you got it from `fact` or `calc`, or quote it from the text under review.

Check every load-bearing figure and attribution: the digest headline, bull and bear theses and points, the ruling, exit triggers, the PM's summary and thesis, and the price target and its math. Also answer:
- Is the price target derived (shown math from cited figures), or asserted?
- Does the 3-month rating horizon fit the thesis and its catalysts?
- Is the stop placed sensibly against the ATR?
- Does any claimed consensus or "all analysts agree" actually hold in the transcript?
- Are the catalysts real and dated (the fact sheet's earnings calendar is authoritative; already-reported quarters are past)?
- Is the company described correctly, and does the sector framing match its filings?
- Does any text use data the platform may not use (short interest, consensus estimates, analyst price targets)?

Then submit the review with `submit_review`:
- findings: every problem, with severity load_bearing when it sits in a load-bearing place or changes the argument, else minor; location {stage, field, quote} quoting the text exactly; the problem; the correction (with fact keys).
- digest_patch: field-level replacements or deletions that fix load-bearing digest text. Replace only with text whose every figure you checked; delete when the claim cannot be fixed. Never touch the rating.
- decision_flags: problems you may not fix yourself and that need the deciding stages re-run, e.g. "rating not supported by the evidence", "price target not derived", "thesis rests on a misread figure".
- editor_note: at most three short lines for the reader about what the review changed, empty if nothing.
- hold_reason: one line explaining why the report should not be published if it could not be fixed, else empty.

Lint findings are leads, not verdicts: confirm or dismiss each one with the tools."""


def _transcript(state: dict) -> str:
    parts = []
    for stage, field, text in stage_texts(state):
        if stage == "digest":
            continue
        parts.append(f"## {stage} ({field})\n{text}")
    return "\n\n".join(parts)


def build_prompt(state: dict, lint_report: dict, earlier_errata: str = "") -> str:
    flags = [f for f in lint_report.get("flags") or [] if f.get("blocking") or f.get("severity") == "load_bearing"]
    lint_lines = "\n".join(
        f"- [{f['severity']} · {f['kind']}] {f['stage']}/{f['field']}: \"{f['quote'][:220]}\" — {f.get('expected') or ''}"
        for f in flags[:80]) or "(no blocking flags)"
    return (
        f"{BRIEF}\n\n"
        + (f"# Errata from the previous draft (check each fix landed)\n{earlier_errata}\n\n" if earlier_errata else "")
        + f"# Fact sheet\n{state.get('fact_sheet_text') or '(none)'}\n\n"
        f"# Digest (JSON)\n{json.dumps(state.get('report_digest') or {}, ensure_ascii=False)[:30000]}\n\n"
        f"# Portfolio manager's decision (typed)\n{json.dumps(state.get('portfolio_decision') or {}, ensure_ascii=False)}\n\n"
        f"# Lint report (automated; confirm or dismiss)\n{lint_lines}\n\n"
        f"# Transcript\n{_transcript(state)}"
    )


def _run_once(llm, prompt: str, facts: Facts, callbacks=None) -> dict:
    from langchain_core.messages import HumanMessage, ToolMessage

    bound = llm.bind_tools(TOOLS)
    messages = [HumanMessage(content=prompt)]
    for _ in range(MAX_TOOL_ROUNDS):
        reply = bound.invoke(messages, config={"callbacks": callbacks or []})
        messages.append(reply)
        calls = getattr(reply, "tool_calls", None) or []
        if not calls:
            messages.append(HumanMessage(content="Finish by calling submit_review."))
            continue
        for call in calls:
            name, args = call["name"], call.get("args") or {}
            if name == "submit_review":
                return args
            result = calc(args.get("expression", "")) if name == "calc" else fact(facts, args.get("key", ""))
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    raise RuntimeError("the editor did not submit a review")


def review(llm, state: dict, lint_report: dict, earlier_errata: str = "", callbacks=None,
           budget_seconds: float = RETRY_BUDGET_SECONDS, sleep=time.sleep) -> dict:
    """The editor's review, with retries and backoff for up to ``budget_seconds``.
    Raises EditorUnavailable when the budget runs out."""
    facts = Facts(state.get("fact_sheet"))
    prompt = build_prompt(state, lint_report, earlier_errata)
    started, delay, last = time.monotonic(), 5.0, None
    while True:
        try:
            out = _run_once(llm, prompt, facts, callbacks)
            for key in ("findings", "digest_patch", "decision_flags"):
                out[key] = out.get(key) or []
            out["editor_note"] = (out.get("editor_note") or "").strip()
            out["hold_reason"] = (out.get("hold_reason") or "").strip()
            return out
        except Exception as exc:  # noqa: BLE001 — retried, then a hold
            if is_fatal(exc):
                raise
            last = exc
            logger.warning("editor attempt failed: %s", exc)
            if time.monotonic() - started + delay > budget_seconds:
                raise EditorUnavailable(f"editor unavailable: {type(last).__name__}: {last}") from last
            sleep(delay)
            delay = min(delay * 2, 120)


# ---- patching the digest -----------------------------------------------------------------

_PATH = re.compile(r"^(?P<field>\w+)(?:\[(?P<index>\d+)\](?:\.(?P<sub>\w+))?)?$")
_PATCHABLE = {"headline", "bull_thesis", "bull_points", "bear_thesis", "bear_points", "ruling",
              "market_excerpt", "sentiment_excerpt", "news_excerpt", "fundamentals_excerpt",
              "trader_excerpt", "conviction_note", "exit_triggers", "sizing", "entry_style", "review_cycle"}


def apply_patch(digest: dict, patch: list[dict], facts: Facts) -> tuple[dict, list[dict]]:
    """(patched digest, applied patches). A replacement whose text fails the
    lint (a figure off the sheet, a wrong direction) is applied as a deletion
    instead. Unknown fields are ignored; the rating is never touched."""
    out = json.loads(json.dumps(digest or {}))
    applied = []

    def order(p: dict):
        # Indices refer to the digest as written: replacements first, then
        # deletions from the highest index down, so none shifts another.
        m_ = _PATH.match((p.get("field") or "").strip())
        index = int(m_.group("index")) if m_ and m_.group("index") else -1
        return (p.get("action") == "delete", -index)

    for p in sorted(patch or [], key=order):
        m = _PATH.match((p.get("field") or "").strip())
        if not m or m.group("field") not in _PATCHABLE or m.group("field") not in out:
            continue
        action, value = p.get("action"), p.get("value") or ""
        if action == "replace":
            bad = [f for f in lint_text(value, facts, "editor", f"digest.{m.group('field')}") if f.blocking]
            if bad:
                action = "delete"
        field, index, sub = m.group("field"), m.group("index"), m.group("sub")
        target = out.get(field)
        if index is None:
            if action == "delete":
                out[field] = [] if isinstance(target, list) else ("" if isinstance(target, str) else None)
            elif isinstance(target, str) or target is None:
                out[field] = value
            else:
                continue
        else:
            i = int(index)
            if not isinstance(target, list) or i >= len(target):
                continue
            if action == "delete":
                target.pop(i)
            elif sub and isinstance(target[i], dict):
                target[i][sub] = value
            elif isinstance(target[i], str):
                target[i] = value
            else:
                continue
        applied.append({"field": p.get("field"), "action": action})
    return out, applied


def create_editor_llm(config: dict, callbacks: list | None = None):
    """The editor's model: ``config["editor_llm"]`` on the run's provider, at
    ``config["editor_effort"]`` (platform: claude-opus-5-5, high)."""
    from tradingagents.llm_clients.factory import build_llm_kwargs, create_llm_client

    provider = config["llm_provider"].lower()
    kwargs = build_llm_kwargs({**config, "anthropic_effort": config.get("editor_effort") or config.get("anthropic_effort"),
                               "temperature": None})
    extra = {"callbacks": callbacks} if callbacks else {}
    return create_llm_client(provider=provider, model=config.get("editor_llm") or config["deep_think_llm"],
                             base_url=config.get("backend_url"), **kwargs, **extra).get_llm()
