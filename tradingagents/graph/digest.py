"""Post-run report digest: one structured extraction call over the final state.

The pipeline's transcripts are the product of record, but they are thousands
of words each — far too long to read on the report page. After the graph
finishes, ``generate_report_digest`` makes a single quick-model call that
distills the run into the curated layer the report page and PDF render:
thesis lines, evidence bullets, the ruling, risk stances, conviction, and
exit triggers (see ``ReportDigest`` in ``agents/schemas.py``).

The digest is strictly optional decoration: any failure logs a warning and
returns ``None``, and the run itself is never blocked or failed by it.
"""

from __future__ import annotations

import logging
from typing import Any

from tradingagents.agents.schemas import ReportDigest
from tradingagents.agents.utils.structured import bind_structured

logger = logging.getLogger(__name__)

# Per-section budget fed to the extraction prompt. Debate histories run to
# tens of thousands of characters; the conclusions cluster at both ends
# (opening thesis, closing rebuttals), so long sections keep head + tail.
_SECTION_CHAR_BUDGET = 9000

_ELISION = "\n\n[... middle of transcript elided for length ...]\n\n"


def _clip(text: str | None, budget: int = _SECTION_CHAR_BUDGET) -> str | None:
    if not text or len(text) <= budget:
        return text
    head = int(budget * 0.6)
    tail = budget - head
    return text[:head] + _ELISION + text[-tail:]


def _sources_from_state(final_state: dict) -> list[tuple[str, str | None]]:
    debate = final_state.get("investment_debate_state") or {}
    risk = final_state.get("risk_debate_state") or {}
    return [
        ("Market analyst report", _clip(final_state.get("market_report"))),
        ("Sentiment analyst report", _clip(final_state.get("sentiment_report"))),
        ("News analyst report", _clip(final_state.get("news_report"))),
        ("Fundamentals analyst report", _clip(final_state.get("fundamentals_report"))),
        ("Bull researcher argument", _clip(debate.get("bull_history"))),
        ("Bear researcher argument", _clip(debate.get("bear_history"))),
        ("Research manager ruling", _clip(debate.get("judge_decision"))),
        ("Trader plan", _clip(final_state.get("trader_investment_plan"))),
        ("Aggressive risk analyst", _clip(risk.get("aggressive_history"))),
        ("Neutral risk analyst", _clip(risk.get("neutral_history"))),
        ("Conservative risk analyst", _clip(risk.get("conservative_history"))),
        ("Risk judge ruling", _clip(risk.get("judge_decision"))),
        ("Final portfolio decision", _clip(final_state.get("final_trade_decision"))),
    ]


def build_digest_prompt(final_state: dict, computed_context: str | None = None) -> str:
    ticker = final_state.get("company_of_interest", "the instrument")
    trade_date = final_state.get("trade_date", "")
    parts = [
        f"You are the report editor for an AI equity-research pipeline. A full "
        f"multi-agent run on {ticker} (trade date {trade_date}) just finished; "
        f"its reports and debate transcripts are below. Extract the digest "
        f"fields exactly as described by the output schema.",
        "",
        "Rules:",
        "- Every claim and number must come from the source text below. Never invent figures.",
        "- Prefer the points each side argued hardest; keep the concrete numbers.",
        "- Write for a reader deciding whether to trust the verdict without reading the transcripts.",
        "- Leave a field null when its source report is missing from the input "
        "- never write placeholder text such as 'Not provided' into a field.",
        # Numeric-consistency rules: a professional reader divides price by
        # EPS first — one contradicted multiple discredits the whole report.
        "- Label the basis of every margin or valuation multiple you quote "
        "(TTM vs fiscal-year), and keep one basis consistent within a field.",
        "- Do not quote a valuation multiple from the source text without "
        "sanity-checking it against the other figures you are quoting "
        "(price, EPS, market cap, net income); drop or correct multiples "
        "that contradict them. Never cite a forward multiple without the "
        "forward figure it implies.",
        # Style rules from professional review: repetition, loose rating
        # words, imperatives, and inflated ratios are machine-output tells.
        "- Each field must contribute NEW information. Never restate the "
        "same statistics cluster across fields — later fields reference "
        "conclusions ('the balance-sheet strength above'), not the numbers "
        "again.",
        "- Rating words (Buy/Overweight/Hold/Underweight/Sell) are formal "
        "ratings only — for actions use accumulate/add/hold off/trim/exit.",
        "- Descriptive voice, never imperative commands: 'the plan calls "
        "for accumulating in tranches', not 'Buy now'.",
        "- State ratios honestly: 'doubled/tripled' only when the ratio "
        "rounds there; otherwise give the multiple (e.g. 'rose ~2.6x').",
        "- If cash and equivalents exceed total debt, that is a NET CASH "
        "position — never describe it as net debt; if you cannot reconcile "
        "a net-debt figure with the cash and debt figures you cite, omit it.",
        "",
    ]
    if computed_context:
        parts.append(
            "## Computed figures (authoritative — derived from structured "
            "vendor data, not model text)\n\n"
            "When the source text disagrees with these figures, THESE are "
            "correct: quote these and do not repeat the contradicted number.\n\n"
            f"{computed_context}\n"
        )
    for label, text in _sources_from_state(final_state):
        if text:
            parts.append(f"## {label}\n\n{text}\n")
    return "\n".join(parts)


def generate_report_digest(
    quick_llm: Any,
    final_state: dict,
    callbacks: list | None = None,
    computed_context: str | None = None,
) -> dict | None:
    """Extract a ``ReportDigest`` from a finished run's state as a plain dict.

    ``computed_context``: optional block of authoritative figures computed
    from structured vendor data (price, EPS, computed multiples). When given,
    the prompt instructs the model to prefer these over any contradicting
    numbers in the transcripts — the numeric-consistency pass.

    Returns ``None`` — never raises — when the provider lacks structured
    output or the call fails; the digest is optional and must not take a
    finished run down with it.
    """
    structured_llm = bind_structured(quick_llm, ReportDigest, "Report digest")
    if structured_llm is None:
        return None
    try:
        config = {"callbacks": callbacks} if callbacks else None
        result = structured_llm.invoke(
            build_digest_prompt(final_state, computed_context), config=config
        )
        if result is None:
            raise ValueError("structured output returned no parsed result")
        return result.model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001 — decoration only, never fatal
        logger.warning("Report digest extraction failed (%s); continuing without it", exc)
        return None
