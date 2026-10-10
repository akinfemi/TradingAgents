"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# LLMs sometimes write a placeholder string ("None", "N/A", ...) into an optional
# numeric field instead of omitting it. Coerce those to None so the structured
# call validates instead of erroring (#1058). Pydantic still parses real numeric
# strings ("189.5") to float.
_NULLISH_FLOAT = {"", "none", "n/a", "na", "null", "nil", "-", "tbd", "unknown"}


def _coerce_optional_float(value):
    """Normalise an LLM-written optional numeric field before validation.

    Three shapes show up in practice: a placeholder string ("None", "N/A") in
    place of an omitted value (#1058); a percentage where a price was asked for
    ("15%", #1288); and a human-formatted price ("$1,234.50"). A percentage
    cannot be salvaged into an absolute level -- reading "15%" as 15 would put a
    stop at $15 on a $600 stock -- so it is dropped like a placeholder, leaving
    one bad field to null out instead of failing the whole proposal. A formatted
    price is reduced to its number.

    Anything that is not a single number is dropped the same way. A range
    ("150-160") or a hedge ("around 150") would otherwise reach pydantic, fail
    validation, and discard the whole decision, losing every field the model got
    right along with the price.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.lower() in _NULLISH_FLOAT or text.endswith("%"):
        return None
    cleaned = text.replace(",", "").lstrip("$€£¥").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _coerce_str_list(value):
    """A list-of-strings field as a list: models send None, one
    comma-separated string, or a JSON-ish list with non-string items, and a
    failed list would discard the whole structured decision."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip().strip("[]")
        return [part.strip().strip("'\"` ") for part in re.split(r"[,;\n]", text) if part.strip().strip("'\"` ")]
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    return [str(value)]


def _coerce_text(value):
    """A free-text field given as a number or a list (seen from weaker
    tiers): a number becomes its text, a list its items joined. None stays None."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value if v is not None)
    if isinstance(value, (int, float)):
        return str(value)
    return value


def _coerce_enum(value, enum):
    """An enum field matched without regard to case, markdown or trailing
    punctuation ("**buy**", "BUY.", " Hold ")."""
    if not isinstance(value, str):
        return value
    text = value.strip().strip("*_`'\" .:").lower()
    return next((member.value for member in enum if member.value.lower() == text), value)


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(StrEnum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(StrEnum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Conflicting arguments alone are not a "
            "reason to Hold: commit to the stronger side, sized by how "
            "decisively it wins. Choose Hold only when the evidence is still "
            "balanced after weighing, or too thin to support a call."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including sizing guidance relative to a standard allocation. The "
            "research team does not see the caller's holdings; the trader and "
            "portfolio manager apply the actual position."
        ),
    )


    @field_validator("recommendation", mode="before")
    @classmethod
    def _rating_any_case(cls, v):
        return _coerce_enum(v, PortfolioRating)

    @field_validator("rationale", "strategic_actions", mode="before")
    @classmethod
    def _text(cls, v):
        return _coerce_text(v)


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: float | None = Field(
        default=None,
        description=(
            "Optional entry price target as an absolute number in the instrument's "
            "quote currency (e.g. 189.5), never a percentage or a range. Omit it "
            "if you cannot state a specific level."
        ),
    )
    stop_loss: float | None = Field(
        default=None,
        description=(
            "Optional stop-loss as an absolute price in the instrument's quote "
            "currency (e.g. 172.0), never a percentage. Convert a percentage "
            "distance to the price level it implies, or omit it."
        ),
    )
    position_sizing: str | None = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )

    @field_validator("entry_price", "stop_loss", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)

    @field_validator("action", mode="before")
    @classmethod
    def _action_any_case(cls, v):
        return _coerce_enum(v, TraderAction)

    @field_validator("reasoning", "position_sizing", mode="before")
    @classmethod
    def _text(cls, v):
        return _coerce_text(v)


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    # Named even when absent, so a reader can tell a level the trader chose not
    # to give from one the schema never asked for.
    for label, value in (("Entry Price", proposal.entry_price),
                         ("Stop Loss", proposal.stop_loss),
                         ("Position Sizing", proposal.position_sizing)):
        parts.extend(["", f"**{label}**: {value if value is not None and value != '' else 'not provided'}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate. "
            "Conflicting arguments alone are not a reason to Hold: commit to the "
            "stronger side, sized by how decisively it wins. Choose Hold only "
            "when the evidence is still balanced after weighing, or too thin to "
            "support a call."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing "
            "and key risk levels, over the rating horizon given in the prompt. "
            "Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: float | None = Field(
        default=None,
        description=(
            "Target price in the instrument's quote currency, for the end of the "
            "rating horizon, derived in target_math. Leave empty when the fact "
            "sheet can't support one, and say why in the thesis."
        ),
    )
    valuation_method: str | None = Field(
        default=None,
        description=(
            "How the target is derived, e.g. 'EV/Sales on TTM revenue', "
            "'EV/Sales on run-rate revenue', 'P/E on TTM EPS'."
        ),
    )
    valuation_inputs: list[str] = Field(
        default_factory=list,
        description="The fact-sheet keys the target uses, e.g. ['ev_sales.ttm', 'ttm_revenue.2026Q2', 'shares.cover'].",
    )
    target_math: str | None = Field(
        default=None,
        description=(
            "The arithmetic from the inputs to the target, ending '= $<target>', e.g. "
            "'12x × $174.1M TTM revenue = $2.09B EV; + $1.38B cash − $0 debt = $3.47B equity; "
            "÷ 570.6M shares = $6.08'."
        ),
    )
    bear_case_value: float | None = Field(
        default=None, description="Per-share value in the bear case, same method, stated assumption in target_math.",
    )
    bull_case_value: float | None = Field(
        default=None, description="Per-share value in the bull case, same method.",
    )
    # R8 (one-pager spec, 2026-10-10): the cases as a reader scans them.
    valuation_rationale: str | None = Field(
        default=None,
        description="One sentence on why this method suits this company now, e.g. 'EV/EBIT, because a $98B "
                    "non-operating gain distorts net income'.",
    )
    target_multiple: float | None = Field(default=None, description="The multiple the target applies, e.g. 18.0.")
    bear_multiple: float | None = Field(default=None, description="The bear case's multiple.")
    bull_multiple: float | None = Field(default=None, description="The bull case's multiple.")
    base_case: str | None = Field(
        default=None,
        description="What has to be true for the target, in one sentence with its number, e.g. 'Cloud growth holds "
                    "above 30% and the multiple settles at its 5-year 75th percentile'.",
    )
    bear_case: str | None = Field(default=None, description="What has to be true for the bear value, one sentence.")
    bull_case: str | None = Field(default=None, description="What has to be true for the bull value, one sentence.")
    execution_timing: str | None = Field(
        default=None,
        description=(
            "How to enter or exit the position, e.g. 'build over 3-5 sessions' "
            "or 'trim on strength over the next week'. Not the rating horizon."
        ),
    )
    time_horizon: str | None = Field(
        default=None,
        description=(
            "Legacy; leave empty. The rating horizon is fixed by the prompt."
        ),
    )

    @field_validator("price_target", "bear_case_value", "bull_case_value", "target_multiple", "bear_multiple",
                     "bull_multiple", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)

    @field_validator("valuation_inputs", mode="before")
    @classmethod
    def _inputs_as_list(cls, v):
        # None or "ev_sales.ttm, ttm_revenue.2026Q2" failed validation and
        # dropped the whole decision to free text.
        return _coerce_str_list(v)

    @field_validator("rating", mode="before")
    @classmethod
    def _rating_any_case(cls, v):
        return _coerce_enum(v, PortfolioRating)

    @field_validator("executive_summary", "investment_thesis", "valuation_method", "target_math",
                     "execution_timing", "time_horizon", "valuation_rationale", "base_case", "bear_case",
                     "bull_case", mode="before")
    @classmethod
    def _text(cls, v):
        return _coerce_text(v)


def render_pm_decision(decision: PortfolioDecision, rating_horizon: str | None = None) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    # Named even when absent: a missing line reads as a field nobody asked for,
    # so a reader cannot tell "no target" from "target not reported".
    target = decision.price_target if decision.price_target is not None else "not provided"
    parts.extend(["", f"**Price Target**: {target}"])
    if decision.valuation_method:
        parts.extend(["", f"**Valuation Method**: {decision.valuation_method}"])
    if decision.valuation_inputs:
        parts.extend(["", f"**Valuation Inputs**: {', '.join(decision.valuation_inputs)}"])
    if decision.target_math:
        parts.extend(["", f"**Target Math**: {decision.target_math}"])
    if decision.bear_case_value is not None:
        parts.extend(["", f"**Bear Case Value**: {decision.bear_case_value}"])
    if decision.bull_case_value is not None:
        parts.extend(["", f"**Bull Case Value**: {decision.bull_case_value}"])
    if rating_horizon:  # the window the call is graded on, e.g. "3 months (63 trading days) vs SPY"
        parts.extend(["", f"**Rating Horizon**: {rating_horizon}"])
    if decision.execution_timing:
        parts.extend(["", f"**Execution Timing**: {decision.execution_timing}"])
    if decision.time_horizon:  # legacy runs only; new runs rate over the fixed horizon
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(StrEnum):
    """Discrete sentiment direction produced by the Sentiment Analyst.

    Six tiers keep the signal granular enough to be actionable while remaining
    small enough for every provider to map reliably from its JSON output.
    """

    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"


class SentimentReport(BaseModel):
    """Structured sentiment report produced by the Sentiment Analyst.

    Replaces the previous free-form prose output so downstream consumers
    (dashboards, audit logs, PDF renderers, other agents) can read
    ``overall_band`` and ``overall_score`` without maintaining fragile regex
    fallbacks that drift with every model release. ``narrative`` preserves the
    rich source-by-source analysis; ``render_sentiment_report`` prepends a
    deterministic header so the saved report stays human-readable.
    """

    overall_band: SentimentBand = Field(
        description=(
            "Overall sentiment direction. Exactly one of: "
            "Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. "
            "Use Mixed when sources point in clearly different directions. "
            "Use Neutral only when all sources are genuinely silent or non-committal."
        ),
    )
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Numeric sentiment intensity on a 0–10 scale. "
            "0 = maximally bearish, 5 = neutral, 10 = maximally bullish. "
            "Guideline for consistency with overall_band: "
            "Bullish ~6.5–10, Mildly Bullish ~5.5–6.4, Neutral/Mixed ~4.5–5.5, "
            "Mildly Bearish ~3.5–4.4, Bearish ~0–3.4. "
            "Only the 0–10 bounds are enforced."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the assessment based on data quality and sample size. "
            "Use 'low' when one or more sources returned a placeholder or fewer "
            "than 5 data points; 'medium' when data is present but sparse; "
            "'high' when all three sources returned substantive data."
        ),
    )
    narrative: str = Field(
        description=(
            "Full sentiment report covering, in order: "
            "(1) source-by-source breakdown with specific evidence (cite message "
            "counts, ratios, notable posts); "
            "(2) cross-source divergences and alignments; "
            "(3) dominant narrative themes; "
            "(4) catalysts and risks surfaced by the data; "
            "(5) a markdown table summarising key sentiment signals, their "
            "direction, source, and supporting evidence. "
            "Keep it informative and substantive: develop each section thoroughly "
            "with concrete evidence so every point adds new signal for the trader."
        ),
    )


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a SentimentReport to the markdown shape the rest of the system expects.

    The structured header (band + score + confidence) is prepended to the
    narrative so the saved report is both human-readable and machine-parseable
    without regex.
    """
    return "\n".join([
        f"**Overall Sentiment:** **{report.overall_band.value}** "
        f"(Score: {report.overall_score:.1f}/10)",
        f"**Confidence:** {report.confidence.capitalize()}",
        "",
        report.narrative,
    ])


# --- tickeragent.ai: the report digest (post-run curated summary) ----------------

class DigestPoint(BaseModel):
    """One crisp evidence point: a bolded headline plus a single sentence."""

    title: str = Field(
        description="Three-to-six-word headline for the point, e.g. 'Capex is strategic, not destructive'.",
    )
    detail: str = Field(
        description=(
            "One sentence of supporting evidence with the concrete numbers "
            "from the source report, e.g. '$24.6B of free cash flow survived "
            "$27.9B of quarterly capex'."
        ),
    )


class ExitTrigger(DigestPoint):
    """A trigger as a row: what is watched, where it is, where it fires, and
    when it is next checked (one-pager spec, 2026-10-10)."""

    metric: str | None = Field(default=None, description="What is watched, e.g. 'Data Center revenue growth, YoY'.")
    current: str | None = Field(default=None, description="Its value now, from the fact sheet, e.g. '107%'.")
    threshold: str | None = Field(
        default=None,
        description="Where it fires, with deliberate headroom from today's value (not at it), e.g. 'below 60%'.",
    )
    check: str | None = Field(default=None, description="When it is next checked, e.g. 'Q3 report, Nov 4'.")
    moves_to: str | None = Field(
        default=None,
        description="Which way the rating would move if it fires, e.g. 'toward Hold' or 'to Sell'. A condition "
                    "that only confirms the call is not a trigger.",
    )


class RiskLens(BaseModel):
    """One risk analyst's position, condensed to a stance plus rationale."""

    stance: str = Field(
        description="One short headline sentence capturing the position, e.g. 'Lean into the momentum.'",
    )
    summary: str = Field(
        description=(
            "What this lens would do differently from the decision, in one "
            "third-person sentence about the stock ('Would reconsider the trims "
            "if the stock holds its post-earnings range and Data Center growth "
            "stays above 80%'), then at most one sentence of rationale. Never "
            "quote the turn: no 'I', 'we' or 'you', no BUY/SELL from the trader's "
            "proposal, and no talk about the lens itself ('this lens adds'). "
            "Do not restate the debate, the ruling, or another lens."
        ),
    )


class ReportDigest(BaseModel):
    """Curated, chart-side summary layer extracted from a completed run.

    One extraction call after the pipeline finishes turns the multi-thousand-
    word transcripts into the crisp content the report page and PDF render:
    thesis lines, evidence bullets, risk stances, conviction, and the exit
    triggers the decision leaves behind. Every claim must be drawn from the
    source reports — never invented.
    """

    headline: str = Field(
        description=(
            "A 6-12 word editorial title stating the report's view, like a "
            "sell-side note headline: '<Rating>: <the one reason>', e.g. "
            "'Overweight: operating strength outweighs capex risk'. No "
            "sizing mechanics, no tranches, no price levels."
        ),
    )
    bull_thesis: str = Field(
        description="The bull researcher's core thesis in one punchy sentence.",
    )
    bull_points: list[DigestPoint] = Field(
        description="The bull's 3-4 strongest evidence points, each anchored in numbers from the debate.",
    )
    bear_thesis: str = Field(
        description="The bear researcher's core thesis in one punchy sentence.",
    )
    bear_points: list[DigestPoint] = Field(
        description="The bear's 3-4 strongest evidence points, each anchored in numbers from the debate.",
    )
    ruling: str = Field(
        description=(
            "The research manager's ruling in two to three sentences: which "
            "side won and why, the deciding argument, and what caps or boosts "
            "conviction. No sizing, levels or valuation (those belong to the "
            "decision). Written for a reader who skipped the transcripts."
        ),
    )
    debate_winner: Literal["bull", "bear", "split"] = Field(
        description="Which side the research manager's ruling favored; 'split' when genuinely balanced.",
    )
    market_excerpt: str | None = Field(
        default=None,
        description="Two-sentence takeaway of the market/technical report (trend, momentum, entry read). None if that report is absent.",
    )
    sentiment_excerpt: str | None = Field(
        default=None,
        description="Two-sentence takeaway of the sentiment report (direction, sources, froth signals). None if absent.",
    )
    news_excerpt: str | None = Field(
        default=None,
        description="Two-sentence takeaway of the news report (headline mix, catalysts, risks). None if absent.",
    )
    fundamentals_excerpt: str | None = Field(
        default=None,
        description="Two-to-three-sentence takeaway of the fundamentals report with the key figures (margins, growth, balance sheet). None if absent.",
    )
    trader_excerpt: str | None = Field(
        default=None,
        description=(
            "One-to-two-sentence summary of the trader's transaction plan (entry style, levels) in "
            "descriptive words (accumulate/trim/exit), never the trader's BUY/HOLD/SELL. None if absent."
        ),
    )
    risk_aggressive: RiskLens | None = Field(
        default=None, description="The aggressive risk analyst's lens. None if that debate is absent.",
    )
    risk_neutral: RiskLens | None = Field(
        default=None, description="The neutral risk analyst's lens. None if absent.",
    )
    risk_conservative: RiskLens | None = Field(
        default=None, description="The conservative risk analyst's lens. None if absent.",
    )
    risk_alignment: Literal["aggressive", "neutral", "conservative"] | None = Field(
        default=None,
        description="Which risk lens the final decision sided with, if the risk judge or portfolio manager says so.",
    )
    conviction: int = Field(
        ge=0,
        le=100,
        description=(
            "Debate margin, 0-100: how decisively the evidence in the bull/"
            "bear debate favoured the final rating. 0 is evenly split, 100 "
            "one-sided. It measures the strength of the argument, NOT the "
            "probability the call is right and NOT the writers' confidence. "
            "Use the whole scale: a near-even split is 0-20, a narrow win "
            "with real open concerns 20-45, a clear win 45-75, a decisive "
            "win on checked evidence with aligned reviews 75+. The ruling's "
            "wording must match the band ('narrowly' only under 45, "
            "'decisively' only from 75). (Shown to readers as 'Debate "
            "margin'; the field keeps its old name.)"
        ),
    )
    conviction_note: str = Field(
        description=(
            "One short clause (under 10 words) explaining the debate margin, "
            "e.g. 'Measured -- bull won the debate, not decisively'. "
            "Rendered beside a meter; it must stay short."
        ),
    )
    entry_style: str | None = Field(
        default=None,
        description=(
            "How the decision says to enter, as a short phrase (under 8 "
            "words), e.g. 'Pullbacks over breakouts' or '3-4 tranches over "
            "several weeks'. None if unspecified."
        ),
    )
    sizing: str | None = Field(
        default=None,
        description=(
            "Position sizing from the decision, as a short phrase (under 6 "
            "words), e.g. '4-6% core position'. Rendered as a fact chip -- "
            "no clauses or caveats. None if unspecified."
        ),
    )
    review_cycle: str | None = Field(
        default=None,
        description=(
            "When to revisit the call, as a short phrase (under 6 words), "
            "e.g. 'Quarterly, or on trigger'. None if unspecified."
        ),
    )
    call_thesis: str | None = Field(
        default=None,
        description="The call in one plain-language sentence a non-specialist follows: what the stock does over "
                    "the horizon and why. No sizing, no levels.",
    )
    deciding_variable: str | None = Field(
        default=None,
        description="The single variable that decides the call, stated as something you could test, with its "
                    "number and when it is known, e.g. 'Cloud growth must stay above 30% at the Q3 print (Oct 28) "
                    "to justify 10x sales'.",
    )
    catalyst: str | None = Field(
        default=None,
        description="Why now: the dated catalyst inside the horizon and what it must show, e.g. 'Q3 report, Nov 4: "
                    "Data Center growth against 107% last quarter'. None if the decision names none.",
    )
    flags: list[str] = Field(
        default_factory=list,
        description="Anomalies and data gaps a reader should know before trusting the numbers, from the fact-sheet "
                    "flags and missing sources given below: one sentence each, with its figure, in reader words "
                    "(e.g. 'Cash and investments rose $116B in Q2 against −$5.9B of free cash flow: most of it is an "
                    "equity stake marked to market, not cash.'). At most five. Empty when there are none.",
    )
    exit_triggers: list[ExitTrigger] = Field(
        description=(
            "What would change the view: the 3-4 conditions that would make the "
            "rating wrong and move it (for an Underweight, what would move it "
            "toward Hold or above; a further slide that only confirms the call "
            "is not one), each with moves_to — the watchlist this report leaves behind. Each title "
            "names the trigger, each detail says what observable change fires "
            "it, in one sentence of at most 240 characters. At least one is a "
            "fundamental threshold with its current value (segment growth, "
            "gross or operating margin, free cash flow, guidance), not a "
            "price level or indicator; at most one is price-based. Fill metric, "
            "current, threshold, check and moves_to for each; thresholds keep "
            "headroom from today's value, and a year-on-year threshold is checked "
            "against the next report's comparison base on the fact sheet."
        ),
    )

    @field_validator("exit_triggers", mode="before")
    @classmethod
    def _points_as_triggers(cls, v):
        # A plain DigestPoint (older callers, the editor's patches) is a trigger
        # without its row fields.
        return [p.model_dump() if isinstance(p, DigestPoint) and not isinstance(p, ExitTrigger) else p
                for p in (v or [])]
