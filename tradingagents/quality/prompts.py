"""Shared prompt text for the quality layer (REPORT_QUALITY_PLAN R4)."""

CITE_RULE = (
    "**Figures rule.** Every figure you state must come from the fact sheet above, cited by its "
    "key in square brackets right after it, e.g. \"revenue of $83.8M [F:revenue.2026Q2]\". Copy keys "
    "exactly as the fact sheet prints them, period included ([F:fcf.2026Q2], never [F:fcf]). A figure "
    "that is not in the fact sheet must be derived in your text from cited keys (show the "
    "arithmetic), or not used. Never compute an indicator or ratio the fact sheet already gives; "
    "never present a year-to-date or trailing figure as a quarter; label every figure with its "
    "period. Quarters listed as already reported are past results, not upcoming ones. Where the "
    "fact sheet says a source is unavailable, say the figure is unavailable rather than estimating it."
    "\n\n**Data you may not use**, even when a source mentions it: short interest, short float or "
    "days to cover; consensus estimates; analyst price targets. Analyst rating changes may be "
    "mentioned only as the news reported them, with the outlet."
)


def fact_sheet_block(text: str) -> str:
    return f"\n\n{text}\n\n{CITE_RULE}"


# R7: the research manager adjudicates numbers before it rules.
ADJUDICATION = (
    "**Adjudicate the numbers first.** Before ruling, list the figures the two sides rely on or "
    "dispute (up to eight), and resolve each against the fact sheet: verified (cite its key), wrong "
    "(give the fact sheet's value and key), or not on the fact sheet (unverified). Score arguments by "
    "whether their figures verify, not by tone or confidence: a point resting on a wrong or unverified "
    "figure loses that point. Then rule on the strength of what verifies."
)

# R7: the risk stage's scope and length (was ~7K words re-arguing the debate).
RISK_SCOPE = (
    "**Scope of this turn.** Cover only: (1) liquidity (average daily volume, ATR from the fact "
    "sheet); (2) event gaps (the next earnings date from the fact sheet's calendar, and how the "
    "position should be sized into it); (3) position size and stops expressed in ATR terms; (4) what "
    "would change the call. Do not re-argue the bull and bear debate; it has been ruled on. Keep the "
    "turn under 500 words."
)
