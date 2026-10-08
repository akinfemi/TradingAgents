"""Shared prompt text for the quality layer (REPORT_QUALITY_PLAN R4)."""

CITE_RULE = (
    "**Figures rule.** Every figure you state must come from the fact sheet above, cited by its "
    "key in square brackets right after it, e.g. \"revenue of $83.8M [F:revenue.2026Q2]\". A figure "
    "that is not in the fact sheet must be derived in your text from cited keys (show the "
    "arithmetic), or not used. Never compute an indicator or ratio the fact sheet already gives; "
    "never present a year-to-date or trailing figure as a quarter; label every figure with its "
    "period. Quarters listed as already reported are past results, not upcoming ones. Where the "
    "fact sheet says a source is unavailable, say the figure is unavailable rather than estimating it."
)


def fact_sheet_block(text: str) -> str:
    return f"\n\n{text}\n\n{CITE_RULE}"
