"""The quality loop (REPORT_QUALITY_PLAN R6): run, review, revise or hold.

    pass 0: propagate (gates on) → digest → lint → editor → patch → re-lint
    open load-bearing errors or decision flags?
        no  → clean (pass 0) or revised (a later pass)
        yes → errata = lint ∪ editor findings; restart at the earliest stage
              that introduced one, keeping every earlier stage's output; re-run
              with the errata on top of every re-run prompt.
    after two revisions, still open → held (refunded, not published)

The editor failing past its retry budget is also a hold: nothing publishes
unchecked.
"""

from __future__ import annotations

import logging

from tradingagents.quality import editor, errata
from tradingagents.quality.lint import Facts, lint_state

logger = logging.getLogger(__name__)

MAX_REVISIONS = 2
ANALYST_REPORT_KEYS = {"market": "market_report", "social": "sentiment_report", "news": "news_report",
                       "fundamentals": "fundamentals_report"}
VERSION = 1


def _load_bearing(lint: dict) -> list[dict]:
    return [f for f in lint.get("flags") or [] if f.get("severity") == "load_bearing"]


def run_with_quality(graph, ticker: str, trade_date, editor_llm, *, on_progress=None, callbacks=None,
                     review_fn=editor.review, **propagate_kwargs):
    """(final_state, rating). ``final_state["quality"]`` carries the verdict:
    ``{version, status: clean|revised|held, passes, hold_reason, editor_note,
    lint, gates, open_errata}``."""
    passes: list[dict] = []
    all_errata: list[dict] = []
    revision = None
    status, hold_reason, note = "held", "", ""
    final_state, rating = {}, None
    for attempt in range(MAX_REVISIONS + 1):
        final_state, rating = graph.propagate(ticker, trade_date, on_progress=on_progress, callbacks=callbacks,
                                              revision=revision, record=False, **propagate_kwargs)
        if not final_state.get("fact_sheet"):
            # No sheet, nothing to check against: publish as before the quality layer.
            status = "unchecked"
            break
        if on_progress:
            on_progress("Quality Review", {}, final_state)
        lint = (final_state.get("quality") or {}).get("lint") or lint_state(final_state)
        try:
            review = review_fn(editor_llm, final_state, lint, errata.render(all_errata), callbacks=callbacks)
        except editor.EditorUnavailable as exc:
            logger.error("editor unavailable for %s: %s", ticker, exc)
            status, hold_reason = "held", "editor_unavailable"
            passes.append({"pass": attempt, "lint": lint, "editor": None, "error": str(exc)[:300]})
            break
        patched, applied = editor.apply_patch(final_state.get("report_digest") or {}, review["digest_patch"],
                                              Facts(final_state.get("fact_sheet")))
        if final_state.get("report_digest") is not None:
            final_state["report_digest"] = patched
        relint = lint_state(final_state)
        open_flags = _load_bearing(relint)
        editor_lb = [f for f in review["findings"] if f.get("severity") == "load_bearing"]
        # A load-bearing finding the editor fixed in the digest is closed by its
        # patch; one in the transcript or decision still needs a revision.
        unpatched = [f for f in editor_lb if not str((f.get("location") or {}).get("stage", "")).startswith("digest")
                     or not applied]
        note = review.get("editor_note") or note
        passes.append({"pass": attempt, "lint": lint, "editor": review, "patch_applied": applied,
                       "relint": relint, "open": len(open_flags), "decision_flags": review["decision_flags"]})
        if not open_flags and not review["decision_flags"] and not unpatched:
            status = "clean" if attempt == 0 else "revised"
            break
        if attempt == MAX_REVISIONS:
            status = "held"
            hold_reason = review.get("hold_reason") or "load-bearing errors remained after two revisions"
            break
        new = errata.merge(
            all_errata,
            errata.from_lint(open_flags),
            errata.from_editor(unpatched),
            [{"severity": "load_bearing", "kind": "decision_flag", "source": "research_manager",
              "field": "rating", "quote": flag, "problem": flag,
              "correct": "re-weigh the evidence with the errata and re-derive the call"} for flag in review["decision_flags"]],
        )
        all_errata = new
        rerun, first_later = errata.restart_group(all_errata)
        kept = errata.kept_outputs(final_state, rerun, first_later, ANALYST_REPORT_KEYS)
        revision = {"fact_sheet": final_state.get("fact_sheet"), "fact_sheet_text": final_state.get("fact_sheet_text"),
                    "review_errata": errata.render(all_errata), "kept": kept}
        passes[-1]["restart"] = {"analysts": sorted(rerun), "from": first_later, "kept": sorted(kept)}
        if on_progress:
            on_progress(f"Revision {attempt + 1}", {}, final_state)

    quality = dict(final_state.get("quality") or {})
    quality.update(version=VERSION, status=status, passes=passes, hold_reason=hold_reason,
                   editor_note=note if status in ("clean", "revised") else "",
                   errata=all_errata)
    if passes and passes[-1].get("relint"):
        quality["lint"] = passes[-1]["relint"]
    final_state["quality"] = quality
    # The state log always; the memory log only for a report that publishes.
    if status == "held":
        graph._log_state(trade_date, final_state)
    else:
        graph.record_decision(ticker, trade_date, final_state)
    return final_state, rating
