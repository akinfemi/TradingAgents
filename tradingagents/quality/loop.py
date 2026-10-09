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
import re

from tradingagents.quality import editor, errata
from tradingagents.quality.lint import Facts, lint_state

logger = logging.getLogger(__name__)

MAX_REVISIONS = 2
ANALYST_REPORT_KEYS = {"market": "market_report", "social": "sentiment_report", "news": "news_report",
                       "fundamentals": "fundamentals_report"}
VERSION = 1
# Stages whose problems can force a revision: what a reader sees and acts on.
_DECIDING = re.compile(r"^(research_manager|rm|portfolio_manager|pm|digest)", re.I)


# A decision flag about the call itself re-runs the debate and ruling; one
# about levels (target, stop, exit, sizing) re-runs only the trader and PM
# (staging ONDS, 2026-10-08: a stop placement re-ran the whole debate).
_ABOUT_THE_CALL = re.compile(r"\b(rating|thesis|call|evidence|direction|bull|bear|debate|ruling)\b", re.I)


def _flag_source(flag: str) -> str:
    return "research_manager" if _ABOUT_THE_CALL.search(flag) else "trader"


def _load_bearing(lint: dict) -> list[dict]:
    return [f for f in lint.get("flags") or [] if f.get("severity") == "load_bearing"]


def run_with_quality(graph, ticker: str, trade_date, editor_llm, *, on_progress=None, callbacks=None,
                     review_fn=editor.review, **propagate_kwargs):
    """(final_state, rating). ``final_state["quality"]`` carries the verdict:
    ``{version, status: clean|revised|held, passes, hold_reason, editor_note,
    lint, gates, open_errata}``."""
    passes: list[dict] = []
    all_errata: list[dict] = []
    dismissed: set[tuple] = set()
    revision = None
    status, hold_reason, note = "held", "", ""
    final_state, rating = {}, None
    max_revisions = int((getattr(graph, "config", None) or {}).get("quality_max_revisions", MAX_REVISIONS))

    def report(step: str, **detail) -> None:
        """A review-activity event for the run page (node "Quality Review",
        delta {"_review": …}); never fails the run."""
        if on_progress:
            try:
                on_progress("Quality Review", {"_review": {"step": step, **detail}}, final_state)
            except Exception:  # noqa: BLE001 — progress is best-effort
                logger.debug("review progress event failed", exc_info=True)

    for attempt in range(max_revisions + 1):
        final_state, rating = graph.propagate(ticker, trade_date, on_progress=on_progress, callbacks=callbacks,
                                              revision=revision, record=False, **propagate_kwargs)
        if not final_state.get("fact_sheet"):
            # No sheet, nothing to check against: publish as before the quality layer.
            status = "unchecked"
            break
        if on_progress:
            on_progress("Quality Review", {}, final_state)
        lint = (final_state.get("quality") or {}).get("lint") or lint_state(final_state)
        leads = sum(1 for f in lint.get("flags") or [] if f.get("blocking") or f.get("severity") == "load_bearing")
        report("checking", passes=attempt, leads=leads)
        try:
            review = review_fn(editor_llm, final_state, lint, errata.render(all_errata), callbacks=callbacks,
                               on_step=lambda ev: report("tool", **ev))
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
        # The editor checks every lint lead with the tools; one it dismissed
        # stays dismissed while its text is unchanged (staging ONDS, 2026-10-09:
        # a hold on lint flags the editor had verified as correct).
        dismissed |= {editor.dismissed_key(d) for d in review.get("lint_dismissed") or []}
        open_flags = [f for f in _load_bearing(relint) if editor.dismissed_key(f) not in dismissed]
        editor_lb = [f for f in review["findings"] if f.get("severity") == "load_bearing"
                     and _DECIDING.match(str((f.get("location") or {}).get("stage", "")))]
        # Only the ruling, the decision and the digest can force a revision; a
        # digest finding the patch fixed is closed by it.
        unpatched = [f for f in editor_lb if not str((f.get("location") or {}).get("stage", "")).startswith("digest")
                     or not applied]
        note = review.get("editor_note") or note
        passes.append({"pass": attempt, "lint": lint, "editor": review, "patch_applied": applied,
                       "relint": relint, "open": len(open_flags), "decision_flags": review["decision_flags"]})
        forced = (getattr(graph, "config", None) or {}).get("quality_force_hold_ticker") or ""
        if forced and forced.upper() == str(ticker).upper():
            # Staging test hook: exercise the hold and release path.
            status, hold_reason = "held", "forced hold for testing (quality_force_hold_ticker)"
            break
        if applied:
            report("patched", fields=len(applied))
        if not open_flags and not review["decision_flags"] and not unpatched:
            status = "clean" if attempt == 0 else "revised"
            report("passed", status=status)
            break
        if attempt == max_revisions:
            status = "held"
            report("held")
            hold_reason = review.get("hold_reason") or (
                "Errors in figures the call relies on were still there after "
                + ("the revision." if max_revisions == 1 else f"{max_revisions} revisions."))
            break
        new = errata.merge(
            all_errata,
            errata.from_lint(open_flags),
            errata.from_editor(unpatched),
            [{"severity": "load_bearing", "kind": "decision_flag", "source": _flag_source(flag),
              "field": "decision", "quote": flag, "problem": flag,
              "correct": "fix this in the decision and restate the corrected value"} for flag in review["decision_flags"]],
        )
        all_errata = new
        rerun, first_later = errata.restart_group(all_errata)
        kept = errata.kept_outputs(final_state, rerun, first_later, ANALYST_REPORT_KEYS)
        revision = {"fact_sheet": final_state.get("fact_sheet"), "fact_sheet_text": final_state.get("fact_sheet_text"),
                    "review_errata": errata.render(all_errata), "kept": kept}
        passes[-1]["restart"] = {"analysts": sorted(rerun), "from": first_later, "kept": sorted(kept)}
        report("revising", corrections=len(all_errata), restart_from=first_later, analysts=sorted(rerun))
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
