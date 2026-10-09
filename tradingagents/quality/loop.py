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
import time

from tradingagents.budget import is_budget_exceeded
from tradingagents.quality import editor, errata
from tradingagents.quality.lint import Facts, lint_state

logger = logging.getLogger(__name__)

MAX_REVISIONS = 2
ANALYST_REPORT_KEYS = {"market": "market_report", "social": "sentiment_report", "news": "news_report",
                       "fundamentals": "fundamentals_report"}
VERSION = 1
# Stages whose problems can force a revision: what a reader sees and acts on.
_DECIDING = {"research_manager", "portfolio_manager", "digest"}

# Reader-facing hold reasons for a review that broke down (the server shows
# hold_reason to the run's owner).
REVISION_FAILED = "The revision couldn't be completed, so the open issues in this draft weren't fixed."
REVISION_OVER_BUDGET = ("The run reached its token budget during the revision, so the open issues in this "
                        "draft weren't fixed.")
REVIEW_NOT_APPLIED = "The review's corrections couldn't be applied, so the report wasn't fully checked."


def _location(finding: dict) -> dict:
    loc = finding.get("location")
    return loc if isinstance(loc, dict) else {}


def _digest_field(finding: dict) -> str | None:
    """The digest field an editor finding sits in ("headline",
    "bear_points[3].title"), from its field or a "digest.<field>" stage."""
    loc = _location(finding)
    for raw in (loc.get("field"), loc.get("stage")):
        field = editor.patch_location(raw) if isinstance(raw, str) else None
        if field and field != "digest":
            return field
    return None


def _patched(field: str | None, applied: list[dict]) -> bool:
    """True when an applied patch covers ``field``: the same path, or a patch
    to the whole item or field the finding sits in."""
    if not field:
        return False
    for a in applied:
        path = editor.patch_location(a.get("field"))
        if path and (field == path or field.startswith(path + ".") or field.startswith(path + "[")):
            return True
    return False


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

    reviewed: tuple | None = None   # (state, rating) of the last pass the editor reviewed

    def hold_reviewed(attempt: int, exc: Exception) -> None:
        """A revision broke down (a crash, the token budget): hold the last
        reviewed draft with its open issues instead of failing a run whose
        report is complete. Nothing more is spent."""
        nonlocal final_state, rating, status, hold_reason
        logger.error("revision %d of %s failed: %s", attempt, ticker, exc, exc_info=True)
        final_state, rating = reviewed
        status = "held"
        hold_reason = REVISION_OVER_BUDGET if is_budget_exceeded(exc) else REVISION_FAILED
        passes.append({"pass": attempt, "lint": None, "editor": None,
                       "error": f"{type(exc).__name__}: {exc}"[:300], "relint": passes[-1].get("relint")})
        report("held")

    for attempt in range(max_revisions + 1):
        try:
            final_state, rating = graph.propagate(ticker, trade_date, on_progress=on_progress, callbacks=callbacks,
                                                  revision=revision, record=False, **propagate_kwargs)
        except Exception as exc:
            if reviewed is None:
                raise
            hold_reviewed(attempt, exc)
            break
        if not final_state.get("fact_sheet"):
            # No sheet, nothing to check against: publish as before the quality layer.
            status = "unchecked"
            break
        if on_progress:
            on_progress("Quality Review", {}, final_state)
        lint = (final_state.get("quality") or {}).get("lint") or lint_state(final_state)
        leads = sum(1 for f in lint.get("flags") or [] if f.get("blocking") or f.get("severity") == "load_bearing")
        report("checking", passes=attempt, leads=leads)
        counts = {"facts": 0, "calcs": 0, "last": 0.0}

        def on_step(ev: dict, counts: dict = counts) -> None:
            """Tool calls as running totals, at most one event every two
            seconds: a long review must not crowd the run's event backlog,
            and a retried attempt restarts the count (code review 2026-10-09)."""
            if ev.get("kind") == "retry":
                counts.update(facts=0, calcs=0, last=0.0)
                report("retrying")
                return
            counts["calcs" if ev.get("kind") == "calc" else "facts"] += 1
            now = time.monotonic()
            if now - counts["last"] >= 2.0:
                counts["last"] = now
                report("tool", kind=ev.get("kind"), detail=ev.get("detail"),
                       facts=counts["facts"], calcs=counts["calcs"])

        try:
            review = review_fn(editor_llm, final_state, lint, errata.render(all_errata), callbacks=callbacks,
                               on_step=on_step)
        except editor.EditorUnavailable as exc:
            logger.error("editor unavailable for %s: %s", ticker, exc)
            status, hold_reason = "held", "editor_unavailable"
            report("held")
            passes.append({"pass": attempt, "lint": lint, "editor": None, "error": str(exc)[:300]})
            break
        except Exception as exc:
            # A fatal editor error (billing, the run's token budget) on a
            # revision: the draft before it was reviewed and is complete.
            if reviewed is None:
                raise
            hold_reviewed(attempt, exc)
            break
        try:
            patched, applied = editor.apply_patch(final_state.get("report_digest") or {}, review.get("digest_patch"),
                                                  Facts(final_state.get("fact_sheet")))
            if final_state.get("report_digest") is not None:
                final_state["report_digest"] = patched
            relint = lint_state(final_state)
        except Exception as exc:  # noqa: BLE001 — a malformed review holds, it never crashes a paid run
            logger.error("applying the review for %s failed: %s", ticker, exc, exc_info=True)
            status, hold_reason = "held", REVIEW_NOT_APPLIED
            passes.append({"pass": attempt, "lint": lint, "editor": review,
                           "error": f"{type(exc).__name__}: {exc}"[:300]})
            report("held")
            break
        # The editor checks every lint lead with the tools; one it dismissed
        # stays dismissed while its text is unchanged (staging ONDS, 2026-10-09:
        # a hold on lint flags the editor had verified as correct).
        dismissed |= {editor.dismissed_key(d) for d in review.get("lint_dismissed") or []}
        open_flags = [f for f in _load_bearing(relint) if editor.dismissed_key(f) not in dismissed]
        editor_lb = [f for f in review["findings"] if f.get("severity") == "load_bearing"
                     and errata.normalize_stage(_location(f).get("stage")) in _DECIDING]
        # Only the ruling, the decision and the digest can force a revision; a
        # digest finding is closed only by an applied patch to its own field
        # (an exit-trigger tidy-up must not close a wrong headline).
        unpatched = [f for f in editor_lb if errata.normalize_stage(_location(f).get("stage")) != "digest"
                     or not _patched(_digest_field(f), applied)]
        note = review.get("editor_note") or note
        passes.append({"pass": attempt, "lint": lint, "editor": review, "patch_applied": applied,
                       "relint": relint, "open": len(open_flags), "decision_flags": review["decision_flags"]})
        forced = (getattr(graph, "config", None) or {}).get("quality_force_hold_ticker") or ""
        if forced and forced.upper() == str(ticker).upper():
            # Staging test hook: exercise the hold and release path.
            status, hold_reason = "held", "forced hold for testing (quality_force_hold_ticker)"
            report("held")
            break
        reviewed = (final_state, rating)
        report("checked", facts=counts["facts"], calcs=counts["calcs"])
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
