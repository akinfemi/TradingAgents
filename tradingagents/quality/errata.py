"""Errata (REPORT_QUALITY_PLAN R6, § Revisions): the merged, de-duplicated
lint and editor findings that a revision re-runs with, rendered at the top of
every re-run stage's prompt, and the stage a revision restarts from.
"""

from __future__ import annotations

import re

# Pipeline order of the stages a finding can come from. A revision restarts at
# the earliest stage any open load-bearing erratum names.
STAGE_ORDER = [
    "market_analyst", "sentiment_analyst", "news_analyst", "fundamentals_analyst",
    "bull_researcher", "bear_researcher", "research_manager",
    "trader",
    "risk_aggressive", "risk_conservative", "risk_neutral", "risk_debate",
    "portfolio_manager", "digest",
]
# Restart groups: a restart inside a group re-runs the whole group (a debate
# can't resume mid-way).
GROUP = {
    "market_analyst": "market", "sentiment_analyst": "social", "news_analyst": "news",
    "fundamentals_analyst": "fundamentals",
    "bull_researcher": "research", "bear_researcher": "research", "research_manager": "research",
    "trader": "trader",
    "risk_aggressive": "risk", "risk_conservative": "risk", "risk_neutral": "risk", "risk_debate": "risk",
    "portfolio_manager": "pm", "digest": "pm",
}
# What the editor and the lint write for a stage besides its STAGE_ORDER name:
# the transcript's field names (rm, pm, bull…), the group names, and words.
_STAGE_ALIASES = {
    "rm": "research_manager", "research_manager": "research_manager", "research": "research_manager",
    "ruling": "research_manager", "investment_plan": "research_manager", "research_judge": "research_manager",
    "pm": "portfolio_manager", "portfolio_manager": "portfolio_manager", "decision": "portfolio_manager",
    "final_trade_decision": "portfolio_manager",
    "market": "market_analyst", "market_report": "market_analyst", "technical_analyst": "market_analyst",
    "sentiment": "sentiment_analyst", "social": "sentiment_analyst", "social_analyst": "sentiment_analyst",
    "social_media_analyst": "sentiment_analyst", "sentiment_report": "sentiment_analyst",
    "news": "news_analyst", "news_report": "news_analyst",
    "fundamentals": "fundamentals_analyst", "fundamental_analyst": "fundamentals_analyst",
    "fundamentals_report": "fundamentals_analyst",
    "bull": "bull_researcher", "bear": "bear_researcher", "bull_analyst": "bull_researcher",
    "bear_analyst": "bear_researcher",
    "trader_investment_plan": "trader",
    "risk": "risk_debate", "risk_analysts": "risk_debate", "risk_team": "risk_debate",
    "aggressive": "risk_aggressive", "conservative": "risk_conservative", "neutral": "risk_neutral",
    "aggressive_analyst": "risk_aggressive", "conservative_analyst": "risk_conservative",
    "neutral_analyst": "risk_neutral", "report_digest": "digest",
    # Digest fields named as the stage.
    "headline": "digest", "bull_thesis": "digest", "bear_thesis": "digest", "bull_points": "digest",
    "bear_points": "digest", "exit_triggers": "digest",
}


def normalize_stage(stage) -> str:
    """A stage as its STAGE_ORDER name: case, spaces, hyphens and dots fold
    ("Research Manager", "rm" → research_manager; "PM" → portfolio_manager;
    "digest.headline" → digest). Unknown names come back folded."""
    text = re.sub(r"[\s\-]+", "_", str(stage or "").strip().lower()).strip("_")
    head = text.split(".")[0].split("[")[0].split("/")[0]
    if head in ("digest", "report_digest"):
        return "digest"
    if head in GROUP:
        return head
    if head in _STAGE_ALIASES:
        return _STAGE_ALIASES[head]
    # "portfolio_manager_decision", "rm_ruling": the stage named first.
    for name in sorted([*GROUP, *_STAGE_ALIASES], key=len, reverse=True):
        if head.startswith(name + "_"):
            return name if name in GROUP else _STAGE_ALIASES[name]
    return head


ANALYST_GROUPS = ("market", "social", "news", "fundamentals")
LATER_GROUPS = ("research", "trader", "risk", "pm")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\[F:[^\]]+\]", "", text or "")).strip().lower()[:120]


def from_lint(flags: list[dict], load_bearing_only: bool = False) -> list[dict]:
    out = []
    for f in flags:
        if not f.get("blocking") and f.get("severity") != "load_bearing":
            continue
        if load_bearing_only and f.get("severity") != "load_bearing":
            continue
        out.append({
            "severity": f.get("severity", "minor"), "kind": f.get("kind", ""), "source": f.get("stage", ""),
            "field": f.get("field", ""), "quote": f.get("quote", ""), "problem": f.get("expected") or f.get("kind"),
            "correct": ", ".join(f"[F:{k}]" for k in f.get("fact_keys") or []) or None, "origin": "lint",
        })
    return out


def from_editor(findings: list[dict]) -> list[dict]:
    out = []
    for f in findings or []:
        loc = f.get("location") if isinstance(f.get("location"), dict) else {}
        out.append({
            "severity": f.get("severity", "minor"), "kind": f.get("kind") or "editor",
            "source": normalize_stage(loc.get("stage")) if loc.get("stage") else "", "field": loc.get("field", ""),
            "quote": loc.get("quote", ""), "problem": f.get("problem", ""), "correct": f.get("correction"),
            "origin": "editor",
        })
    return out


def merge(*groups: list[dict]) -> list[dict]:
    """De-duplicated by quote; a later duplicate adds its location to "also_in".
    Ids E1… are stable within the list returned."""
    merged: list[dict] = []
    seen: dict[str, dict] = {}
    for group in groups:
        for e in group:
            key = _norm(e.get("quote", "")) or f"{e.get('kind')}:{e.get('problem')}"
            if key in seen:
                first = seen[key]
                loc = f"{e.get('source')}.{e.get('field')}"
                if loc not in first.setdefault("also_in", []) and loc != f"{first.get('source')}.{first.get('field')}":
                    first["also_in"].append(loc)
                if e.get("severity") == "load_bearing":
                    first["severity"] = "load_bearing"
                continue
            entry = {**e, "also_in": list(e.get("also_in") or [])}
            seen[key] = entry
            merged.append(entry)
    for i, e in enumerate(merged, start=1):
        e["id"] = f"E{i}"
    return merged


def render(errata: list[dict]) -> str:
    """The block shown at the top of every re-run stage."""
    if not errata:
        return ""
    lines = ["**Review errata.** The previous draft of this report failed review. Each item below is a "
             "statement that does not hold; do not repeat it, and restate any argument that depended on it "
             "with cited fact-sheet keys. Write for the reader, who never sees this list: do not mention the "
             "errata, their ids, the review or the previous draft.", ""]
    for e in errata:
        lines.append(f"[{e['id']}] {e.get('severity')} · {e.get('kind')} · source: {e.get('source')}")
        lines.append(f"  Claim:   \"{(e.get('quote') or '')[:260]}\"")
        if e.get("problem"):
            lines.append(f"  Problem: {e['problem']}")
        if e.get("correct"):
            lines.append(f"  Correct: {e['correct']}")
        if e.get("also_in"):
            lines.append(f"  Also in: {', '.join(e['also_in'][:6])}")
    return "\n".join(lines)


def restart_group(errata: list[dict]) -> tuple[set[str], str | None]:
    """(analyst groups to re-run, first later group to re-run) for the
    load-bearing errata. Analysts re-run individually; from the first later
    group on, everything re-runs. No load-bearing erratum: re-run the PM."""
    sources = {GROUP.get(normalize_stage(e.get("source")), "pm") for e in errata if e.get("severity") == "load_bearing"}
    for loc in (loc for e in errata if e.get("severity") == "load_bearing" for loc in e.get("also_in") or []):
        sources.add(GROUP.get(normalize_stage(loc), "pm"))
    analysts = {g for g in sources if g in ANALYST_GROUPS}
    later = [g for g in LATER_GROUPS if g in sources]
    first_later = "research" if analysts else (later[0] if later else "pm")
    return analysts, first_later


def kept_outputs(state: dict, rerun_analysts: set[str], first_later: str, analyst_report_keys: dict) -> dict:
    """{group: the previous pass's output} for every stage before the restart:
    a kept stage returns this instead of running."""
    kept: dict = {}
    for group, report_key in analyst_report_keys.items():
        if group not in rerun_analysts and state.get(report_key):
            kept[group] = {report_key: state[report_key]}
            # The sentiment analyst's typed block (score, sample, coverage) goes
            # with its report, or a revision drops it (staging ONDS, 2026-10-08).
            if group == "social" and state.get("sentiment_structured") is not None:
                kept[group]["sentiment_structured"] = state["sentiment_structured"]
    order = list(LATER_GROUPS)
    for group in order[: order.index(first_later)]:
        if group == "research":
            kept["research"] = {"investment_debate_state": state.get("investment_debate_state"),
                                "investment_plan": state.get("investment_plan")}
        elif group == "trader":
            kept["trader"] = {"trader_investment_plan": state.get("trader_investment_plan")}
        elif group == "risk":
            kept["risk"] = {"risk_debate_state": state.get("risk_debate_state")}
    return kept
