"""Judge every report of an evaluation with Claude Opus 5.5 (REPORT_QUALITY_PLAN R1).

    cd server && uv run python ../TradingAgents/eval/judge.py --label baseline --env ../.env.staging.host

Reads runs/<label>/<ticker>/run.json, writes each judgement next to it
(judgement.json, skipped when present) and the evaluation summary to
results/<date>-<label>.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EVAL_DIR, RESULTS_DIR, RUNS_DIR, load_env  # noqa: E402

JUDGE_MODEL = "claude-opus-5-5"
# Opus 5.5 at $4/$20 per Mtok (server/app/prices.py).
JUDGE_PRICE = (4.00, 20.00)
CLASSES = ("arithmetic", "misread", "consistency", "context", "unsupported", "reasoning", "presentation")


class Finding(BaseModel):
    error_class: Literal["arithmetic", "misread", "consistency", "context", "unsupported", "reasoning", "presentation"]
    load_bearing: bool
    location: str = Field(description="Section and field, e.g. 'digest.bear_points[0]' or 'market analyst report'.")
    quote: str = Field(description="The exact text at fault.")
    problem: str
    correction: str = Field(description="The correct value or wording, or 'n/a' if it can't be stated.")


class GroundedFinding(Finding):
    checked: bool = Field(description="True when you confirmed this with the fact and calc tools; "
                                      "false for a judgment call or a figure you could not check.")


class Scores(BaseModel):
    arithmetic: int = Field(ge=0, le=5)
    misread: int = Field(ge=0, le=5)
    consistency: int = Field(ge=0, le=5)
    context: int = Field(ge=0, le=5)
    unsupported: int = Field(ge=0, le=5)
    reasoning: int = Field(ge=0, le=5)
    presentation: int = Field(ge=0, le=5)


class Judgement(BaseModel):
    scores: Scores
    findings: list[Finding] = Field(description="Every error found, most serious first.")
    verdict: Literal["publish", "publish_with_fixes", "do_not_publish"]
    summary: str = Field(description="Two sentences on the report's quality.")


class GroundedJudgement(Judgement):
    findings: list[GroundedFinding] = Field(description="Every error found, most serious first.")


GROUNDING = """
## Ground truth and tools (this review)

You also have the run's fact sheet: every figure computed in code from the
company's SEC filings and its prices, each with a key like revenue.2026Q2.
It is ground truth for fundamentals as well as prices. Two tools:
- `fact` returns one fact-sheet key's value, unit and period;
- `calc` evaluates an arithmetic expression.

Before you report a figure as wrong, unsupported or inconsistent, check it
with the tools. A figure derived by arithmetic from fact-sheet values (shown
or reproducible with `calc`) is supported, even if it isn't itself on the
sheet. Figures from news or social posts aren't on the sheet; they are
unsupported only when the report treats them as established fact. Mark each
finding `checked` when the tools confirmed it; judgment calls (reasoning,
levels, horizon) are `checked: false`. When done, call `submit_judgement`
once with the full judgement.
"""

_SECTIONS = (
    ("Market analyst report", lambda s: s.get("market_report")),
    ("Sentiment analyst report", lambda s: s.get("sentiment_report")),
    ("News analyst report", lambda s: s.get("news_report")),
    ("Fundamentals analyst report", lambda s: s.get("fundamentals_report")),
    ("Bull researcher", lambda s: (s.get("investment_debate_state") or {}).get("bull_history")),
    ("Bear researcher", lambda s: (s.get("investment_debate_state") or {}).get("bear_history")),
    ("Research manager ruling", lambda s: (s.get("investment_debate_state") or {}).get("judge_decision")),
    ("Trader plan", lambda s: s.get("trader_investment_plan")),
    ("Risk review: aggressive", lambda s: (s.get("risk_debate_state") or {}).get("aggressive_history")),
    ("Risk review: neutral", lambda s: (s.get("risk_debate_state") or {}).get("neutral_history")),
    ("Risk review: conservative", lambda s: (s.get("risk_debate_state") or {}).get("conservative_history")),
    ("Portfolio decision", lambda s: s.get("final_trade_decision")),
)


def build_prompt(run: dict, grounded: bool = False) -> str:
    state = run["state"]
    parts = [
        (EVAL_DIR / "rubric.md").read_text(encoding="utf-8"),
        GROUNDING if grounded else "",
        f"\n# Report: {run['ticker']}, trade date {run['trade_date']}, rating {run['rating']}\n",
        "## Price facts computed in code (ground truth)\n",
        run.get("price_facts") or "(none available)",
        *(["\n## Fact sheet (computed in code from filings and prices; ground truth)\n",
           state.get("fact_sheet_text") or "(none)"] if grounded else []),
        "\n## Digest (the summary readers see first)\n",
        json.dumps(state.get("report_digest"), indent=1),
    ]
    for title, get in _SECTIONS:
        text = get(state)
        if text:
            parts.append(f"\n## {title}\n\n{text}")
    return "\n".join(parts)


def _strict_schema(node):
    """Pydantic's schema made acceptable to the API's structured output:
    additionalProperties false on every object, no numeric bounds."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            node["additionalProperties"] = False
        # Numeric bounds aren't accepted in the schema; the response is
        # validated against the Pydantic model (0-5 scores) afterwards.
        for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            node.pop(key, None)
        for value in node.values():
            _strict_schema(value)
    elif isinstance(node, list):
        for value in node:
            _strict_schema(value)
    return node


JUDGEMENT_SCHEMA = _strict_schema(Judgement.model_json_schema())


def judge_one(client, run: dict) -> tuple[Judgement, dict]:
    """One judgement, streamed (long input and output). Refusals and
    unparseable output raise, so the caller records the failure."""
    with client.messages.stream(
        model=JUDGE_MODEL,
        max_tokens=32000,
        output_config={
            "effort": "high",
            "format": {"type": "json_schema", "schema": JUDGEMENT_SCHEMA},
        },
        messages=[{"role": "user", "content": build_prompt(run)}],
    ) as stream:
        message = stream.get_final_message()
    if message.stop_reason == "refusal":
        raise RuntimeError(f"judge refused: {getattr(message, 'stop_details', None)}")
    text = next(b.text for b in message.content if b.type == "text")
    usage = {"tokens_in": message.usage.input_tokens, "tokens_out": message.usage.output_tokens}
    return Judgement.model_validate_json(text), usage


MAX_TOOL_ROUNDS = 40


def judge_grounded(client, run: dict) -> tuple[Judgement, dict]:
    """The judgement with the fact sheet and the fact/calc tools; it ends by
    calling submit_judgement. Usage sums every round (cache reads included)."""
    from tradingagents.quality.editor import calc, fact
    from tradingagents.quality.lint import Facts

    facts = Facts((run.get("state") or {}).get("fact_sheet"))
    tools = [
        {"name": "fact", "description": "Look up one fact-sheet key, e.g. revenue.2026Q2.",
         "input_schema": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}},
        {"name": "calc", "description": "Evaluate an arithmetic expression (numbers, + - * / ** and parentheses).",
         "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}},
        {"name": "submit_judgement", "description": "Submit the finished judgement. Call exactly once, last.",
         "input_schema": _strict_schema(GroundedJudgement.model_json_schema())},
    ]
    messages = [{"role": "user", "content": [
        {"type": "text", "text": build_prompt(run, grounded=True), "cache_control": {"type": "ephemeral"}}]}]
    usage = {"tokens_in": 0, "tokens_out": 0, "cache_read": 0, "cache_write": 0, "rounds": 0, "tool_calls": 0}
    for _ in range(MAX_TOOL_ROUNDS):
        with client.messages.stream(model=JUDGE_MODEL, max_tokens=32000, output_config={"effort": "high"},
                                    tools=tools, messages=messages) as stream:
            message = stream.get_final_message()
        u = message.usage
        usage["tokens_in"] += u.input_tokens
        usage["tokens_out"] += u.output_tokens
        usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
        usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
        usage["rounds"] += 1
        if message.stop_reason == "refusal":
            raise RuntimeError("judge refused")
        messages.append({"role": "assistant", "content": message.content})
        calls = [b for b in message.content if b.type == "tool_use"]
        if not calls:
            messages.append({"role": "user", "content": "Finish by calling submit_judgement."})
            continue
        results = []
        for call in calls:
            if call.name == "submit_judgement":
                return GroundedJudgement.model_validate(call.input), usage
            usage["tool_calls"] += 1
            out = calc(call.input.get("expression", "")) if call.name == "calc" else fact(facts, call.input.get("key", ""))
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": out})
        messages.append({"role": "user", "content": results})
    raise RuntimeError("the judge did not submit a judgement")


def _judge_cost(usage: dict) -> float:
    """Input at list price, cache writes at 1.25x, cache reads at 0.1x."""
    return (usage["tokens_in"] * JUDGE_PRICE[0] + usage.get("cache_write", 0) * JUDGE_PRICE[0] * 1.25
            + usage.get("cache_read", 0) * JUDGE_PRICE[0] * 0.1 + usage["tokens_out"] * JUDGE_PRICE[1]) / 1e6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--env", help="env file with ANTHROPIC_API_KEY")
    parser.add_argument(
        "--judge-run", type=int, default=1,
        help="judge the same reports again under a new number, to measure judge noise",
    )
    parser.add_argument("--grounded", action="store_true",
                        help="give the judge the fact sheet and the fact/calc tools (judgement-grounded*.json)")
    args = parser.parse_args()
    suffix = ("-grounded" if args.grounded else "") + ("" if args.judge_run == 1 else f"-{args.judge_run}")
    load_env(args.env)

    import anthropic

    sys.path.insert(0, str(EVAL_DIR.parent.parent / "server"))
    from app.prices import estimate_cost_usd

    client = anthropic.Anthropic()
    label_dir = RUNS_DIR / args.label
    rows, judge_cost, pipeline_cost, failed = [], 0.0, 0.0, []
    for run_file in sorted(label_dir.glob("*/run.json")):
        run = json.loads(run_file.read_text(encoding="utf-8"))
        out = run_file.parent / f"judgement{suffix}.json"
        if out.is_file():
            saved = json.loads(out.read_text(encoding="utf-8"))
        else:
            try:
                judgement, usage = (judge_grounded if args.grounded else judge_one)(client, run)
            except Exception as exc:  # noqa: BLE001 — record and continue
                failed.append(run["ticker"])
                print(f"  {run['ticker']}: judge failed ({exc})", flush=True)
                continue
            saved = {"judgement": judgement.model_dump(), "usage": usage}
            out.write_text(json.dumps(saved, indent=1), encoding="utf-8")
        j, usage = saved["judgement"], saved["usage"]
        cost_j = _judge_cost(usage)
        cost_p = estimate_cost_usd(run["usage_by_model"]) or 0.0
        judge_cost += cost_j
        pipeline_cost += cost_p
        lb = sum(1 for f in j["findings"] if f["load_bearing"])
        lb_checked = sum(1 for f in j["findings"] if f["load_bearing"] and f.get("checked"))
        rows.append({
            "ticker": run["ticker"],
            "rating": run["rating"],
            "scores": j["scores"],
            "mean_score": round(sum(j["scores"].values()) / len(CLASSES), 2),
            "load_bearing_errors": lb,
            "load_bearing_checked": lb_checked,
            "minor_errors": len(j["findings"]) - lb,
            "verdict": j["verdict"],
            "pipeline_cost_usd": round(cost_p, 4),
            "seconds": run["seconds"],
            "summary": j["summary"],
        })
        print(f"  {run['ticker']}: mean {rows[-1]['mean_score']}, {lb} load-bearing, {j['verdict']}", flush=True)

    if not rows:
        print("nothing judged")
        return 1
    n = len(rows)
    summary = {
        "label": args.label,
        "judge_run": args.judge_run,
        "date": dt.date.today().isoformat(),
        "settings": json.loads((label_dir / "settings.json").read_text(encoding="utf-8")),
        "judge_model": JUDGE_MODEL,
        "reports": n,
        "mean_scores": {c: round(sum(r["scores"][c] for r in rows) / n, 2) for c in CLASSES},
        "mean_score": round(sum(r["mean_score"] for r in rows) / n, 2),
        "load_bearing_errors": sum(r["load_bearing_errors"] for r in rows),
        "load_bearing_checked": sum(r["load_bearing_checked"] for r in rows),
        "grounded": args.grounded,
        "reports_with_load_bearing_errors": sum(1 for r in rows if r["load_bearing_errors"]),
        "verdicts": {v: sum(1 for r in rows if r["verdict"] == v) for v in ("publish", "publish_with_fixes", "do_not_publish")},
        "pipeline_cost_usd": round(pipeline_cost, 2),
        "pipeline_cost_per_report_usd": round(pipeline_cost / n, 3),
        "judge_cost_usd": round(judge_cost, 2),
        "median_seconds": sorted(r["seconds"] for r in rows)[n // 2],
        "judge_failed": failed,
        "rows": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{summary['date']}-{args.label}{suffix and '-judge' + suffix}.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\n{out.relative_to(EVAL_DIR)}: mean {summary['mean_score']}/5, "
          f"{summary['load_bearing_errors']} load-bearing errors in "
          f"{summary['reports_with_load_bearing_errors']}/{n} reports, verdicts {summary['verdicts']}, "
          f"pipeline ${summary['pipeline_cost_usd']}, judge ${summary['judge_cost_usd']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
