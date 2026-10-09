"""Judge every report of an evaluation (REPORT_QUALITY_PLAN R1; grounded and
routed through OpenRouter since 2026-10-09).

    python eval/judge.py --label <label> [--judge-model anthropic/claude-opus-5.5]

The judge reads the whole report with the run's fact sheet as ground truth,
checks figures with two tools (`fact`, `calc`) and submits a structured
judgement. Every judge, whatever its family, takes the same path, so two
judges differ only in the model. Reads runs/<label>/<ticker>/run.json; writes
judgement-<model>.json beside it (reused only when the prompt, the report and
the model are unchanged) and the summary to results/<date>-<label>-<model>.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EVAL_DIR, RESULTS_DIR, RUNS_DIR, golden_set, load_env  # noqa: E402

DEFAULT_JUDGE = "anthropic/claude-opus-5.5"
JUDGE_VERSION = 2  # 2: the ruling is in the prompt; grounded; OpenRouter
CLASSES = ("arithmetic", "misread", "consistency", "context", "unsupported", "reasoning", "presentation")
MAX_TOOL_ROUNDS = 40
MAX_SCHEMA_RETRIES = 3


class Finding(BaseModel):
    error_class: Literal["arithmetic", "misread", "consistency", "context", "unsupported", "reasoning", "presentation"]
    load_bearing: bool
    location: str = Field(description="Section and field, e.g. 'digest.bear_points[0]' or 'market analyst report'.")
    quote: str = Field(description="The exact text at fault.")
    problem: str
    correction: str = Field(description="The correct value or wording, or 'n/a' if it can't be stated.")
    checked: bool = Field(description="True when a fact or calc result you received confirmed this; "
                                      "false for a judgment call or a figure you could not check.")


_SCORE = "integer from 0 to 5"


class Scores(BaseModel):
    arithmetic: int = Field(ge=0, le=5, description=_SCORE)
    misread: int = Field(ge=0, le=5, description=_SCORE)
    consistency: int = Field(ge=0, le=5, description=_SCORE)
    context: int = Field(ge=0, le=5, description=_SCORE)
    unsupported: int = Field(ge=0, le=5, description=_SCORE)
    reasoning: int = Field(ge=0, le=5, description=_SCORE)
    presentation: int = Field(ge=0, le=5, description=_SCORE)


class Judgement(BaseModel):
    scores: Scores
    findings: list[Finding] = Field(description="Every error found, most serious first.")
    verdict: Literal["publish", "publish_with_fixes", "do_not_publish"]
    summary: str = Field(description="Two sentences on the report's quality.")


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
finding `checked` only when a tool result you have received confirmed it;
judgment calls (reasoning, levels, horizon) are `checked: false`. When done,
call `submit_judgement` once, on its own, with the full judgement.
"""

_SECTIONS = (
    ("Market analyst report", lambda s: s.get("market_report")),
    ("Sentiment analyst report", lambda s: s.get("sentiment_report")),
    ("News analyst report", lambda s: s.get("news_report")),
    ("Fundamentals analyst report", lambda s: s.get("fundamentals_report")),
    ("Bull researcher", lambda s: (s.get("investment_debate_state") or {}).get("bull_history")),
    ("Bear researcher", lambda s: (s.get("investment_debate_state") or {}).get("bear_history")),
    # The ruling is investment_plan since upstream cf960d6; until 2026-10-09
    # the judge read the old key and never saw it.
    ("Research manager ruling",
     lambda s: s.get("investment_plan") or (s.get("investment_debate_state") or {}).get("judge_decision")),
    ("Trader plan", lambda s: s.get("trader_investment_plan")),
    ("Risk review: aggressive", lambda s: (s.get("risk_debate_state") or {}).get("aggressive_history")),
    ("Risk review: neutral", lambda s: (s.get("risk_debate_state") or {}).get("neutral_history")),
    ("Risk review: conservative", lambda s: (s.get("risk_debate_state") or {}).get("conservative_history")),
    ("Portfolio decision", lambda s: s.get("final_trade_decision")),
)


def build_prompt(run: dict) -> str:
    state = run["state"]
    missing = [title for title, get in _SECTIONS if not get(state)]
    if "Research manager ruling" in missing or "Portfolio decision" in missing:
        raise ValueError(f"report is missing {missing}: refusing to judge an incomplete report")
    parts = [
        (EVAL_DIR / "rubric.md").read_text(encoding="utf-8"),
        GROUNDING,
        f"\n# Report: {run['ticker']}, trade date {run['trade_date']}, rating {run['rating']}\n",
        "## Price facts computed in code (ground truth)\n",
        run.get("price_facts") or "(none available)",
        "\n## Fact sheet (computed in code from filings and prices; ground truth)\n",
        state.get("fact_sheet_text") or "(none)",
        "\n## Digest (the summary readers see first)\n",
        json.dumps(state.get("report_digest"), indent=1),
    ]
    for title, get in _SECTIONS:
        text = get(state)
        if text:
            parts.append(f"\n## {title}\n\n{text}")
    return "\n".join(parts)


def _schema(node):
    """Pydantic's schema with additionalProperties false on every object. The
    0–5 bounds are stated in the field descriptions and enforced on parse; a
    judgement that fails the parse is sent back with the error."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            node["additionalProperties"] = False
        for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            node.pop(key, None)
        for value in node.values():
            _schema(value)
    elif isinstance(node, list):
        for value in node:
            _schema(value)
    return node


def _tools() -> list[dict]:
    def tool(name, description, parameters):
        return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}

    return [
        tool("fact", "Look up one fact-sheet key, e.g. revenue.2026Q2.",
             {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}),
        tool("calc", "Evaluate an arithmetic expression (numbers, + - * / ** and parentheses).",
             {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}),
        tool("submit_judgement", "Submit the finished judgement. Call it once, last, and on its own.",
             _schema(Judgement.model_json_schema())),
    ]


def judge(client, model: str, run: dict) -> tuple[Judgement, dict]:
    """One grounded judgement through OpenRouter. Usage and OpenRouter's own
    billed cost sum over every round. A judgement that breaks the schema, or
    one submitted beside unanswered tool calls, is sent back to be redone."""
    from tradingagents.quality.editor import calc, fact
    from tradingagents.quality.lint import Facts

    facts = Facts((run.get("state") or {}).get("fact_sheet"))
    prompt = {"type": "text", "text": build_prompt(run)}
    if model.startswith("anthropic/"):
        # Explicit caching on Claude (OpenRouter passes it through); the
        # others cache long prefixes automatically.
        prompt["cache_control"] = {"type": "ephemeral"}
    messages: list[dict] = [{"role": "user", "content": [prompt]}]
    usage = {"tokens_in": 0, "tokens_out": 0, "cache_read": 0, "cache_write": 0, "cost_usd": 0.0,
             "rounds": 0, "tool_calls": 0, "schema_retries": 0}
    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=_tools(), max_tokens=32000,
            extra_body={"reasoning": {"effort": "high"}, "usage": {"include": True}})
        if not resp.choices:
            raise RuntimeError(f"no choices in the response: {getattr(resp, 'error', None) or resp}")
        u = resp.usage
        extra = getattr(u, "model_extra", None) or {}
        details = getattr(u, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        written = (getattr(details, "model_extra", None) or {}).get("cache_write_tokens", 0) or 0
        usage["tokens_in"] += u.prompt_tokens - cached - written
        usage["cache_read"] += cached
        usage["cache_write"] += written
        usage["tokens_out"] += u.completion_tokens
        usage["cost_usd"] += float(extra.get("cost") or 0)
        usage["rounds"] += 1
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError("the judge ran out of output tokens")
        msg = choice.message
        messages.append(msg.model_dump(exclude_none=True))
        calls = msg.tool_calls or []
        if not calls:
            messages.append({"role": "user", "content": "Finish by calling submit_judgement."})
            continue
        others = [c for c in calls if c.function.name != "submit_judgement"]
        for call in calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                messages.append({"role": "tool", "tool_call_id": call.id, "content": f"error: invalid JSON ({exc})"})
                continue
            if name == "submit_judgement":
                if others:
                    messages.append({"role": "tool", "tool_call_id": call.id, "content":
                                     "error: not submitted. Read the tool results first, then call "
                                     "submit_judgement on its own."})
                    continue
                try:
                    return Judgement.model_validate(args), usage
                except ValidationError as exc:
                    usage["schema_retries"] += 1
                    if usage["schema_retries"] > MAX_SCHEMA_RETRIES:
                        raise
                    messages.append({"role": "tool", "tool_call_id": call.id,
                                     "content": f"error: the judgement does not match the schema; fix and resubmit:\n{exc}"})
                    continue
            usage["tool_calls"] += 1
            out = calc(args.get("expression", "")) if name == "calc" else fact(facts, args.get("key", ""))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": out})
    raise RuntimeError("the judge did not submit a judgement")


def _slug(model: str) -> str:
    return model.split("/")[-1]


def _fingerprint(run_file: Path, model: str) -> dict:
    """What a saved judgement was made from; a mismatch means re-judge."""
    prompt_parts = (EVAL_DIR / "rubric.md").read_bytes() + GROUNDING.encode() + Path(__file__).read_bytes()
    return {"judge_version": JUDGE_VERSION, "model": model,
            "prompt_sha256": hashlib.sha256(prompt_parts).hexdigest()[:16],
            "run_sha256": hashlib.sha256(run_file.read_bytes()).hexdigest()[:16]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--env", help="env file with OPENROUTER_API_KEY")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE, help="an OpenRouter model id")
    parser.add_argument("--strict", action="store_true", help="exit 1 when any report could not be judged")
    args = parser.parse_args()
    load_env(args.env)

    from openai import OpenAI

    sys.path.insert(0, str(EVAL_DIR.parent.parent / "server"))
    from app.prices import estimate_cost_usd

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    label_dir = RUNS_DIR / args.label
    expected = [t["ticker"] for t in golden_set()["tickers"]]
    present = {p.parent.name for p in label_dir.glob("*/run.json")}
    missing_runs = [t for t in expected if t not in present]
    rows, failed = [], []
    for ticker in [t for t in expected if t in present]:
        run_file = label_dir / ticker / "run.json"
        run = json.loads(run_file.read_text(encoding="utf-8"))
        out = run_file.parent / f"judgement-{_slug(args.judge_model)}.json"
        made_from = _fingerprint(run_file, args.judge_model)
        saved = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else None
        if not saved or saved.get("made_from") != made_from:
            try:
                judgement, usage = judge(client, args.judge_model, run)
            except Exception as exc:  # noqa: BLE001 — record and continue
                failed.append(ticker)
                print(f"  {ticker}: judge failed ({type(exc).__name__}: {str(exc)[:300]})", flush=True)
                continue
            saved = {"judgement": judgement.model_dump(), "usage": usage, "made_from": made_from}
            out.write_text(json.dumps(saved, indent=1), encoding="utf-8")
        j, usage = saved["judgement"], saved["usage"]
        lb = [f for f in j["findings"] if f["load_bearing"]]
        quality = (run.get("state") or {}).get("quality") or {}
        rows.append({
            "ticker": ticker, "rating": run["rating"], "scores": j["scores"],
            "mean_score": round(sum(j["scores"].values()) / len(CLASSES), 2),
            "load_bearing_errors": len(lb), "load_bearing_checked": sum(1 for f in lb if f.get("checked")),
            "minor_errors": len(j["findings"]) - len(lb), "verdict": j["verdict"],
            "quality_status": run.get("quality_status") or quality.get("status"),
            "revisions": run.get("revisions"),
            "pipeline_cost_usd": estimate_cost_usd(run.get("usage_by_model") or {}),
            "judge_cost_usd": round(usage.get("cost_usd") or 0, 4),
            "seconds": run["seconds"], "summary": j["summary"],
        })
        print(f"  {ticker}: mean {rows[-1]['mean_score']}, {len(lb)} load-bearing, {j['verdict']}", flush=True)

    if not rows:
        print("nothing judged")
        return 1
    n = len(rows)
    costs = [r["pipeline_cost_usd"] for r in rows]
    summary = {
        "label": args.label, "date": dt.date.today().isoformat(), "judge_model": args.judge_model,
        "judge_version": JUDGE_VERSION,
        "settings": json.loads((label_dir / "settings.json").read_text(encoding="utf-8")),
        "reports": n, "tickers": [r["ticker"] for r in rows],
        "missing_runs": missing_runs, "judge_failed": failed,
        "mean_scores": {c: round(sum(r["scores"][c] for r in rows) / n, 2) for c in CLASSES},
        "mean_score": round(sum(r["mean_score"] for r in rows) / n, 2),
        "load_bearing_errors": sum(r["load_bearing_errors"] for r in rows),
        "load_bearing_checked": sum(r["load_bearing_checked"] for r in rows),
        "reports_with_load_bearing_errors": sum(1 for r in rows if r["load_bearing_errors"]),
        "verdicts": {v: sum(1 for r in rows if r["verdict"] == v) for v in ("publish", "publish_with_fixes", "do_not_publish")},
        # None when a model is unpriced: a missing price never reads as $0.
        "pipeline_cost_usd": round(sum(costs), 2) if all(c is not None for c in costs) else None,
        "judge_cost_usd": round(sum(r["judge_cost_usd"] for r in rows), 2),
        "median_seconds": sorted(r["seconds"] for r in rows)[n // 2],
        "rows": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{summary['date']}-{args.label}-{_slug(args.judge_model)}.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\n{out.relative_to(EVAL_DIR)}: {n} reports (missing runs {missing_runs}, judge failed {failed}), "
          f"mean {summary['mean_score']}/5, {summary['load_bearing_errors']} load-bearing, "
          f"verdicts {summary['verdicts']}, judge ${summary['judge_cost_usd']}")
    return 1 if (failed and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
