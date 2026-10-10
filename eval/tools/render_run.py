"""Render the PDF for an eval run, the way the server would (payload from the
run's final state). Usage: render_run.py <run.json> <out.pdf> [layout]"""
import json
import sys
from types import SimpleNamespace

from app.pdf import build_report_pdf, _render
from app.pdf_onepage import spilled_blocks
from app.runs import _merged_decision
from app.textclean import clamp_exit_triggers, strip_emoji, strip_fact_keys
from data.charts import display_fact_sheet

run_json, out, layout = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "onepager")
rec = json.load(open(run_json))
state = rec["state"]
sections = {"final_decision": state.get("final_trade_decision"), "research_manager": state.get("investment_plan"),
            "market": state.get("market_report"), "news": state.get("news_report"),
            "fundamentals": state.get("fundamentals_report"), "sentiment": state.get("sentiment_report")}
sheet, credits = display_fact_sheet(state.get("fact_sheet"), SimpleNamespace(display_price_sources="tiingo"))
digest = clamp_exit_triggers(strip_emoji(strip_fact_keys(state.get("report_digest") or {})))
decision = strip_fact_keys(_merged_decision(state, sections) or {})
payload = {
    "run": {"id": f"eval-{rec['ticker'].lower()}", "ticker": rec["ticker"], "trade_date": rec["trade_date"],
            "asset_type": "stock", "status": "done", "rating": rec.get("rating"),
            "finished_at": rec["trade_date"] + "T12:00:00", "llm_calls": rec.get("llm_calls")},
    "decision": decision, "digest": digest, "fact_sheet": sheet, "sections": sections, "charts": {},
    "data_attribution": credits,
    "cost": {"usage_by_model": rec.get("usage_by_model") or {}, "estimate_usd": None},
    "disclaimer": "AI-generated research for informational purposes only. Not financial advice.",
}
open(out, "wb").write(build_report_pdf(payload, layout))
print("quality:", rec.get("quality_status"), "| spilled:", spilled_blocks(_render(payload, layout).summary_blocks))
