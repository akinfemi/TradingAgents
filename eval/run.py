"""Run the pipeline over the golden set (REPORT_QUALITY_PLAN R1).

    cd server && uv run python ../TradingAgents/eval/run.py --label baseline --env ../.env.staging.host

Each ticker's full final state, reports, digest, price facts, usage and timing
are written to runs/<label>/<ticker>/. A ticker that already has a result is
skipped, so an interrupted evaluation resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS_DIR, golden_set, load_env, platform_config  # noqa: E402


def run_one(ticker: str, trade_date: str, out_dir: Path, base_config: dict) -> dict:
    import common  # noqa: F401  (puts server/ on sys.path in the child process)

    sys.path.insert(0, str(common.SERVER_DIR))
    from cli.stats_handler import StatsCallbackHandler
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.quality.technicals import computed_context

    out_dir.mkdir(parents=True, exist_ok=True)
    config = dict(base_config)
    config["results_dir"] = str(out_dir / "results")
    config["data_cache_dir"] = str(out_dir / "cache")
    stats = StatsCallbackHandler()
    started = time.monotonic()
    graph = TradingAgentsGraph(selected_analysts=["market", "social", "news", "fundamentals"], config=config)
    final_state, rating = graph.propagate(ticker, trade_date, callbacks=[stats])
    elapsed = time.monotonic() - started
    graph.save_reports(final_state, ticker, save_path=out_dir / "reports", html=False)
    record = {
        "ticker": ticker,
        "trade_date": trade_date,
        "rating": rating,
        "seconds": round(elapsed),
        "price_facts": computed_context(ticker, trade_date),
        "usage_by_model": stats.usage_by_model,
        "tokens_in": stats.tokens_in,
        "tokens_out": stats.tokens_out,
        "llm_calls": stats.llm_calls,
        "state": final_state,
    }
    (out_dir / "run.json").write_text(json.dumps(record, default=str, indent=1), encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--env", help="env file with keys and model settings")
    parser.add_argument("--tickers", help="comma-separated subset of the golden set")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    settings = load_env(args.env)
    gs = golden_set()
    wanted = {t.strip().upper() for t in args.tickers.split(",")} if args.tickers else None
    tickers = [t["ticker"] for t in gs["tickers"] if not wanted or t["ticker"] in wanted]
    base = platform_config()
    label_dir = RUNS_DIR / args.label
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / "settings.json").write_text(
        json.dumps({"env": settings, "config": {k: base[k] for k in (
            "llm_provider", "deep_think_llm", "quick_think_llm", "price_vendor",
            "max_debate_rounds", "max_risk_discuss_rounds")}, "trade_date": gs["trade_date"]},
            indent=1),
        encoding="utf-8",
    )
    todo = [t for t in tickers if not (label_dir / t / "run.json").is_file()]
    print(f"{args.label}: {len(todo)} to run, {len(tickers) - len(todo)} already done", flush=True)

    from app.prices import estimate_cost_usd  # server/ is on sys.path via common

    failures = 0
    # One process per pipeline, as in production: the fork keeps its data
    # config (price vendor, cache dir) in a module global, which threads in
    # one process would share.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, args.concurrency), mp_context=ctx) as pool:
        futures = {
            pool.submit(run_one, t, gs["trade_date"], label_dir / t, base): t for t in todo
        }
        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                rec = fut.result()
                cost = estimate_cost_usd(rec["usage_by_model"])
                print(f"  {ticker}: {rec['rating']} in {rec['seconds']}s, ${cost}", flush=True)
            except Exception:  # noqa: BLE001 — one ticker must not stop the set
                failures += 1
                (label_dir / ticker).mkdir(parents=True, exist_ok=True)
                (label_dir / ticker / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                print(f"  {ticker}: FAILED (see runs/{args.label}/{ticker}/error.txt)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
