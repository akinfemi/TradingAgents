# Report-quality evaluation (REPORT_QUALITY_PLAN R1)

Runs the full pipeline over a fixed golden set, grades every report with a
Claude Opus 5.5 judge against the broker-review rubric (`rubric.md`), and
writes one summary per evaluation to `results/`. Every later release (R2–R7)
is compared against the baseline.

## Run

From the monorepo, using the server's environment (it has the fork, the
Anthropic SDK and the server's run-config builder):

```
cd server
uv run python ../TradingAgents/eval/run.py   --label baseline --env ../.env.staging.host
uv run python ../TradingAgents/eval/judge.py --label baseline --env ../.env.staging.host
```

- `--env` is the env file whose keys and model settings the runs use (the
  platform LLM config, `PRICE_VENDOR`, `TIINGO_API_KEY`, `FRED_API_KEY`).
  Values are read into the process only, never printed.
- `--tickers ONDS,MSFT` runs a subset; `--concurrency` (default 3) runs that
  many pipelines at once.
- Run artifacts (full state, reports, digest) go to `runs/<label>/`
  (git-ignored). The summary, scores and cost go to
  `results/<date>-<label>.json` (committed).

## How to compare two versions (noise measured 2026-10-07)

One run judged once is not enough. On the same 12 reports re-judged, a
report's mean score moved about 0.24 and its load-bearing error count about
1.7. Between two runs of the same code they moved about 0.3 and 2.2, and the
rating itself flipped on 3-4 of 12 tickers. So for every comparison:

1. **Run each version twice** (`--label X` and `--label X-2`) and judge both.
2. **Decide on load-bearing errors,** averaged over the runs. A difference
   counts when the ranges don't overlap and most tickers move the same way.
   Mean score is reported but too noisy to decide on (about ±0.1 at set
   level).
3. **Report rating stability:** how many tickers changed rating between the
   two runs of the same version.

`judge.py --judge-run 2` re-judges the same reports to measure judge noise.

## What it measures

Per report: the judge's 0–5 score on each of the seven error classes, its
list of load-bearing errors (each quoted, located and corrected), a
publish verdict, and the pipeline's cost and latency. Across the set: means,
the total of load-bearing errors, and the cost.

The judge gets the report (digest plus every agent's section) and the
price facts computed in code for the same date as ground truth. Until R4
adds EDGAR facts, fundamentals claims are judged on internal consistency
only.

## Cost

About $0.65 per pipeline run and $0.30 per judgement at current prices, so
roughly $12 for the 12-ticker set. Both scripts print the spend.
