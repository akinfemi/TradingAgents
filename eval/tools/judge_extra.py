"""judge.py for tickers outside the golden set: EXTRA_TICKERS=KO,..."""
import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    import judge

    base = judge.golden_set()
    extra = [t.strip().upper() for t in os.environ.get("EXTRA_TICKERS", "").split(",") if t.strip()]
    judge.golden_set = lambda: {**base, "tickers": [{"ticker": t, "why": "sample"} for t in extra]}
    sys.exit(judge.main())
