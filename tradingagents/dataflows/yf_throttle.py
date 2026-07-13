"""Process-wide min-interval gate in front of yfinance network calls.

The hosted tiers funnel every user's data traffic through one server IP
against unofficial Yahoo endpoints with no SLA, so calls are paced instead of
burst. The gate is per-process by design: run subprocesses are
least-privilege (no Redis/DB), so a cross-process token bucket is not
reachable from here — the global rate is bounded by worker concurrency ×
this per-process rate.

Off unless TRADINGAGENTS_YF_MIN_INTERVAL (seconds) is set; the worker sets
it for itself and for run subprocesses. Thread-safe: dataflow calls run in
worker threads.
"""

from __future__ import annotations

import os
import threading
import time

_lock = threading.Lock()
_last_call = 0.0


def _min_interval() -> float:
    try:
        return float(os.environ.get("TRADINGAGENTS_YF_MIN_INTERVAL", "0"))
    except ValueError:
        return 0.0


def yf_gate() -> None:
    """Block until the configured interval since the previous gated call."""
    global _last_call
    interval = _min_interval()
    if interval <= 0:
        return
    with _lock:
        wait = _last_call + interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
