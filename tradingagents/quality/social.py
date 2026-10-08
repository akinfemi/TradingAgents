"""Sentiment sample rules (REPORT_QUALITY_PLAN R7 § Sentiment, rules 1–5).

The sentiment analyst scored ONDS from 30 StockTwits messages, half of them
from a handful of accounts and 53% with no stance, and printed the score to a
decimal. These rules, applied in code to what the sources actually returned:

1. a minimum sample — under 15 on-topic posts, 8 distinct authors, or with one
   live source, the score is "insufficient data" and confidence is low;
2. authors and events are counted, not messages (reposts are one event);
3. a claim from one author that the news doesn't carry is an "unverified
   social claim (n=1)" (prompt rule here; the linter blocks it from
   load-bearing fields);
4. the report says what the sample covers;
5. Reddit is skipped for a ticker that is also a common word unless posts are
   screened for the company.
"""

from __future__ import annotations

import re

MIN_POSTS = 15
MIN_AUTHORS = 8

# Tickers that are also everyday words: a ticker search returns posts about
# the word (plan § Sentiment). Screening (Jev) makes them usable; without it,
# Reddit is skipped for these.
AMBIGUOUS_TICKERS = {
    "A", "AI", "ALL", "ANY", "ARE", "BE", "BIG", "CAR", "CAT", "DOG", "EAT", "F", "FUN", "GO", "GOOD", "HAS",
    "HE", "IT", "KEY", "LIFE", "LOVE", "MAN", "NOW", "ON", "ONE", "OPEN", "OUT", "PLAY", "REAL", "SEE", "SO",
    "TV", "TWO", "UP", "WELL", "YOU",
}

_ST_LINE = re.compile(r"^\[(?P<date>[^\]·]+?)\s*·\s*@(?P<author>[^\s·]+)\s*·\s*(?P<tag>[^\]]+)\]\s*(?P<body>.*)$", re.M)
_RD_LINE = re.compile(r"^\s+\[(?P<date>\d{4}-\d{2}-\d{2}|\?)(?:\s*·\s*u/(?P<author>[^\]\s]+))?\]\s*(?P<body>.*)$", re.M)

SOCIAL_CLAIM_RULE = (
    "**Social claims.** A specific claim (a figure, an order, a filing, an insider move, a rating "
    "change) that comes from a single social-media author and is not in the news block is written as "
    "\"unverified social claim (n=1)\" and never presented as a fact. Report sentiment from the "
    "authors and events counted below, not from repeated messages."
)


def _norm(body: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", body.lower())[:80].strip()


def sample(stocktwits_block: str, reddit_block: str, news_block: str = "") -> dict:
    """What the social sources actually returned: posts, distinct authors,
    distinct events, live sources and the date span."""
    st = list(_ST_LINE.finditer(stocktwits_block or ""))
    rd = list(_RD_LINE.finditer(reddit_block or ""))
    authors = {m.group("author").lower() for m in st} | {
        (m.group("author") or "").lower() for m in rd if m.group("author")}
    events = {_norm(m.group("body")) for m in [*st, *rd] if m.group("body").strip()}
    dates = sorted({m.group("date").strip()[:10] for m in [*st, *rd] if m.group("date").strip()[:4].isdigit()})
    live = sum(1 for n in (len(st), len(rd)) if n)
    return {
        "posts": len(st) + len(rd),
        "stocktwits": len(st),
        "reddit": len(rd),
        "authors": len(authors),
        "events": len(events),
        "live_sources": live,
        # Company news present (articles listed, not a placeholder).
        "news": bool(re.search(r"^###\s", news_block or "", re.M)),
        "first": dates[0] if dates else None,
        "last": dates[-1] if dates else None,
    }


def coverage(s: dict) -> str:
    """Rule 4: "30 StockTwits messages and 12 Reddit posts from 22 authors (19
    distinct events), spanning 2026-10-03 to 2026-10-05"."""
    parts = []
    if s["stocktwits"]:
        parts.append(f"{s['stocktwits']} StockTwits messages")
    if s["reddit"]:
        parts.append(f"{s['reddit']} Reddit posts")
    if not parts:
        return "no social posts in the window"
    span = f", spanning {s['first']} to {s['last']}" if s.get("first") else ""
    return f"{' and '.join(parts)} from {s['authors']} authors ({s['events']} distinct events){span}"


def insufficient(s: dict) -> str | None:
    """Rule 1: why the sample can't carry a score, or None. The company's own
    news counts as a second source: Reddit rate-limits often enough (HTTP 429)
    that requiring two social feeds would leave most reports unscored."""
    reasons = []
    if s["posts"] < MIN_POSTS:
        reasons.append(f"{s['posts']} on-topic posts (minimum {MIN_POSTS})")
    if s["authors"] < MIN_AUTHORS:
        reasons.append(f"{s['authors']} distinct authors (minimum {MIN_AUTHORS})")
    if s["live_sources"] + (1 if s.get("news") else 0) <= 1:
        reasons.append("only one live source (no company news either)")
    return "insufficient data: " + "; ".join(reasons) if reasons else None


def is_ambiguous(ticker: str) -> bool:
    return ticker.upper().split(".")[0] in AMBIGUOUS_TICKERS


def apply(report: dict, s: dict) -> dict:
    """The typed sentiment report after the rules: a whole-number score, or
    none with the reason; confidence low on an insufficient sample; the
    sample's coverage stated."""
    out = dict(report or {})
    out["sample"] = s
    out["coverage"] = coverage(s)
    reason = insufficient(s)
    if reason:
        out["overall_score"] = None
        out["insufficient_reason"] = reason
        out["confidence"] = "low"
    elif isinstance(out.get("overall_score"), (int, float)):
        out["overall_score"] = round(out["overall_score"])
    return out
