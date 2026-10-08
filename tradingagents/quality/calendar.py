"""Earnings date confirmed from the company's own announcement (REPORT_QUALITY_PLAN R4b).

The fact sheet estimates the next report date from filing cadence. Companies
announce the real date two to four weeks ahead ("Ondas to Report Third
Quarter 2026 Results on November 12"); when the run's news carries that
headline, the date is extracted in code, with the article link, and the
estimate is upgraded to "confirmed". No model reads the headline.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"], start=1)}
_MONTH_RE = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
_DATE = re.compile(_MONTH_RE + r"\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(20\d\d))?", re.IGNORECASE)
# An announcement of results timing, not a story about results already out.
_ANNOUNCES = re.compile(
    r"\b(to\s+(?:report|announce|release|host|hold|present)|will\s+(?:report|announce|release|host)|"
    r"sets?\s+(?:date|conference call)|schedules?|announces?\s+(?:date|timing|conference call|upcoming)|"
    r"date\s+(?:of|for)\s+(?:its\s+)?(?:\w+\s+){0,3}(?:results|earnings))",
    re.IGNORECASE,
)
_RESULTS = re.compile(r"\b(results|earnings|financial\s+results|quarter)\b", re.IGNORECASE)
_ARTICLE = re.compile(r"^###\s+(?P<title>.+?)\s*(?:\(source:\s*(?P<source>[^)]+)\))?\s*$", re.MULTILINE)


def _month(token: str) -> int:
    token = token.lower().rstrip(".")
    return next(n for name, n in _MONTHS.items() if name.startswith(token[:3]))


def _articles(news_text: str) -> list[dict]:
    """[{title, source, link, body}] from the news tool's markdown."""
    out = []
    matches = list(_ARTICLE.finditer(news_text or ""))
    for i, m in enumerate(matches):
        body = news_text[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(news_text)]
        link = re.search(r"Link:\s*(\S+)", body)
        out.append({"title": m.group("title").strip(), "source": (m.group("source") or "").strip(),
                    "link": link.group(1) if link else None, "body": body.strip()})
    return out


def confirmed_date(news_text: str, company_names: list[str], trade_date: str,
                   estimate: str | None = None) -> dict | None:
    """The announced next earnings date, or None.

    An article counts when its title names the company, announces results
    timing, and carries a date between the trade date and 120 days after it.
    Without a year the date is the next such day on or after the trade date.
    An estimate more than 45 days away from the announced date means the
    headline is about something else (or another year): rejected."""
    as_of = date.fromisoformat(trade_date)
    names = [n.lower() for n in company_names if n and len(n) >= 3]
    for article in _articles(news_text):
        title = article["title"]
        if not any(n in title.lower() for n in names):
            continue
        if not (_ANNOUNCES.search(title) and _RESULTS.search(title)):
            continue
        m = _DATE.search(title)
        if not m:
            continue
        month, day = _month(m.group(1)), int(m.group(2))
        year = int(m.group(3)) if m.group(3) else as_of.year
        try:
            when = date(year, month, day)
        except ValueError:
            continue
        if not m.group(3) and when < as_of:
            when = date(year + 1, month, day)
        if not (as_of <= when <= as_of + timedelta(days=120)):
            continue
        if estimate and abs((when - date.fromisoformat(estimate)).days) > 45:
            continue
        return {"date": when.isoformat(), "title": title, "source": article["source"] or None,
                "link": article["link"]}
    return None
