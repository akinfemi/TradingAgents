"""The session clock (REPORT_QUALITY_PLAN R4): when the report was written
relative to the market, so no stage reads "no price reaction" into a bar
that hasn't traded yet.

ONDS was written at 06:33 ET on a Monday; the last bar was Friday's close and
the bear called Monday-morning news "ignored by the market". The clock says
which session the latest bar is, and whether news after it has traded.

US equities only (NYSE/Nasdaq hours and holidays, computed here rather than
from a calendar package); crypto trades around the clock; other markets get
the timestamp and the last bar's date without session words.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
OPEN, CLOSE = time(9, 30), time(16, 0)
PRE_OPEN, AFTER_CLOSE = time(4, 0), time(20, 0)


def _easter(year: int) -> date:
    """Gregorian Easter (anonymous algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def us_holidays(year: int) -> set[date]:
    """NYSE full-day closures for ``year``."""
    days = {
        _nth_weekday(year, 1, 0, 3),            # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),            # Washington's Birthday
        _easter(year) - timedelta(days=2),      # Good Friday
        _last_weekday(year, 5, 0),              # Memorial Day
        _observed(date(year, 7, 4)),            # Independence Day
        _nth_weekday(year, 9, 0, 1),            # Labor Day
        _nth_weekday(year, 11, 3, 4),           # Thanksgiving
        _observed(date(year, 12, 25)),          # Christmas
    }
    new_year = date(year, 1, 1)
    if new_year.weekday() != 5:                 # a Saturday New Year is not observed on Dec 31
        days.add(_observed(new_year))
    if year >= 2022:
        days.add(_observed(date(year, 6, 19)))  # Juneteenth
    return days


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in us_holidays(day.year)


def previous_trading_day(day: date) -> date:
    day -= timedelta(days=1)
    while not is_trading_day(day):
        day -= timedelta(days=1)
    return day


def us_session(started: datetime) -> dict:
    """Where ``started`` falls in the US equity day, in ET."""
    local = started.astimezone(ET)
    today, now = local.date(), local.time()
    trading = is_trading_day(today)
    if trading and OPEN <= now < CLOSE:
        state = "market open"
        last_complete = previous_trading_day(today)
    elif trading and now >= CLOSE:
        state = "after the close" if now < AFTER_CLOSE else "market closed"
        last_complete = today
    elif trading and now < OPEN:
        state = "pre-market" if now >= PRE_OPEN else "market closed"
        last_complete = previous_trading_day(today)
    else:
        state = "market closed (weekend)" if today.weekday() >= 5 else "market closed (holiday)"
        last_complete = previous_trading_day(today)
    return {"local": local, "state": state, "last_complete": last_complete, "open_now": state == "market open"}


def clock(run_started_at: str | None, asset_type: str, last_bar: str | None, us_listed: bool) -> dict:
    """The session clock as facts and one sentence for the prompts.

    ``last_bar`` is the date of the newest bar in the run's own price data —
    what the agents actually see, which can lag the calendar (vendor delay).
    """
    started = datetime.fromisoformat(run_started_at.replace("Z", "+00:00")) if run_started_at else None
    if started is not None and started.tzinfo is None:
        started = started.replace(tzinfo=ZoneInfo("UTC"))
    out: dict = {"written_at": started.isoformat() if started else None, "last_bar": last_bar}
    if started is None:
        out["text"] = (f"The latest price bar in this run's data is {last_bar}." if last_bar else "")
        return out
    if asset_type == "crypto":
        out["state"] = "24/7 market"
        out["text"] = (
            f"Report written {started.astimezone(ZoneInfo('UTC')):%a %Y-%m-%d %H:%M} UTC; crypto trades "
            f"around the clock. The latest daily bar in this run's data is {last_bar}; anything "
            "after that bar's close is not in the price data."
        )
        return out
    if not us_listed:
        out["text"] = (
            f"Report written {started.astimezone(ZoneInfo('UTC')):%a %Y-%m-%d %H:%M} UTC. The latest "
            f"price bar in this run's data is {last_bar}; news dated after that bar has not traded "
            "in the data the analysts see."
        )
        return out
    s = us_session(started)
    last_complete = s["last_complete"].isoformat()
    out.update(state=s["state"], last_complete_session=last_complete)
    text = (
        f"Report written {s['local']:%a %Y-%m-%d %H:%M} ET, {s['state']}. "
        f"Last completed session: {s['last_complete']:%a %Y-%m-%d} close."
    )
    if s["open_now"]:
        text += " Today's session is still trading; today's move is not in a completed bar."
    if last_bar and last_bar < last_complete:
        text += (f" This run's price data ends at {last_bar}, before that session; "
                 "do not describe later sessions from it.")
    text += " News dated after the last completed close has not traded in any bar the analysts see."
    out["text"] = text
    return out
