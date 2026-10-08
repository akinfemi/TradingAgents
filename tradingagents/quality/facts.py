"""The fact sheet (REPORT_QUALITY_PLAN R4, Layer 1).

Every figure a report may use, computed in code before any agent runs, each
with a key, a unit, a period, a source and — when derived — how. Every stage
gets the whole sheet and cites figures by key (``[F:revenue.2026Q2]``); a
figure that isn't here must be derived in the text from cited keys, or not
used. The ONDS report invented a "$179.8M SG&A" from two Yahoo fields and a
"bearish MACD cross" that was bullish; neither survives a sheet that states
G&A, S&M and R&D as filed and the MACD cross with its date.

The builder never raises: a missing source becomes a note in
``unavailable`` and the run goes on with what the sheet has.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, date, datetime, timedelta
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from tradingagents.quality import edgar_ext
from tradingagents.quality.session import clock

logger = logging.getLogger(__name__)

VERSION = 1

Unit = Literal["usd", "usd_per_share", "shares", "pct", "ratio", "x", "days", "date", "text", "quarters"]
Source = Literal["sec_xbrl", "sec_text", "computed", "price", "news"]


class Fact(BaseModel):
    key: str
    value: float | str | None
    unit: Unit
    period: str | None = None
    concept: str
    source: Source
    derivation: str | None = None
    filed: str | None = None


class QuarterCol(BaseModel):
    end: str
    calendar: str          # "2026Q2"
    fiscal: str | None     # "Q2 FY2027" when it differs from the calendar label


class FactSheet(BaseModel):
    version: int = VERSION
    ticker: str
    asset_type: str = "stock"
    trade_date: str
    built_at: str
    identity: dict = Field(default_factory=dict)
    session: dict = Field(default_factory=dict)
    quarters: list[QuarterCol] = Field(default_factory=list)
    years: list[QuarterCol] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    calendar: dict = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)
    unavailable: list[str] = Field(default_factory=list)

    def get(self, key: str) -> Fact | None:
        return next((f for f in self.facts if f.key == key), None)

    def value(self, key: str):
        fact = self.get(key)
        return None if fact is None else fact.value


# ---- price and technicals --------------------------------------------------------


def _price_facts(symbol: str, trade_date: str, frame: pd.DataFrame | None = None) -> tuple[list[Fact], str | None]:
    """Indicators and states from the run's own OHLCV (the frame the market
    analyst's tools read). Returns (facts, last bar date)."""
    from stockstats import wrap

    if frame is None:
        from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv

        frame = load_ohlcv(symbol, trade_date)
    if frame is None or frame.empty or len(frame) < 2:
        return [], None
    frame = frame.reset_index(drop=True)
    df = wrap(frame.copy())
    dates = pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d").tolist()
    asof = dates[-1]
    close = float(frame["Close"].iloc[-1])

    try:
        from tradingagents.quality.technicals import _price_source

        source_name = _price_source(symbol)
    except Exception:  # noqa: BLE001
        source_name = "price vendor"

    facts: list[Fact] = []

    def add(key, value, unit, concept, derivation=None, period=asof):
        facts.append(Fact(key=key, value=value, unit=unit, period=period, concept=concept,
                          source="price" if derivation is None else "computed", derivation=derivation))

    add("price.close", round(close, 4), "usd", "close")
    facts.append(Fact(key="price.source", value=source_name, unit="text", period=asof, concept="price_source",
                      source="price"))
    for col, key, window in (("close_10_ema", "ema10", 10), ("close_50_sma", "sma50", 50),
                             ("close_200_sma", "sma200", 200)):
        if len(frame) < window:
            continue
        avg = float(df[col].iloc[-1])
        add(f"{key}.value", round(avg, 4), "usd", key, f"{window}-day average of closes to {asof}")
        add(f"{key}.gap_usd", round(close - avg, 4), "usd", f"{key}_gap", f"close − {key}")
        add(f"{key}.gap_pct", round((close / avg - 1) * 100, 2), "pct", f"{key}_gap", f"close ÷ {key} − 1")

    if len(frame) >= 35:
        macd = df["macd"].astype(float).tolist()
        signal = df["macds"].astype(float).tolist()
        add("macd.value", round(macd[-1], 4), "ratio", "macd", "EMA12 − EMA26 of closes")
        add("macd.signal", round(signal[-1], 4), "ratio", "macd_signal", "9-day EMA of MACD")
        add("macd.hist", round(macd[-1] - signal[-1], 4), "ratio", "macd_hist", "MACD − signal")
        add("macd.position", "above signal" if macd[-1] > signal[-1] else "below signal", "text", "macd_position",
            "sign of MACD − signal")
        for i in range(len(macd) - 1, 0, -1):
            prev, curr = macd[i - 1] - signal[i - 1], macd[i] - signal[i]
            if (prev > 0) != (curr > 0) and curr != 0:
                add("macd.cross", "bullish" if curr > 0 else "bearish", "text", "macd_cross",
                    "last sign change of MACD − signal")
                add("macd.cross_date", dates[i], "date", "macd_cross_date", "session of the last sign change")
                break
    if len(frame) >= 15:
        add("rsi14.value", round(float(df["rsi_14"].iloc[-1]), 1), "ratio", "rsi14", "Wilder RSI, 14 sessions")
    if len(frame) >= 20:
        mid, upper, lower = (float(df[c].iloc[-1]) for c in ("boll", "boll_ub", "boll_lb"))
        add("boll.upper", round(upper, 4), "usd", "boll_upper", "20-day SMA + 2σ")
        add("boll.lower", round(lower, 4), "usd", "boll_lower", "20-day SMA − 2σ")
        if upper > lower:
            add("boll.pct_b", round((close - lower) / (upper - lower), 3), "ratio", "boll_pct_b",
                "(close − lower) ÷ (upper − lower)")
        add("volume.avg20", round(float(frame["Volume"].tail(20).mean())), "shares", "avg_volume_20d",
            "mean of the last 20 sessions' volume")
    if len(frame) >= 15:
        atr = float(df["atr_14"].iloc[-1])
        add("atr14.usd", round(atr, 4), "usd", "atr14", "14-session average true range")
        add("atr14.pct", round(atr / close * 100, 2), "pct", "atr14_pct", "ATR ÷ close")

    year = frame.tail(252)
    year_dates = dates[-len(year):]
    closes = year["Close"].astype(float).tolist()
    hi, lo = max(closes), min(closes)
    span = "52w" if len(year) >= 252 else f"{len(year)}d"
    add(f"{span}.high", round(hi, 4), "usd", "high_52w", f"highest close of the last {len(year)} sessions")
    add(f"{span}.high_date", year_dates[closes.index(hi)], "date", "high_52w_date", "session of that close")
    add(f"{span}.from_high_pct", round((close / hi - 1) * 100, 2), "pct", "from_high", "close ÷ high − 1")
    add(f"{span}.low", round(lo, 4), "usd", "low_52w", f"lowest close of the last {len(year)} sessions")
    add(f"{span}.low_date", year_dates[closes.index(lo)], "date", "low_52w_date", "session of that close")
    add(f"{span}.from_low_pct", round((close / lo - 1) * 100, 2), "pct", "from_low", "close ÷ low − 1")
    return facts, asof


# ---- statements --------------------------------------------------------------------


_UNIT = {"USD": "usd", "USD/shares": "usd_per_share", "shares": "shares"}


def _q_label(end: str, ends: list[str], fy_end) -> QuarterCol:
    start = edgar_ext.quarter_start(end, ends)
    cal = edgar_ext.calendar_quarter(start, date.fromisoformat(end))
    fiscal = edgar_ext.fiscal_label(date.fromisoformat(end), fy_end)
    fiscal_text = f"Q{fiscal[1]} FY{fiscal[0]}" if fiscal else None
    # Calendar-year filers' fiscal quarters ARE calendar quarters: don't repeat them.
    if fiscal and fiscal_text == f"Q{cal[-1]} FY{cal[:4]}":
        fiscal_text = None
    return QuarterCol(end=end, calendar=cal, fiscal=fiscal_text)


def _y_label(end: str, fy_end) -> QuarterCol:
    fiscal = edgar_ext.fiscal_label(date.fromisoformat(end), fy_end)
    fy = fiscal[0] if fiscal else date.fromisoformat(end).year
    return QuarterCol(end=end, calendar=f"FY{fy}", fiscal=None)


def _statement_facts(st: edgar_ext.Statements, n_show: int) -> tuple[list[Fact], list[QuarterCol], list[QuarterCol], dict]:
    """(facts, shown quarter columns, year columns, {concept: {end: value}} for derivations)."""
    cols_all = [_q_label(e, st.quarter_ends, st.fy_end) for e in st.quarter_ends]
    shown = cols_all[-n_show:]
    years = [_y_label(e, st.fy_end) for e in st.year_ends]
    facts: list[Fact] = []
    values: dict[str, dict[str, float]] = {}
    for concept, kind, _tags in edgar_ext.LINES:
        values[concept] = {e: v.value for e, v in st.quarters.get(concept, {}).items()}
        for col in shown:
            v = st.quarters.get(concept, {}).get(col.end)
            if v is None:
                continue
            # One key grammar for every row: <item>.<calendar quarter>. Balances
            # keep their exact date in ``period`` (staging: agents wrote
            # cash.2026Q2 for a key that was cash.2026-06-30).
            key = f"{concept}.{col.calendar}"
            period = (f"at {col.end}" if kind == "stock"
                      else col.calendar + (f" ({col.fiscal})" if col.fiscal else ""))
            facts.append(Fact(key=key, value=v.value, unit=_UNIT.get(v.unit, "usd"), period=period,
                              concept=v.tag, source="computed" if v.derivation else "sec_xbrl",
                              derivation=v.derivation, filed=f"{v.filed} {v.accn}".strip()))
        for col in years:
            v = st.years.get(concept, {}).get(col.end)
            if v is None:
                continue
            key = f"{concept}.{col.calendar}"
            if kind == "stock" and any(f.key == key for f in facts):
                continue
            facts.append(Fact(key=key, value=v.value, unit=_UNIT.get(v.unit, "usd"),
                              period=col.calendar if kind == "flow" else col.end, concept=v.tag,
                              source="sec_xbrl", filed=f"{v.filed} {v.accn}".strip()))
    return facts, shown, years, values


def _derived(st: edgar_ext.Statements, cols: list[QuarterCol], values: dict, close: float | None,
             price_date: str | None) -> tuple[list[Fact], list[str]]:
    facts: list[Fact] = []
    flags: list[str] = []
    ends = st.quarter_ends

    def add(key, value, unit, concept, derivation, period):
        if value is None:
            return
        facts.append(Fact(key=key, value=round(value, 6) if isinstance(value, float) else value, unit=unit,
                          period=period, concept=concept, source="computed", derivation=derivation))

    def v(concept, end):
        return values.get(concept, {}).get(end)

    def year_ago(end):
        target = date.fromisoformat(end) - timedelta(days=365)
        near = [e for e in ends if abs((date.fromisoformat(e) - target).days) <= 10]
        return near[0] if near else None

    def ttm(concept, end):
        i = ends.index(end)
        window = ends[i - 3:i + 1] if i >= 3 else []
        vals = [v(concept, e) for e in window]
        return sum(vals) if len(window) == 4 and all(x is not None for x in vals) else None

    for col in cols:
        end, cal = col.end, col.calendar
        period = cal + (f" ({col.fiscal})" if col.fiscal else "")
        rev, gp, oi = v("revenue", end), v("gross_profit", end), v("operating_income", end)
        if rev:
            add(f"gross_margin.{cal}", gp / rev * 100 if gp is not None else None, "pct", "gross_margin",
                "gross profit ÷ revenue", period)
            add(f"op_margin.{cal}", oi / rev * 100 if oi is not None else None, "pct", "operating_margin",
                "operating income ÷ revenue", period)
        opex = v("opex", end)
        if opex is not None and gp:
            add(f"opex_gp.{cal}", opex / gp, "x", "opex_to_gross_profit", "operating expenses ÷ gross profit", period)
        sbc, amort = v("sbc", end), v("amortization", end)
        if oi is not None and sbc is not None:
            add(f"op_income_ex_sbc_amort.{cal}", oi + sbc + (amort or 0), "usd", "operating_income_before_sbc_amortization",
                "operating income + stock comp" + (" + amortization" if amort is not None else ""), period)
        ocf, capex = v("ocf", end), v("capex", end)
        if ocf is not None and capex is not None:
            add(f"fcf.{cal}", ocf - capex, "usd", "free_cash_flow", "operating cash flow − capex", period)
        cash, sti = v("cash", end), v("sti", end)
        # Investments tagged in an earlier quarter but not this one is a tag
        # gap, not money gone: no cash + investments total for this quarter.
        sti_gap = sti is None and any(v("sti", c.end) is not None for c in cols if c.end < end)
        if cash is not None and not sti_gap:
            add(f"cash_sti.{cal}", cash + (sti or 0), "usd", "cash_and_short_term_investments",
                "cash + short-term investments" + ("" if sti is not None else " (none tagged)"), f"at {end}")
        prior = year_ago(end)
        if prior:
            for concept, key in (("revenue", "revenue_yoy"), ("shares_weighted", "shares_yoy")):
                now, then = v(concept, end), v(concept, prior)
                if now is not None and then:
                    add(f"{key}.{cal}", (now / then - 1) * 100, "pct", key,
                        f"{concept} this quarter ÷ same quarter a year earlier ({prior}) − 1", period)

    if not cols:
        return facts, flags
    last = cols[-1]
    end, cal = last.end, last.calendar
    for concept in ("revenue", "net_income", "operating_income", "ocf", "capex", "gross_profit"):
        total = ttm(concept, end)
        add(f"ttm_{concept}.{cal}", total, "usd", f"ttm_{concept}", f"sum of the four quarters to {end}",
            f"TTM:{cal}")
    if ttm("ocf", end) is not None and ttm("capex", end) is not None:
        add(f"ttm_fcf.{cal}", ttm("ocf", end) - ttm("capex", end), "usd", "ttm_free_cash_flow",
            "TTM operating cash flow − TTM capex", f"TTM:{cal}")
    prior = year_ago(end)
    if prior and ttm("revenue", end) and ttm("revenue", prior):
        add(f"ttm_revenue_yoy.{cal}", (ttm("revenue", end) / ttm("revenue", prior) - 1) * 100, "pct",
            "ttm_revenue_yoy", f"TTM revenue to {end} ÷ TTM revenue to {prior} − 1", f"TTM:{cal}")

    cash_sti = (v("cash", end) or 0) + (v("sti", end) or 0) if v("cash", end) is not None else None
    fcf_q = (v("ocf", end) - v("capex", end)) if v("ocf", end) is not None and v("capex", end) is not None else None
    if cash_sti is not None and fcf_q is not None and fcf_q < 0:
        add(f"runway_quarters.{cal}", cash_sti / -fcf_q, "quarters", "cash_runway",
            f"cash + short-term investments at {end} ÷ that quarter's cash burn (−FCF)", f"{cal}")

    shares = st.cover_shares.value if st.cover_shares else None
    if shares is not None:
        facts.append(Fact(key="shares.cover", value=shares, unit="shares", period=st.cover_shares_date,
                          concept="dei:EntityCommonStockSharesOutstanding", source="sec_xbrl",
                          filed=f"{st.cover_shares.filed} {st.cover_shares.accn}"))
    if shares and close:
        mcap = shares * close
        add("market_cap", mcap, "usd", "market_cap",
            f"cover-page shares ({st.cover_shares_date}) × close ({price_date})", price_date)
        debt = (v("debt", end) or 0) + (v("debt_current", end) or 0)
        debt_note = "" if v("debt", end) is not None or v("debt_current", end) is not None else \
            f"; no debt tagged at {end}, counted as 0"
        if cash_sti is not None:
            ev = mcap - cash_sti + debt
            add("ev", ev, "usd", "enterprise_value",
                f"market cap − cash − short-term investments + debt at {end}{debt_note}; "
                "derivative liabilities shown separately, not in EV", price_date)
            ttm_rev = ttm("revenue", end)
            if ttm_rev:
                add("ev_sales.ttm", ev / ttm_rev, "x", "ev_to_sales_ttm", f"EV ÷ TTM revenue to {end}", price_date)
            if v("revenue", end):
                add("ev_sales.run_rate", ev / (v("revenue", end) * 4), "x", "ev_to_sales_run_rate",
                    f"EV ÷ ({cal} revenue × 4)", price_date)

    # ---- flags: facts the stages must address --------------------------------
    oi, ni, nonop = v("operating_income", end), v("net_income", end), v("non_operating", end)
    if oi is not None and ni is not None and (oi < 0) != (ni < 0):
        flags.append(
            f"In {cal} operating income ({_m(oi)}) and net income ({_m(ni)}) have opposite signs: "
            f"non-operating items ({_m(nonop) if nonop is not None else 'see the statements'}) drive net income. "
            "Do not describe net income as operating performance."
        )
    ttm_oi, ttm_ni = ttm("operating_income", end), ttm("net_income", end)
    if ttm_oi is not None and ttm_ni is not None and (ttm_oi < 0) != (ttm_ni < 0):
        flags.append(
            f"TTM operating income ({_m(ttm_oi)}) and TTM net income ({_m(ttm_ni)}) have opposite signs: "
            "non-operating items drive the TTM profit or loss."
        )
    prev = edgar_ext._prev_quarter_end(end, ends)
    if prev and v("cash", end) is not None and v("cash", prev) is not None:
        cash_drop = v("cash", end) - v("cash", prev)
        both_now = (v("cash", end) or 0) + (v("sti", end) or 0)
        both_prev = (v("cash", prev) or 0) + (v("sti", prev) or 0)
        if cash_drop < 0 and (both_now - both_prev) > cash_drop * 0.5:
            flags.append(
                f"Cash fell {_m(cash_drop)} in {cal} but cash + short-term investments changed only "
                f"{_m(both_now - both_prev)}: money moved into investments. Use cash + short-term "
                "investments for liquidity and runway, and operating cash flow for burn."
            )
    if prior and v("shares_weighted", end) and v("shares_weighted", prior):
        growth = v("shares_weighted", end) / v("shares_weighted", prior) - 1
        if growth > 0.25:
            flags.append(
                f"Weighted shares grew {growth * 100:.0f}% year on year to {cal}: discuss dilution with the "
                "share bridge (stock sold for cash, stock issued for acquisitions)."
            )
    return facts, flags


def _m(x: float | None) -> str:
    if x is None:
        return "n/a"
    sign = "−" if x < 0 else ""
    a = abs(x)
    return f"{sign}${a / 1e9:,.2f}B" if a >= 1e9 else f"{sign}${a / 1e6:,.1f}M"


# ---- calendar and identity ----------------------------------------------------------


def _next_earnings(st: edgar_ext.Statements, cols: list[QuarterCol], as_of: str) -> dict:
    """Estimated from filing cadence: the date last year's filing for the same
    fiscal quarter came out, a year on. Confirmed later from news."""
    if not cols:
        return {}
    last_end = date.fromisoformat(cols[-1].end)
    next_end = last_end + timedelta(days=91)
    rows = edgar_ext.filings(st.submissions)
    year_ago = [r for r in rows
                if abs((date.fromisoformat(r["period"]) - (next_end - timedelta(days=365))).days) <= 12]
    fiscal = edgar_ext.fiscal_label(next_end, st.fy_end)
    start = last_end + timedelta(days=1)
    cal = edgar_ext.calendar_quarter(start, next_end)
    fiscal_text = f"Q{fiscal[1]} FY{fiscal[0]}" if fiscal else None
    covers = cal + (f" ({fiscal_text})" if fiscal_text and fiscal_text != f"Q{cal[-1]} FY{cal[:4]}" else "")
    if year_ago:
        filed = date.fromisoformat(year_ago[0]["filed"]) + timedelta(days=364)
        basis = (f"last year's {year_ago[0]['form']} for the same quarter was filed "
                 f"{year_ago[0]['filed']}; results are usually released on or a few days before the filing")
    else:
        filed = next_end + timedelta(days=45)
        basis = "no filing a year earlier for the same quarter; the 10-Q deadline is 40–45 days after quarter end"
    out = {"covers": covers, "period_end_approx": next_end.isoformat(), "date": filed.isoformat(),
           "status": "estimated", "basis": basis}
    if filed.isoformat() < as_of:
        out["note"] = "the estimated date has passed without a filing on record; the report may be imminent"
    reported = [f"{c.calendar}" + (f" ({c.fiscal})" if c.fiscal else "") for c in cols]
    out["already_reported"] = reported
    return out


def _identity(st: edgar_ext.Statements | None) -> dict:
    if st is None:
        return {}
    s = st.submissions
    fy = st.fy_end
    return {
        "name": s.get("name"),
        "cik": st.cik,
        "sic": s.get("sic"),
        "sic_description": s.get("sicDescription"),
        "exchanges": s.get("exchanges"),
        "fiscal_year_end": f"{date(2000, fy[0], 1):%B} {fy[1]}" if fy else None,
        "filer_category": re.sub(r"<[^>]+>", ", ", s.get("category") or "") or None,
    }


_DESCRIBE_PROMPT = (
    "Below is the opening of {name}'s annual report (Form 10-K, Item 1, filed {filed}). In two or "
    "three plain sentences, say what the company does: its products or services, its customers "
    "and markets, and its segments if named. Use only this text. No figures unless the text states "
    "them, no opinions, no forward-looking claims, no marketing adjectives.\n\n---\n{excerpt}\n---"
)


def _describe_business(sheet: FactSheet, st: edgar_ext.Statements, trade_date: str, describe) -> None:
    """identity.description from the latest 10-K's Item 1, summarised once
    per filing (cached by accession). Never raises."""
    import json as _json
    from pathlib import Path

    from tradingagents.dataflows.config import get_config

    try:
        filing = edgar_ext.latest_annual_report(st.submissions, trade_date)
        if filing is None:
            return
        cache = Path(get_config()["data_cache_dir"]) / "sec_edgar" / f"description-{filing['accn']}.json"
        if cache.exists():
            text = _json.loads(cache.read_text(encoding="utf-8")).get("description")
        else:
            excerpt = edgar_ext.fetch_item1(st.cik, filing)
            if not excerpt:
                return
            text = str(describe(_DESCRIBE_PROMPT.format(
                name=sheet.identity.get("name") or sheet.ticker, filed=filing["filed"], excerpt=excerpt))).strip()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(_json.dumps({"description": text, "filing": filing}), encoding="utf-8")
        if text:
            sheet.identity.update(
                description=text,
                description_source=f"10-K filed {filing['filed']} ({filing['accn']}), Item 1, summarised",
            )
    except Exception as exc:  # noqa: BLE001
        logger.info("fact sheet: business description for %s skipped: %s", sheet.ticker, exc)


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:40]


def segment_facts(rows: list[dict], cols: list[QuarterCol], years: list[QuarterCol], filed: str) -> list[Fact]:
    """Facts for segment and product-line rows: seg_<concept>.<slug>.<period>,
    with year-on-year change where the filing carries the comparative."""
    out: list[Fact] = []
    if not rows:
        return out
    # The latest period and its year-ago comparative only, at most 8 lines per
    # kind and measure, largest first: the sheet goes to every stage.
    latest = max(r["end"] for r in rows)
    span = next(r["span"] for r in rows if r["end"] == latest)
    keep_ends = {latest} | {r["end"] for r in rows if r["span"] == span
                            and abs((date.fromisoformat(latest) - date.fromisoformat(r["end"])).days - 364) <= 10}
    rows = [r for r in rows if r["end"] in keep_ends and r["span"] == span]
    top: dict[tuple, set] = {}
    for r in sorted(rows, key=lambda r: -abs(r["value"]) if r["end"] == latest else 0):
        if r["end"] == latest:
            names = top.setdefault((r["axis"], r["concept"]), set())
            if len(names) < 8:
                names.add(r["label"])
    rows = [r for r in rows if r["label"] in top.get((r["axis"], r["concept"]), set())]
    rows.sort(key=lambda r: (r["axis"] != "segment", r["concept"], r["end"] != latest, -abs(r["value"])))
    by_end = {c.end: c for c in cols}
    by_year = {y.end: y for y in years}
    values: dict[tuple, float] = {}
    for r in rows:
        col = by_end.get(r["end"]) if r["span"] == "Q" else by_year.get(r["end"])
        label = col.calendar if col else (f"FY{r['end'][:4]}" if r["span"] == "FY" else None)
        if label is None:
            continue
        kind = "segment" if r["axis"] == "segment" else "product line"
        key = f"seg_{r['concept']}.{_slug(r['label'])}.{label}"
        values[(r["concept"], _slug(r["label"]), r["end"], r["span"])] = r["value"]
        out.append(Fact(key=key, value=r["value"], unit="usd", period=label, concept=f"{kind}: {r['label']}",
                        source="sec_xbrl", filed=filed))
    for r in rows:
        prior = next((e for (c, s_, e, sp) in values
                      if c == r["concept"] and s_ == _slug(r["label"]) and sp == r["span"]
                      and abs((date.fromisoformat(r["end"]) - date.fromisoformat(e)).days - 364) <= 10), None)
        col = by_end.get(r["end"]) if r["span"] == "Q" else by_year.get(r["end"])
        label = col.calendar if col else (f"FY{r['end'][:4]}" if r["span"] == "FY" else None)
        then = values.get((r["concept"], _slug(r["label"]), prior, r["span"])) if prior else None
        if label and then:
            out.append(Fact(key=f"seg_{r['concept']}_yoy.{_slug(r['label'])}.{label}",
                            value=round((r["value"] / then - 1) * 100, 2), unit="pct", period=label,
                            concept=f"{r['label']} year on year", source="computed",
                            derivation=f"this period ÷ the period ended {prior} − 1, both as filed"))
    return out


def _add_segments(sheet: FactSheet, st: edgar_ext.Statements, trade_date: str) -> None:
    """Segment and product-line figures from the latest 10-Q/10-K's
    dimensional XBRL (R4b). Never raises."""
    try:
        filing = next((r for r in edgar_ext.filings(st.submissions) if r["filed"] <= trade_date), None)
        if filing is None:
            return
        ends = list({*st.quarter_ends, *st.year_ends})
        rows = edgar_ext.fetch_segments(st.cik, filing, ends)
        facts = segment_facts(rows, sheet.quarters, sheet.years, f"{filing['filed']} {filing['accn']}")
        sheet.facts.extend(facts)
        if facts:
            sheet.identity["segments_source"] = f"{filing['form']} for {filing['period']}, filed {filing['filed']}"
    except Exception as exc:  # noqa: BLE001
        logger.info("fact sheet: segments for %s skipped: %s", sheet.ticker, exc)


def _confirm_from_news(nxt: dict, ticker: str, trade_date: str, identity: dict,
                       news_text: str | None = None) -> None:
    """Upgrade the estimated date to "confirmed" when the company's own
    announcement is in the news (R4b). Never raises."""
    from tradingagents.quality.calendar import confirmed_date

    try:
        if news_text is None:
            from tradingagents.dataflows.router import route_to_vendor

            start = (date.fromisoformat(trade_date) - timedelta(days=45)).isoformat()
            news_text = route_to_vendor("get_news", ticker, start, trade_date)
        name = (identity.get("name") or "").split()
        names = [ticker, name[0].rstrip(",.") if name else ""]
        found = confirmed_date(str(news_text), names, trade_date, nxt.get("date"))
    except Exception as exc:  # noqa: BLE001 — the estimate stands
        logger.info("fact sheet: earnings confirmation for %s skipped: %s", ticker, exc)
        return
    if found:
        nxt.update(
            date=found["date"], status="confirmed",
            basis=f"announced by the company: \"{found['title']}\""
                  + (f" ({found['source']})" if found.get("source") else ""),
            link=found.get("link"),
        )


# ---- build -------------------------------------------------------------------------


def build(ticker: str, trade_date: str, run_started_at: str | None = None, asset_type: str = "stock",
          *, statements: edgar_ext.Statements | None = None, ohlcv: pd.DataFrame | None = None,
          offline: bool = False, n_quarters: int = 5, news_text: str | None = None,
          describe=None) -> FactSheet:
    """The fact sheet for one run. Never raises.

    ``describe``: optional ``fn(excerpt) -> str`` that summarises the 10-K's
    Item 1 (the deep model, from the graph); without it there is no
    business description."""
    sheet = FactSheet(ticker=ticker.upper(), asset_type=asset_type, trade_date=trade_date,
                      built_at=datetime.now(UTC).isoformat(timespec="seconds"))
    close, last_bar = None, None
    try:
        price, last_bar = _price_facts(ticker, trade_date, ohlcv)
        sheet.facts.extend(price)
        close = next((f.value for f in price if f.key == "price.close"), None)
        if not price:
            sheet.unavailable.append("price history unavailable: no technicals")
    except Exception as exc:  # noqa: BLE001
        logger.warning("fact sheet: price facts for %s failed: %s", ticker, exc)
        sheet.unavailable.append("price history unavailable: no technicals")

    st = statements
    if st is None and asset_type != "crypto" and not offline:
        try:
            st = edgar_ext.load(ticker, trade_date, n_quarters=max(8, n_quarters))
        except Exception as exc:  # noqa: BLE001
            logger.warning("fact sheet: EDGAR for %s failed: %s", ticker, exc)
            sheet.unavailable.append(f"SEC statements unavailable ({type(exc).__name__}); no fundamentals figures")
    if st is None and not sheet.unavailable and asset_type != "crypto":
        sheet.unavailable.append("not a US SEC filer: no filed statements, so no fundamentals figures")
    if st is not None:
        try:
            facts, cols, years, values = _statement_facts(st, n_quarters)
            sheet.facts.extend(facts)
            sheet.quarters, sheet.years = cols, years
            derived, flags = _derived(st, cols, values, close, last_bar)
            sheet.facts.extend(derived)
            sheet.flags.extend(flags)
            sheet.identity = _identity(st)
            if describe is not None and not offline:
                _describe_business(sheet, st, trade_date, describe)
            if not offline:
                _add_segments(sheet, st, trade_date)
            sheet.calendar = {"earnings_next": _next_earnings(st, cols, trade_date)}
            nxt = sheet.calendar["earnings_next"]
            if nxt and (news_text is not None or not offline):
                _confirm_from_news(nxt, ticker, trade_date, sheet.identity, news_text)
            if nxt:
                sheet.facts.append(Fact(
                    key="earnings.next", value=nxt["date"], unit="date", period=nxt["covers"],
                    concept="next_earnings_date", source="computed",
                    derivation=f"{nxt['status']}: {nxt['basis']}"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("fact sheet: statements for %s failed: %s", ticker, exc, exc_info=True)
            sheet.unavailable.append("SEC statements could not be read; no fundamentals figures")

    sheet.session = clock(run_started_at, asset_type, last_bar, us_listed=st is not None)
    return sheet


# ---- render for prompts ---------------------------------------------------------------


def _fmt(f: Fact) -> str:
    v = f.value
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    if f.unit == "usd":
        a = abs(v)
        if a >= 1e9:
            return f"{'−' if v < 0 else ''}${a / 1e9:,.2f}B"
        if a >= 1e6:
            return f"{'−' if v < 0 else ''}${a / 1e6:,.1f}M"
        if a >= 1e5:
            return f"{'−' if v < 0 else ''}${a / 1e6:,.2f}M"
        return f"{'−' if v < 0 else ''}${a:,.0f}"
    if f.unit == "usd_per_share":
        return f"{'−' if v < 0 else ''}${abs(v):,.2f}"
    if f.unit == "pct":
        return f"{v:+,.1f}%"
    if f.unit == "x":
        return f"{v:,.2f}x"
    if f.unit == "shares":
        return f"{v / 1e6:,.1f}M sh" if abs(v) >= 1e6 else f"{v:,.0f} sh"
    if f.unit == "quarters":
        return f"{v:,.1f} quarters"
    return f"{v:,.4g}" if isinstance(v, float) else str(v)


_ROWS = [
    ("revenue", "Revenue"), ("revenue_yoy", "Revenue YoY"), ("gross_profit", "Gross profit"),
    ("gross_margin", "Gross margin"), ("sga", "SG&A"), ("ga", "G&A"), ("sm", "Sales & marketing"),
    ("rnd", "R&D"), ("amortization", "Amortization of intangibles"), ("opex", "Operating expenses"),
    ("opex_gp", "Opex ÷ gross profit"), ("operating_income", "Operating income"),
    ("op_margin", "Operating margin"), ("op_income_ex_sbc_amort", "Op. income before stock comp & amortization"),
    ("non_operating", "Non-operating income"),
    ("warrant_fair_value", "Warrant fair-value adjustment (cash-flow add-back: negative = gain)"),
    ("net_income", "Net income"), ("eps_diluted", "Diluted EPS"), ("sbc", "Stock-based comp"),
    ("ocf", "Operating cash flow"), ("capex", "Capex"), ("fcf", "Free cash flow"),
    ("stock_sold_cash", "Cash from stock sold"), ("stock_for_acquisitions", "Stock issued for acquisitions"),
    ("acquisitions_cash", "Cash paid for acquisitions"), ("shares_weighted", "Weighted shares (basic)"),
    ("shares_yoy", "Weighted shares YoY"),
]
_LINE_ITEMS = {concept for concept, _kind, _tags in edgar_ext.LINES}
_BALANCE_ROWS = [
    ("cash", "Cash"), ("sti", "Short-term investments"), ("cash_sti", "Cash + short-term investments"),
    ("debt", "Long-term debt"), ("debt_current", "Current debt"), ("derivative_liabilities", "Derivative liabilities"),
    ("goodwill", "Goodwill"), ("intangibles", "Intangibles"), ("liabilities", "Total liabilities"),
    ("equity", "Stockholders' equity"),
]


def render(sheet: FactSheet) -> str:
    """The sheet as prompt text: every figure with its key."""
    out = [f"## Fact sheet: {sheet.ticker} (as of {sheet.trade_date}; computed in code from filings and prices)"]
    ident = sheet.identity
    if ident:
        out.append(
            f"Company: {ident.get('name')} (SEC CIK {ident.get('cik')}; SIC {ident.get('sic')} "
            f"{ident.get('sic_description') or ''}; {', '.join(ident.get('exchanges') or []) or 'exchange n/a'}; "
            f"fiscal year ends {ident.get('fiscal_year_end') or 'n/a'})."
        )
    if ident.get("description"):
        out.append(f"Business (from the 10-K, {ident.get('description_source', '')}): {ident['description']}")
    if sheet.session.get("text"):
        out.append(f"Session clock: {sheet.session['text']}")
    if sheet.unavailable:
        out.append("Unavailable: " + "; ".join(sheet.unavailable) + ".")

    price = [f for f in sheet.facts if f.key.split(".")[0] in
             ("price", "ema10", "sma50", "sma200", "macd", "rsi14", "boll", "volume", "atr14")
             or re.match(r"^\d+(w|d)\.", f.key)]
    if price:
        out.append("\n### Price and technicals (as of " + (price[0].period or "") + ")")
        out.extend(f"- [F:{f.key}] {_fmt(f)}" + (f" ({f.derivation})" if f.derivation and f.unit != "text" else "")
                   for f in price)

    if sheet.quarters:
        cols = sheet.quarters
        head = ["Line item"] + [c.calendar + (f" ({c.fiscal})" if c.fiscal else "") for c in cols]
        out.append("\n### Quarterly statements (SEC filings as filed; calendar quarters, fiscal label in "
                   "brackets where different). Key = <item>.<calendar quarter>; * = derived, see notes")
        out.append("| " + " | ".join(head) + " |")
        out.append("|" + "---|" * len(head))
        notes: list[str] = []
        for concept, label in _ROWS:
            cells, any_value = [], False
            for c in cols:
                f = sheet.get(f"{concept}.{c.calendar}")
                if f is None:
                    cells.append("—")
                    continue
                any_value = True
                star = "*" if f.derivation and concept in _LINE_ITEMS else ""
                if star:
                    notes.append(f"{f.key} = {f.derivation}")
                cells.append(_fmt(f) + star)
            if any_value:
                out.append(f"| {label} [{concept}] | " + " | ".join(cells) + " |")
        out.append("\nBalances at quarter end (key = <item>.<calendar quarter>):")
        out.append("| " + " | ".join(["Balance"] + [f"{c.calendar} (at {c.end})" for c in cols]) + " |")
        out.append("|" + "---|" * (len(cols) + 1))
        for concept, label in _BALANCE_ROWS:
            cells = [_fmt(f) if (f := sheet.get(f"{concept}.{c.calendar}")) else "—" for c in cols]
            if any(c != "—" for c in cells):
                out.append(f"| {label} [{concept}] | " + " | ".join(cells) + " |")
        if notes:
            out.append("Derived quarters: " + "; ".join(notes[:12]) + ("; …" if len(notes) > 12 else "") + ".")

    if sheet.years:
        out.append("\n### Fiscal years (10-K)")
        for concept, label in _ROWS[:22]:
            vals = [(y.calendar, sheet.get(f"{concept}.{y.calendar}")) for y in sheet.years]
            if any(f for _, f in vals):
                out.append(f"- {label}: " + "; ".join(f"[F:{concept}.{y}] {_fmt(f)}" for y, f in vals if f))

    derived = [f for f in sheet.facts if f.key.startswith(("ttm_", "market_cap", "ev", "runway_", "shares.cover"))]
    if derived:
        out.append("\n### Derived (computed from the figures above)")
        out.extend(f"- [F:{f.key}] {_fmt(f)} — {f.derivation or f.concept}" + (f" ({f.period})" if f.period else "")
                   for f in derived)

    seg = [f for f in sheet.facts if f.key.startswith("seg_")]
    if seg:
        out.append(f"\n### Segments and product lines (as filed: {sheet.identity.get('segments_source', '')}). "
                   "Product lines can overlap or be parts of each other; they need not sum to revenue.")
        out.extend(f"- [F:{f.key}] {f.concept}, {f.period}: {_fmt(f)}" for f in seg)

    nxt = (sheet.calendar or {}).get("earnings_next")
    if nxt:
        out.append("\n### Earnings calendar")
        out.append(f"- Already reported (latest last): {', '.join(nxt.get('already_reported') or [])}. "
                   "These are PAST results, not upcoming ones.")
        out.append(f"- [F:earnings.next] next report covers {nxt['covers']}; date {nxt['status']}: "
                   f"{nxt['date']} ({nxt['basis']})." + (f" Note: {nxt['note']}." if nxt.get("note") else ""))

    if sheet.flags:
        out.append("\n### Flags you must address where relevant")
        out.extend(f"- {flag}" for flag in sheet.flags)
    return "\n".join(out)
