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
import math
import re
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from statistics import median
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from tradingagents.budget import reraise_if_budget
from tradingagents.quality import edgar_ext
from tradingagents.quality.session import clock

logger = logging.getLogger(__name__)

VERSION = 1

Unit = Literal["usd", "usd_per_share", "shares", "pct", "ratio", "x", "days", "date", "text", "quarters"]
Source = Literal["sec_xbrl", "sec_text", "computed", "price", "news", "fred"]


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
    """The column for one quarter end. Calendar labels strictly increase along
    ``ends``: COST's 12/12/12/16-week quarters put two quarter midpoints in one
    calendar quarter (2026-02-15 and 2026-05-10 both read 2026Q1, and their
    fact keys collided); a repeat moves to the next calendar quarter."""
    labels = _monotonic_labels(tuple(ends), fy_end)
    return labels.get(end) or _raw_q_label(end, ends, fy_end)


@lru_cache(maxsize=256)
def _monotonic_labels(ends: tuple[str, ...], fy_end) -> dict[str, QuarterCol]:
    out: dict[str, QuarterCol] = {}
    prev = None
    for e in ends:
        col = _raw_q_label(e, list(ends), fy_end)
        y, q = int(col.calendar[:4]), int(col.calendar[-1])
        if prev is not None and (y, q) <= prev:
            y, q = (prev[0] + 1, 1) if prev[1] == 4 else (prev[0], prev[1] + 1)
            col = QuarterCol(end=col.end, calendar=f"{y}Q{q}", fiscal=col.fiscal)
        prev = (y, q)
        out[e] = col
    return out


def _raw_q_label(end: str, ends: list[str], fy_end) -> QuarterCol:
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


# Without a debt tag, EV counts debt as 0 only when the balance sheet could
# not hide much: liabilities other than derivative liabilities (which EV
# excludes by design) at most a quarter of total assets (liabilities +
# equity). Ford's 10-Qs tag no total debt beside $245B of liabilities on $282B
# of assets: "no debt tagged, counted as 0" put its EV at $14B (release review
# 2026-10-09). ONDS, whose liabilities are mostly warrant derivatives (12% of
# assets once those are set aside), keeps its EV.
MATERIAL_LIABILITIES = 0.25
# R8 flags: a quarter's gross margin this many points from the median of the
# other quarters shown; antidilutive securities or warrants at this share of
# basic weighted shares.
ONE_OFF_MARGIN_PTS = 8.0
DILUTION_FLAG = 0.03
# Anomalies: a balance line moving this share in a quarter and at least this
# much; non-operating income at this share of net income.
ANOMALY_QOQ = 0.30
ANOMALY_MIN_USD = 1e9
ANOMALY_NONOP_SHARE = 0.25
STATUTORY_TAX = 0.21
# A cover-page share count is the latest count when it is current: older than
# this before the run, it's another era (Ford's companyfacts end its dei
# series in 2011).
COVER_MAX_AGE_DAYS = 400
# The cover-page count against the latest weighted basic count: shares rarely
# fall 10% in a quarter, and a count dated before the weighted quarter can't
# lead it by more. A cover dated after the quarter may lead it (ONDS issued
# heavily after June: 570.6M on the cover against 500.7M weighted), up to 2×.
COVER_MAX_GAP = 0.10
COVER_MAX_LEAD = 2.0


def _share_basis(st: edgar_ext.Statements, cols: list[QuarterCol], values: dict, as_of: str | None):
    """(shares, label, key, note) for market cap: the cover-page count when it
    is current and agrees with the weighted count, else the latest weighted
    basic count with ``note`` saying why. (None, …) when neither is usable."""
    weighted = [(c, values.get("shares_weighted", {}).get(c.end)) for c in cols]
    weighted = [(c, w) for c, w in weighted if w]
    latest = weighted[-1] if weighted else None
    cover = st.cover_shares.value if st.cover_shares else None
    problem = None
    if cover:
        cover_date = st.cover_shares_date or ""
        age = (date.fromisoformat(as_of) - date.fromisoformat(cover_date)).days if as_of and cover_date else 0
        if age > COVER_MAX_AGE_DAYS:
            problem = f"the cover-page count ({cover:,.0f}) is dated {cover_date}, over {COVER_MAX_AGE_DAYS} days old"
        elif latest:
            col, w = latest
            ratio = cover / w
            lead_ok = cover_date > col.end and ratio <= COVER_MAX_LEAD
            if ratio < 1 - COVER_MAX_GAP or (ratio > 1 + COVER_MAX_GAP and not lead_ok):
                problem = (f"the cover-page count ({cover:,.0f}, {cover_date}) is {abs(ratio - 1) * 100:.0f}% "
                           f"off the weighted basic count for {col.calendar} ({w:,.0f})")
        if problem is None:
            return cover, f"cover-page shares ({cover_date})", "shares.cover", None
    if latest:
        col, w = latest
        reason = f"cover-page share count not used: {problem}" if problem else "no cover-page share count filed"
        return (w, f"weighted basic shares for {col.calendar} (a period average)", f"shares_weighted.{col.calendar}",
                f"{reason}; market cap uses the weighted basic count for {col.calendar} [shares_weighted.{col.calendar}]")
    return None, None, None, (f"share count unavailable: {problem}; no market cap or EV" if problem else None)


def _derived(st: edgar_ext.Statements, cols: list[QuarterCol], values: dict, close: float | None,
             price_date: str | None, unavailable: list[str] | None = None) -> tuple[list[Fact], list[str]]:
    facts: list[Fact] = []
    flags: list[str] = []
    unavailable = unavailable if unavailable is not None else []
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
        # Acquisition-heavy balance sheets get argued as tangible equity; on the
        # sheet, the figure checks (staging ONDS, 2026-10-09: "tangible equity
        # ~$0.33B against $1.24B of acquisition intangibles" held a report).
        gw, intang, eq = v("goodwill", end), v("intangibles", end), v("equity", end)
        if gw is not None or intang is not None:
            add(f"goodwill_intangibles.{cal}", (gw or 0) + (intang or 0), "usd", "goodwill_and_intangibles",
                "goodwill + intangibles", f"at {end}")
            if eq is not None:
                add(f"tangible_equity.{cal}", eq - (gw or 0) - (intang or 0), "usd", "tangible_equity",
                    "stockholders' equity − goodwill − intangibles", f"at {end}")
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

    shares, shares_label, shares_key, shares_note = _share_basis(st, cols, values, st.as_of or price_date)
    if st.cover_shares is not None and shares_key == "shares.cover":
        facts.append(Fact(key="shares.cover", value=shares, unit="shares", period=st.cover_shares_date,
                          concept="dei:EntityCommonStockSharesOutstanding", source="sec_xbrl",
                          filed=f"{st.cover_shares.filed} {st.cover_shares.accn}"))
    if shares_note:
        # A stale or inconsistent cover count is left off the sheet: an agent
        # citing [F:shares.cover] would be dividing by another decade's count.
        unavailable.append(shares_note)
    if shares and close:
        mcap = shares * close
        add("market_cap", mcap, "usd", "market_cap",
            f"{shares_label} [{shares_key}] × close ({price_date})", price_date)
        debt = (v("debt", end) or 0) + (v("debt_current", end) or 0)
        debt_note = ""
        ev_ok = True
        if v("debt", end) is None and v("debt_current", end) is None:
            parts = {c: v(c, end) for c in _DEBT_PARTS if v(c, end) is not None}
            if parts:
                debt = sum(parts.values())
                add(f"debt_total.{cal}", debt, "usd", "debt_total",
                    "sum of " + ", ".join(_DEBT_PARTS[c] for c in parts) + " (no total-debt tag filed)", f"at {end}")
                debt_note = f"; debt is the sum of the instruments filed at {end} [debt_total.{cal}]"
            else:
                liab, eq = v("liabilities", end), v("equity", end)
                other = (liab or 0) - (v("derivative_liabilities", end) or 0)
                assets = (liab or 0) + (eq or 0)
                material = liab is None or eq is None or assets <= 0 or other > MATERIAL_LIABILITIES * assets
                if material:
                    ev_ok = False
                    unavailable.append(
                        f"enterprise value: no debt is tagged at {end}"
                        + (f" and liabilities are {_m(liab)} against {_m(assets)} of assets" if liab is not None
                           and assets > 0 else "")
                        + ", so debt can't be set to 0; no EV or EV/Sales (use market cap)")
                else:
                    debt_note = (f"; no debt tagged at {end}, counted as 0 (liabilities other than derivatives "
                                 f"are {_m(other)}, {other / assets * 100:.0f}% of assets)")
        if cash_sti is not None and ev_ok:
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

    # ---- R8: earnings multiples ------------------------------------------------
    ev_fact = next((f for f in facts if f.key == "ev"), None)
    mcap_fact = next((f for f in facts if f.key == "market_cap"), None)
    window = ends[ends.index(end) - 3:ends.index(end) + 1] if ends.index(end) >= 3 else []
    # Banks, REITs and others file no operating income: their profit test is
    # net income (JPM, O: no OperatingIncomeLoss in any quarter).
    has_oi = any(v("operating_income", e) is not None for e in window)
    ttm_oi = ttm("operating_income", end)
    common = ttm("net_income_common", end)
    ttm_ni = common if common is not None else ttm("net_income", end)
    ni_key = f"ttm_net_income_common.{cal}" if common is not None else f"ttm_net_income.{cal}"
    if common is not None:
        add(f"ttm_net_income_common.{cal}", common, "usd", "ttm_net_income_to_common",
            f"sum of the four quarters to {end}, net income available to common holders", f"TTM:{cal}")
    profitable = (ttm_oi is not None and ttm_oi > 0) if has_oi else (ttm_ni is not None and ttm_ni > 0)
    # Per-share lines are never derived for a Q4: the latest diluted count, or
    # the previous quarter's when the latest is a fiscal Q4.
    prev_q = edgar_ext._prev_quarter_end(end, ends)
    dil_shares, dil_key = None, None
    for concept, at in (("shares_diluted", end), ("shares_diluted", prev_q), ("shares_weighted", end),
                        ("shares_weighted", prev_q)):
        if at and v(concept, at):
            dil_shares = v(concept, at)
            dil_key = f"{concept}.{_q_label(at, ends, st.fy_end).calendar}"
            break
    if ttm_ni is not None and dil_shares:
        ttm_eps = ttm_ni / dil_shares
        add(f"ttm_eps.{cal}", ttm_eps, "usd_per_share", "ttm_eps_gaap",
            f"TTM net income [{ni_key}] ÷ weighted diluted shares [{dil_key}] (GAAP)", f"TTM:{cal}")
        if close and ttm_eps > 0 and profitable:
            add("pe.ttm", close / ttm_eps, "x", "pe_ttm_gaap",
                f"close ({price_date}) ÷ TTM GAAP EPS [ttm_eps.{cal}]; GAAP earnings include acquisition "
                "amortization, non-operating gains and one-off items", price_date)
    eps_q = v("eps_diluted", end)
    eps_basis = f"{cal} diluted EPS [eps_diluted.{cal}]"
    if eps_q is None and v("net_income", end) is not None and dil_shares:
        eps_q = v("net_income", end) / dil_shares
        why = "a fiscal Q4 files no quarterly EPS" if cal and end in st.year_ends else "no quarterly EPS tagged"
        eps_basis = f"{cal} net income [net_income.{cal}] ÷ [{dil_key}] ({why})"
    q_profitable = (v("operating_income", end) or 0) > 0 if has_oi else (v("net_income", end) or 0) > 0
    if close and eps_q and eps_q > 0 and q_profitable:
        add("pe.run_rate", close / (eps_q * 4), "x", "pe_run_rate_gaap", f"close ({price_date}) ÷ ({eps_basis} × 4)",
            price_date)
    # Operating multiples: what GOOGL's $98B non-operating gain can't distort.
    if ttm_oi is not None and ttm_oi > 0:
        if ev_fact:
            add("ev_ebit.ttm", ev_fact.value / ttm_oi, "x", "ev_to_ttm_operating_income",
                f"EV [ev] ÷ TTM operating income [ttm_operating_income.{cal}]", price_date)
        ttm_amort = ttm("amortization", end)
        if ev_fact and ttm_amort:
            add("ev_ebita.ttm", ev_fact.value / (ttm_oi + ttm_amort), "x", "ev_to_ttm_operating_income_before_amortization",
                f"EV [ev] ÷ (TTM operating income [ttm_operating_income.{cal}] + TTM amortization of acquired "
                "intangibles): the operating multiple without acquisition amortization", price_date)
        if close and dil_shares:
            op_eps = ttm_oi * (1 - STATUTORY_TAX) / dil_shares
            add("pe_operating.ttm", close / op_eps, "x", "pe_on_operating_earnings",
                f"close ({price_date}) ÷ (TTM operating income [ttm_operating_income.{cal}] × (1 − "
                f"{STATUTORY_TAX:.0%} statutory tax) ÷ [{dil_key}]); excludes non-operating items", price_date)
    ttm_fcf = (ttm("ocf", end) - ttm("capex", end)) if ttm("ocf", end) is not None and ttm("capex", end) is not None else None
    if mcap_fact and ttm_fcf is not None and mcap_fact.value:
        add("fcf_yield.ttm", ttm_fcf / mcap_fact.value * 100, "pct", "fcf_yield_ttm",
            f"TTM free cash flow [ttm_fcf.{cal}] ÷ market cap [market_cap]", price_date)

    # ---- R8: the next report's comparison base -----------------------------------
    # A year-on-year threshold for the next quarter is only a test if it is hard
    # to clear from where the business is now (AMD review: "Data Center growth
    # above 60%" needed ~3% sequential growth on a weak year-ago quarter).
    i_end = ends.index(end)
    base_end = ends[i_end - 3] if i_end >= 3 else None
    if base_end and v("revenue", base_end) and v("revenue", end):
        base_cal = _q_label(base_end, ends, st.fy_end).calendar
        scale, note = _length_scale(end, base_end, ends)
        add(f"next_base.revenue", v("revenue", base_end), "usd", "next_report_year_ago_revenue",
            f"revenue of {base_cal} [revenue.{base_cal}], the year-ago quarter of the next report", base_cal)
        add("next_hurdle.revenue_flat", (v("revenue", end) * scale / v("revenue", base_end) - 1) * 100, "pct",
            "next_report_growth_if_flat",
            f"year-on-year growth the next report shows if revenue only holds at {cal}'s level: "
            f"[revenue.{cal}] ÷ [revenue.{base_cal}] − 1{note}", base_cal)

    # A 3-month target values the business as the market will see it after the
    # next report, not on today's TTM (eval, 2026-10-10: MSFT and JPM held TTM
    # constant; JPM's own 13x on the rolled TTM gave ~$341, not $307). Each
    # figure is today's TTM with the year-ago quarter swapped for the latest
    # quarter's level, so it is what TTM reads if the next quarter is flat.
    if base_end:
        scale, note = _length_scale(end, base_end, ends)
        base_cal = _q_label(base_end, ends, st.fy_end).calendar
        for concept, key in (("revenue", "revenue"), ("operating_income", "operating_income"),
                             ("net_income", "net_income")):
            total, now_q, base_q = ttm(concept, end), v(concept, end), v(concept, base_end)
            if total is None or now_q is None or base_q is None:
                continue
            rolled = total - base_q + now_q * scale
            add(f"ttm_next.{key}", rolled, "usd", f"ttm_{concept}_after_next_report_if_flat",
                f"TTM [ttm_{concept}.{cal}] − {base_cal} [{concept}.{base_cal}] + {cal} [{concept}.{cal}]"
                f"{note}: TTM after the next report if that quarter matches {cal}", f"next:{cal}")
            if concept == "net_income" and dil_shares:
                add("ttm_next.eps", rolled / dil_shares, "usd_per_share", "ttm_eps_after_next_report_if_flat",
                    f"[ttm_next.net_income] ÷ [{dil_key}]", f"next:{cal}")

    # ---- R8: what the investments are --------------------------------------------
    # GOOGL 2026Q2: $186.6B of "marketable securities" held $99.5B of debt
    # securities; the rest was equity marked to market, not cash.
    sti_now, afs = v("sti", end), v("afs_debt", end)
    if sti_now and afs is not None and sti_now - afs > max(0.1 * sti_now, 1e9):
        add(f"sti_equity.{cal}", sti_now - afs, "usd", "marketable_equity_in_short_term_investments",
            f"short-term investments [sti.{cal}] − debt securities available for sale [afs_debt.{cal}]", f"at {end}")
        add(f"cash_debt_securities.{cal}", (v("cash", end) or 0) + afs, "usd", "cash_and_debt_securities",
            f"cash [cash.{cal}] + debt securities [afs_debt.{cal}]: the cash-like part", f"at {end}")
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
    # R8: net income well above operating income (AMD 2026Q2: $2.3B against
    # $2.0B) reads as operating strength unless the gap is named.
    if oi is not None and ni is not None and oi > 0 and ni > oi * 1.1:
        flags.append(
            f"In {cal} net income ({_m(ni)}) exceeds operating income ({_m(oi)}) by {_m(ni - oi)}: non-operating "
            f"items ({_m(nonop) if nonop is not None else 'see the statements'}) or a tax benefit. Name it when "
            "citing net income or EPS; operating income is the operating measure."
        )
    # R8: a quarter whose gross margin sits far from the others is likely a
    # one-off (AMD 2025Q2: an ~$800M inventory charge made the year-on-year
    # "turnaround" look larger than it was).
    gms = {}
    for c in cols:
        r_, g_ = v("revenue", c.end), v("gross_profit", c.end)
        if r_ and g_ is not None:
            gms[c.calendar] = g_ / r_ * 100
    for qcal, gm in gms.items():
        others = [g for k, g in gms.items() if k != qcal]
        # Only a business with a steady positive margin has a "normal" to
        # depart from (WKHS's −190% margins flagged every quarter).
        if len(others) < 3 or median(others) <= 0 or min(others) < 0 or not -100 < gm < 100:
            continue
        if abs(gm - median(others)) >= ONE_OFF_MARGIN_PTS:
            med = median(others)
            flags.append(
                f"{qcal} gross margin ({gm:.1f}%) is {abs(gm - med):.0f} points {'below' if gm < med else 'above'} "
                f"the median of the other quarters shown ({med:.1f}%): likely a one-off (an inventory or "
                f"impairment charge, a write-down, a one-time gain). Say so wherever {qcal} is a comparison base "
                "(year-on-year growth, margin expansion, a segment's swing from loss to profit), and don't present "
                "the change from it as underlying improvement."
            )
    # R8: dilution the basic count doesn't show.
    basic = v("shares_weighted", end)
    if basic:
        for concept, what in (("antidilutive", "securities excluded from diluted shares as antidilutive"),
                              ("warrants_outstanding", "warrants or rights outstanding")):
            # The latest quarter that tags it: a warrant disclosed in one 10-Q
            # (AMD's OpenAI warrant, 160M at 2026Q1) need not be re-tagged.
            at = next((c for c in reversed(cols) if v(concept, c.end)), None)
            n = v(concept, at.end) if at else None
            if n and n / basic >= DILUTION_FLAG:
                flags.append(
                    f"Potential dilution: {n / 1e6:,.1f}M {what} as of {at.calendar} [{concept}.{at.calendar}], "
                    f"{n / basic * 100:.1f}% of basic weighted shares. Weighted-share growth understates it; "
                    "when they can convert (price or milestone conditions) comes only from the filings or news."
                )
    # R8: anomalies a reader would ask about first (GOOGL 2026Q2: cash + short-
    # term investments up $116B in a quarter with −$5.9B FCF and a $98B gain).
    moved: set[str] = set()
    if prev:
        fcf_q = (v("ocf", end) - v("capex", end)) if v("ocf", end) is not None and v("capex", end) is not None else None
        cs_now = (v("cash", end) or 0) + (v("sti", end) or 0) if v("cash", end) is not None else None
        cs_prev = (v("cash", prev) or 0) + (v("sti", prev) or 0) if v("cash", prev) is not None else None
        if cs_now is not None and cs_prev and fcf_q is not None:
            move = cs_now - cs_prev
            unexplained = move - fcf_q
            if abs(move) / cs_prev >= ANOMALY_QOQ and abs(unexplained) >= max(ANOMALY_MIN_USD, 0.1 * cs_prev):
                flags.append(
                    f"Cash + short-term investments moved {_m(move)} ({move / cs_prev * 100:+.0f}%) in {cal} while "
                    f"free cash flow was {_m(fcf_q)}: {_m(unexplained)} is not operating cash. Check investments "
                    "marked to market or reclassified, borrowing, and stock sales before calling it a cushion; "
                    "say what drove it or that the filings don't say."
                )
        def total_debt(at):
            # Long-term plus current: a bond moving to "current" is not a move
            # (COST 2026Q3: −31% long-term, +9% in total). Without the long-term
            # part it is not a total (XOM's current debt read as "Total debt").
            if v("debt", at) is None:
                return None
            return v("debt", at) + (v("debt_current", at) or 0)
        for concept, label, now_, then_ in (
                ("debt", "Total debt", total_debt(end), total_debt(prev)),
                ("equity", "Stockholders' equity", v("equity", end), v("equity", prev)),
                ("goodwill", "Goodwill", v("goodwill", end), v("goodwill", prev)),
                ("liabilities", "Total liabilities", v("liabilities", end), v("liabilities", prev))):
            if concept == "liabilities" and "debt" in moved:
                continue              # one borrowing, one flag (NVDA got three)
            if now_ is not None and then_ and abs(now_ - then_) / abs(then_) >= ANOMALY_QOQ \
                    and abs(now_ - then_) >= ANOMALY_MIN_USD:
                moved.add(concept)
                keys = f"[debt.{cal}] + [debt_current.{cal}]" if concept == "debt" else f"[{concept}.{cal}]"
                flags.append(f"{label} moved {_m(now_ - then_)} ({(now_ / then_ - 1) * 100:+.0f}%) in {cal} "
                             f"{keys}: name the cause (borrowing, an acquisition, a revaluation) "
                             "where it matters to the case.")
    if prior and v("debt", end) is not None and v("debt", prior) and "debt" not in moved:
        total_now = (v("debt", end) or 0) + (v("debt_current", end) or 0)
        total_then = (v("debt", prior) or 0) + (v("debt_current", prior) or 0)
        if total_then and total_now / total_then - 1 >= 0.5 and total_now - total_then >= ANOMALY_MIN_USD:
            flags.append(f"Debt rose from {_m(total_then)} to {_m(total_now)} in a year: say what it funds "
                         "(capex, buybacks, an acquisition) and weigh it in the bear case.")
    if nonop is not None and ni and abs(nonop) >= ANOMALY_NONOP_SHARE * abs(ni) and abs(nonop) >= ANOMALY_MIN_USD:
        word, effect = ("income", "overstate") if nonop > 0 else ("loss", "understate")
        flags.append(f"Non-operating {word} of {_m(nonop)} in {cal} is {abs(nonop) / abs(ni) * 100:.0f}% of net "
                     f"income [non_operating.{cal}]: earnings-based figures that include it (net income, EPS, P/E) "
                     f"{effect} the operating business; use operating income, EV/EBIT or the operating P/E.")
    if sti_now and afs is not None and sti_now - afs > max(0.1 * sti_now, 1e9):
        flags.append(f"Of {_m(sti_now)} short-term investments in {cal}, {_m(sti_now - afs)} is not debt securities "
                     f"[sti_equity.{cal}]: likely marketable equity at market value. It is a volatile position, not "
                     "cash: keep it apart from liquidity and say how the target treats it.")
    if prior and v("shares_weighted", end) and v("shares_weighted", prior):
        growth = v("shares_weighted", end) / v("shares_weighted", prior) - 1
        if growth > 0.25:
            flags.append(
                f"Weighted shares grew {growth * 100:.0f}% year on year to {cal}: discuss dilution with the "
                "share bridge (stock sold for cash, stock issued for acquisitions)."
            )
    return facts, flags


# ---- R8: the stock's own valuation history and the macro rate -----------------------

HIST_QUARTERS = 20     # five years of quarter ends
HIST_MIN_POINTS = 6


def _close_on(frame: pd.DataFrame | None, day: str) -> float | None:
    """The last close on or before ``day``, within a week of it."""
    if frame is None or frame.empty:
        return None
    dates = pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d")
    mask = dates <= day
    if not mask.any():
        return None
    last = dates[mask].iloc[-1]
    if (date.fromisoformat(day) - date.fromisoformat(last)).days > 7:
        return None
    return float(frame.loc[mask, "Close"].iloc[-1])


# Bounds outside which a historical multiple is a data error, not a market
# price (WKHS, a reverse-split micro-cap with a negative-revenue quarter, read
# −3,256,158x): the point is dropped, and the series withheld if many are.
_PLAUSIBLE = {"ev_sales": (0.0, 500.0), "ev_ebit": (0.0, 1000.0), "pe": (0.0, 2000.0)}


def _valuation_history(st: edgar_ext.Statements, values: dict, frame: pd.DataFrame | None,
                       current: dict[str, float | None], price_date: str | None,
                       splits: list[tuple[str, float]] | None = None) -> list[Fact]:
    """EV/TTM sales, EV/TTM operating income and TTM P/E at each past quarter
    end, as low / median / high and today's percentile: the anchor a target
    multiple is argued against (AMD review, 2026-10-10: "assumed 18x" with
    nothing behind it).

    ``frame`` holds split-only closes (not dividend-adjusted). A share count
    filed before a split is scaled by that split; one filed after it was
    restated already. Yahoo lists spin-offs as splits too, and adjusts its
    prices by the same factor, so the market cap stays right either way."""
    ends = st.quarter_ends

    def v(concept, end):
        return values.get(concept, {}).get(end)

    def count_at(concept, i):
        # Q4 counts aren't derived: the nearest quarter's.
        for j in (i, i - 1, i + 1):
            if not 0 <= j < len(ends):
                continue
            filed = st.quarters.get(concept, {}).get(ends[j])
            if filed is None or not filed.value:
                continue
            factor = 1.0
            # Prices are on today's split basis: every split after the count
            # was filed applies, whatever quarter it fell in.
            for day, ratio in splits or []:
                if (filed.filed or "") < day:
                    factor *= ratio
            return filed.value * factor
        return None

    has_oi = any(v("operating_income", e) is not None for e in ends)
    points: dict[str, list[tuple[str, float]]] = {"ev_sales": [], "ev_ebit": [], "pe": []}
    seen = {k: 0 for k in points}
    for i in range(3, len(ends)):
        end, window = ends[i], ends[i - 3:i + 1]
        span_days = (date.fromisoformat(window[-1]) - date.fromisoformat(window[0])).days
        if not 240 <= span_days <= 310:      # four consecutive quarters, no gap
            continue
        px, sh = _close_on(frame, end), count_at("shares_weighted", i)
        if not px or not sh or sh <= 0:
            continue
        label = _q_label(end, ends, st.fy_end).calendar
        revs = [v("revenue", e) for e in window]
        ois = [v("operating_income", e) for e in window]
        nis = [v("net_income_common", e) for e in window]
        if any(x is None for x in nis):
            nis = [v("net_income", e) for e in window]
        cash = v("cash", end)
        ev = None
        if cash is not None:
            debt_tagged = v("debt", end) is not None or v("debt_current", end) is not None
            debt = (v("debt", end) or 0) + (v("debt_current", end) or 0)
            if not debt_tagged:
                debt = sum(v(c, end) or 0 for c in _DEBT_PARTS)
                liab, eq = v("liabilities", end), v("equity", end)
                assets = (liab or 0) + (eq or 0)
                other = (liab or 0) - (v("derivative_liabilities", end) or 0)
                if not debt and (liab is None or eq is None or assets <= 0 or other > MATERIAL_LIABILITIES * assets):
                    debt = None
            if debt is not None:
                ev = px * sh - cash - (v("sti", end) or 0) + debt
        if ev is not None and current.get("ev_sales") is not None and all(r is not None for r in revs) \
                and sum(revs) > 0:
            seen["ev_sales"] += 1
            points["ev_sales"].append((label, ev / sum(revs)))
        if ev is not None and all(x is not None for x in ois) and sum(ois) > 0:
            seen["ev_ebit"] += 1
            points["ev_ebit"].append((label, ev / sum(ois)))
        profitable = (all(x is not None for x in ois) and sum(ois) > 0) if has_oi else True
        if all(x is not None for x in nis) and sum(nis) > 0 and profitable:
            dil = count_at("shares_diluted", i) or sh
            seen["pe"] += 1
            points["pe"].append((label, px / (sum(nis) / dil)))

    index_of = {_q_label(e, ends, st.fy_end).calendar: i for i, e in enumerate(ends)}
    facts: list[Fact] = []
    for metric, what in (("ev_sales", "EV ÷ TTM revenue"), ("ev_ebit", "EV ÷ TTM operating income"),
                         ("pe", "price ÷ TTM GAAP EPS")):
        lo, hi = _PLAUSIBLE[metric]
        raw = points[metric][-HIST_QUARTERS:]
        series = [(q, x) for q, x in raw if math.isfinite(x) and lo < x < hi]
        if len(series) < HIST_MIN_POINTS or len(series) < 0.8 * len(raw):
            continue
        # A history that stopped long ago (MRNA's P/E ends in 2023Q2: losses
        # since) anchors nothing today.
        if index_of[series[-1][0]] < len(ends) - 3:
            continue
        vals = sorted(x for _, x in series)
        span = f"{series[0][0]}–{series[-1][0]}"
        how = (f"{what} at each of {len(series)} quarter ends {span} (the close at the quarter end, adjusted for "
               "splits only; that quarter's weighted shares on the same split basis; cash and debt then)")
        for stat, value in (("low", vals[0]), ("median", median(vals)), ("high", vals[-1])):
            facts.append(Fact(key=f"{metric}_hist.{stat}", value=round(value, 2), unit="x", period=span,
                              concept=f"{metric}_history_{stat}", source="computed", derivation=how))
        now = current.get(metric)
        if now is not None:
            pct = sum(1 for x in vals if x <= now) / len(vals) * 100
            facts.append(Fact(key=f"{metric}_hist.percentile", value=round(pct), unit="ratio", period=price_date,
                              concept=f"{metric}_percentile_in_history", source="computed",
                              derivation=f"share of the {len(series)} quarter-end values at or below today's "
                                         f"{now:.2f}x, in percent (today's uses the cover-page share count)"))
    return facts


def _macro_facts(trade_date: str) -> list[Fact]:
    """The 10-year Treasury yield (FRED DGS10) on or before the trade date, so
    a stage citing it cites a figure on the sheet (AMD, 2026-10-10: "a 5.22%
    10-year yield" came from a tool call no one could check)."""
    from tradingagents.dataflows.vendors import fred

    start = (date.fromisoformat(trade_date) - timedelta(days=14)).isoformat()
    pit = min(trade_date, fred._fred_today())
    obs = fred._request("series/observations", {
        "series_id": "DGS10", "observation_start": start, "observation_end": trade_date,
        "sort_order": "asc", "realtime_start": pit, "realtime_end": pit,
    }).get("observations", [])
    points = [(o["date"], float(o["value"])) for o in obs if o.get("value") not in (".", None, "")]
    if not points:
        return []
    day, value = points[-1]
    return [Fact(key="macro.ust10y", value=value, unit="pct", period=day, concept="us_10y_treasury_yield",
                 source="fred", derivation=f"FRED DGS10, observation of {day}")]


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
    # Last year's results release for the same quarter (8-K Item 2.02), when
    # filed: banks release results weeks before the 10-Q (JPM: Oct 14 against
    # a Nov 4 10-Q; the eval's report timed trims to the wrong date).
    recent = (st.submissions.get("filings") or {}).get("recent") or {}
    prior_end = next_end - timedelta(days=365)
    releases = sorted(
        fd for f, fd, items in zip(recent.get("form") or [], recent.get("filingDate") or [], recent.get("items") or [])
        if f == "8-K" and "2.02" in str(items)
        and 0 < (date.fromisoformat(fd) - prior_end).days <= 75)
    if releases:
        filed = date.fromisoformat(releases[0]) + timedelta(days=364)
        basis = (f"last year's results for the same quarter were released {releases[0]} (8-K, Item 2.02), "
                 "a year on")
    elif year_ago:
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
        reraise_if_budget(exc)   # the summary is a model call on the run's budget
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
        # The filing for the latest quarter shown, so segments line up with the
        # statements (KO: the newest 10-Q was not yet in the structured data).
        last_end = sheet.quarters[-1].end if sheet.quarters else None
        candidates = [r for r in edgar_ext.filings(st.submissions) if r["filed"] <= trade_date]
        filing = next((r for r in candidates if last_end and
                       abs((date.fromisoformat(r["period"]) - date.fromisoformat(last_end)).days) <= 10),
                      candidates[0] if candidates else None)
        if filing is None:
            return
        ends = list({*st.quarter_ends, *st.year_ends})
        rows = edgar_ext.fetch_segments(st.cik, filing, ends)
        facts = segment_facts(rows, sheet.quarters, sheet.years, f"{filing['filed']} {filing['accn']}")
        sheet.facts.extend(facts)
        if facts:
            sheet.identity["segments_source"] = f"{filing['form']} for {filing['period']}, filed {filing['filed']}"
            _add_segment_hurdles(sheet, st, trade_date, facts)
    except Exception as exc:  # noqa: BLE001
        logger.info("fact sheet: segments for %s skipped: %s", sheet.ticker, exc)


def _length_scale(end: str, base_end: str, ends: list[str]) -> tuple[float, str]:
    """(factor putting the latest quarter on the base quarter's length, note).
    COST's 16-week fiscal Q4 against a 12-week Q1 read +42% "if flat"."""
    days = (date.fromisoformat(end) - edgar_ext.quarter_start(end, ends)).days + 1
    base_days = (date.fromisoformat(base_end) - edgar_ext.quarter_start(base_end, ends)).days + 1
    if abs(days - base_days) <= 10 or days <= 0:
        return 1.0, ""
    return base_days / days, (f", the latest quarter scaled from {days} to {base_days} days (the quarters differ "
                              "in length; a seasonal business can still differ)")


def _add_segment_hurdles(sheet: FactSheet, st: edgar_ext.Statements, trade_date: str, latest: list[Fact]) -> None:
    """Each segment's revenue in the next report's year-ago quarter, from that
    quarter's own filing, and the year-on-year growth the next report shows if
    the segment only holds at its latest level."""
    ends = st.quarter_ends
    if len(ends) < 4 or not sheet.quarters:
        return
    base_end, last_cal = ends[-4], sheet.quarters[-1].calendar
    base_filing = next((r for r in edgar_ext.filings(st.submissions)
                        if r["filed"] <= trade_date
                        and abs((date.fromisoformat(r["period"]) - date.fromisoformat(base_end)).days) <= 10), None)
    if base_filing is None:
        return
    base_cal = _q_label(base_end, ends, st.fy_end).calendar
    rows = [r for r in edgar_ext.fetch_segments(st.cik, base_filing, [base_end])
            if r["end"] == base_end and r["span"] == "Q" and r["concept"] == "revenue"]
    now = {f.key.split(".")[1]: f for f in latest if f.key.startswith("seg_revenue.") and f.period == last_cal}
    scale, note = _length_scale(ends[-1], base_end, ends)
    for r in rows:
        slug = _slug(r["label"])
        if slug not in now or not r["value"]:
            continue
        kind = "segment" if r["axis"] == "segment" else "product line"
        sheet.facts.append(Fact(key=f"seg_revenue.{slug}.{base_cal}", value=r["value"], unit="usd", period=base_cal,
                                concept=f"{kind}: {r['label']}", source="sec_xbrl",
                                filed=f"{base_filing['filed']} {base_filing['accn']}"))
        sheet.facts.append(Fact(
            key=f"seg_hurdle.{slug}", value=round((now[slug].value * scale / r["value"] - 1) * 100, 2), unit="pct",
            period=base_cal, concept=f"{r['label']}: next report's growth if flat", source="computed",
            derivation=f"[seg_revenue.{slug}.{last_cal}] ÷ [seg_revenue.{slug}.{base_cal}] − 1: the year-on-year "
                       f"growth the next report shows if {r['label']} only holds at {last_cal}'s level{note}"))


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
          describe=None, history: tuple[pd.DataFrame, list[tuple[str, float]]] | None = None) -> FactSheet:
    """The fact sheet for one run. Never raises.

    ``describe``: optional ``fn(excerpt) -> str`` that summarises the 10-K's
    Item 1 (the deep model, from the graph); without it there is no
    business description."""
    sheet = FactSheet(ticker=ticker.upper(), asset_type=asset_type, trade_date=trade_date,
                      built_at=datetime.now(UTC).isoformat(timespec="seconds"))
    close, last_bar, frame = None, None, ohlcv
    history_inputs = None
    # Tests pass split-only closes and splits; live runs fetch them.
    history_frame, history_splits = history if history is not None else (None, None)
    try:
        if frame is None:
            from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv

            frame = load_ohlcv(ticker, trade_date)
        price, last_bar = _price_facts(ticker, trade_date, frame)
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
            # Five years of quarters for the valuation history (R8); only the
            # last n_quarters are shown.
            st = edgar_ext.load(ticker, trade_date, n_quarters=max(HIST_QUARTERS + 4, n_quarters))
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
            derived, flags = _derived(st, cols, values, close, last_bar, sheet.unavailable)
            sheet.facts.extend(derived)
            sheet.flags.extend(flags)
            history_inputs = (st, values)
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
            reraise_if_budget(exc)
            logger.warning("fact sheet: statements for %s failed: %s", ticker, exc, exc_info=True)
            sheet.unavailable.append("SEC statements could not be read; no fundamentals figures")

    if st is not None and sheet.quarters:
        # SEC's structured data can lag a filing by weeks (KO, 2026-10-09: the
        # 10-Q for the quarter to 2026-07-03, filed 2026-07-29, was not in it).
        recent = ((st.submissions.get("filings") or {}).get("recent") or {})
        filed = [(rd, fd, f) for f, fd, rd in zip(recent.get("form") or [], recent.get("filingDate") or [],
                                                  recent.get("reportDate") or [])
                 if str(f).startswith(("10-Q", "10-K")) and fd <= trade_date and rd]
        if filed:
            report, filed_on, form = max(filed)
            last_end = sheet.quarters[-1].end
            if (date.fromisoformat(report) - date.fromisoformat(last_end)).days > 20:
                note = (f"the {form} for the period to {report} (filed {filed_on}) is not yet in SEC's structured "
                        f"data; statement figures run through {sheet.quarters[-1].calendar}")
                sheet.unavailable.append(note)
                sheet.flags.append(f"The latest filing ({form}, period to {report}, filed {filed_on}) is missing from "
                                   f"the statements: every quarterly figure here ends at {last_end}. Say the figures "
                                   "are a quarter old wherever recency matters, and don't call them the latest quarter.")
                # The "next report" and its comparison base would point at a
                # quarter already reported (KO): drop the hurdles, roll the
                # calendar on to the quarter after the filed one.
                sheet.facts = [f for f in sheet.facts
                               if not f.key.startswith(("next_base.", "next_hurdle.", "seg_hurdle."))]
                nxt = (sheet.calendar or {}).get("earnings_next")
                if nxt:
                    after = date.fromisoformat(report) + timedelta(days=91)
                    est = date.fromisoformat(nxt["date"]) + timedelta(days=91)
                    nxt.update(covers=f"the quarter to about {after.isoformat()}", date=est.isoformat(),
                               period_end_approx=after.isoformat(), status="estimated",
                               basis=f"the {form} for the period to {report} was filed {filed_on}; a quarter on "
                                     "from the earlier estimate", note=None)
                    nxt.pop("note", None)
                    nxt.pop("link", None)
                    nxt["already_reported"] = [*nxt.get("already_reported", []), f"the quarter to {report}"]
                    for f in sheet.facts:
                        if f.key == "earnings.next":
                            f.value, f.period = est.isoformat(), nxt["covers"]
                            f.derivation = f"estimated: {nxt['basis']}"
    if st is not None and not sheet.quarters and not any(u.startswith("SEC statements") for u in sheet.unavailable):
        # A 20-F filer (US GAAP or IFRS) files no quarterly statements: say so,
        # or the agents are told the sheet holds the company's statements.
        forms = ((st.submissions.get("filings") or {}).get("recent") or {}).get("form") or []
        foreign = any(str(f).startswith(("20-F", "40-F")) for f in forms)
        sheet.unavailable.append(
            "SEC statements unavailable: " + ("annual-only 20-F/IFRS filer, " if foreign else "")
            + "no quarterly statements on file; no fundamentals figures")
    if st is not None and history_inputs is not None and sheet.quarters:
        # Its own try: a failure here must not drop the statements above.
        try:
            current = {"ev_sales": sheet.value("ev_sales.ttm"), "ev_ebit": sheet.value("ev_ebit.ttm"),
                       "pe": sheet.value("pe.ttm")}
            if history_frame is None and not offline:
                from tradingagents.dataflows.vendors.yahoo.history import split_history

                history_frame, history_splits = split_history(ticker, trade_date)
            if history_frame is not None:
                sheet.facts.extend(_valuation_history(st, history_inputs[1], history_frame, current, last_bar,
                                                      history_splits))
        except Exception as exc:  # noqa: BLE001
            reraise_if_budget(exc)
            logger.info("fact sheet: valuation history for %s skipped: %s", ticker, exc)
            sheet.unavailable.append("valuation history unavailable (split-adjusted prices could not be read); "
                                     "argue the multiple from growth and margins")
    if not offline and asset_type != "crypto":
        try:
            sheet.facts.extend(_macro_facts(trade_date))
        except Exception as exc:  # noqa: BLE001 — no FRED key or FRED down: the sheet goes on
            reraise_if_budget(exc)
            logger.info("fact sheet: macro facts skipped: %s", exc)
            sheet.unavailable.append("10-year Treasury yield unavailable (FRED); do not quote one")
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
        # Prices, averages, ATR and levels keep their cents: rendering $6.85 as
        # "$7" put every stage on rounded prices (staging ONDS, 2026-10-08).
        if a < 1000:
            return f"{'−' if v < 0 else ''}${a:,.2f}"
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


# Debt filed by instrument (edgar_ext.LINES), summed only when no total is tagged.
_DEBT_PARTS = {"notes_payable": "notes payable", "loans_payable": "loans payable", "secured_debt": "secured debt",
               "commercial_paper": "commercial paper", "credit_line": "credit-line borrowings"}

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
    ("shares_diluted", "Weighted shares (diluted)"), ("shares_yoy", "Weighted shares YoY"),
    ("antidilutive", "Securities excluded as antidilutive"),
]
_LINE_ITEMS = {concept for concept, _kind, _tags in edgar_ext.LINES}
_BALANCE_ROWS = [
    ("cash", "Cash"), ("sti", "Short-term investments"), ("cash_sti", "Cash + short-term investments"),
    ("debt", "Long-term debt"), ("debt_current", "Current debt"), ("notes_payable", "Notes payable"),
    ("loans_payable", "Loans payable"), ("secured_debt", "Secured debt"), ("commercial_paper", "Commercial paper"),
    ("credit_line", "Credit-line borrowings"), ("debt_total", "Debt, summed from instruments"), ("derivative_liabilities", "Derivative liabilities"),
    ("goodwill", "Goodwill"), ("intangibles", "Intangibles"), ("goodwill_intangibles", "Goodwill + intangibles"),
    ("liabilities", "Total liabilities"), ("equity", "Stockholders' equity"),
    ("tangible_equity", "Tangible equity (equity − goodwill − intangibles)"),
    ("afs_debt", "Debt securities available for sale"),
    ("sti_equity", "Short-term investments that are not debt securities (marketable equity)"),
    ("cash_debt_securities", "Cash + debt securities (the cash-like part)"),
    ("equity_securities_fv", "Equity securities at fair value"),
    ("equity_nonmarketable", "Equity stakes without a market price"),
    ("lt_investments", "Long-term investments"),
    ("warrants_outstanding", "Warrants or rights outstanding (count)"),
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

    derived = [f for f in sheet.facts if f.key.startswith(("ttm_", "market_cap", "ev", "runway_", "shares.cover",
                                                           "pe.", "pe_operating.", "fcf_yield"))
               and not f.key.startswith(("next_base", "next_hurdle", "ttm_next"))
               and "_hist." not in f.key]
    if derived:
        out.append("\n### Derived (computed from the figures above)")
        out.extend(f"- [F:{f.key}] {_fmt(f)} — {f.derivation or f.concept}" + (f" ({f.period})" if f.period else "")
                   for f in derived)

    hist = [f for f in sheet.facts if "_hist." in f.key]
    if hist:
        out.append("\n### Valuation history (the stock's own range; argue a target multiple against it)")
        for metric, label in (("ev_sales", "EV / TTM sales"), ("ev_ebit", "EV / TTM operating income"),
                              ("pe", "P/E on TTM GAAP EPS")):
            got = {f.key.split(".")[1]: f for f in hist if f.key.startswith(f"{metric}_hist.")}
            if "median" not in got:
                continue
            line = (f"- {label}: low [F:{metric}_hist.low] {_fmt(got['low'])}, median [F:{metric}_hist.median] "
                    f"{_fmt(got['median'])}, high [F:{metric}_hist.high] {_fmt(got['high'])}")
            if "percentile" in got:
                line += f"; today at percentile {got['percentile'].value:.0f} [F:{metric}_hist.percentile]"
            out.append(line + f" ({got['median'].derivation})")

    hurdles = [f for f in sheet.facts if f.key.startswith(("next_base.", "next_hurdle.", "seg_hurdle.", "ttm_next."))]
    if hurdles:
        out.append("\n### The next report's comparison base (check any year-on-year threshold against it)")
        out.extend(f"- [F:{f.key}] {_fmt(f)} — {f.derivation}" for f in hurdles)
        out.append("A growth threshold at or below the 'if flat' figure is cleared by standing still: it tests "
                   "nothing. State a threshold with the level it implies and the sequential change from the latest "
                   "quarter, and set it where it would actually discriminate. The 'if flat' figures are hurdles for "
                   "tests, not a base case. ttm_next.* is TTM after the next report if that quarter matches the "
                   "latest one: the base a 3-month target's multiple applies to (or your own explicit "
                   "next-quarter scenario).")

    macro = [f for f in sheet.facts if f.key.startswith("macro.")]
    if macro:
        out.append("\n### Macro")
        out.extend(f"- [F:{f.key}] {_fmt(f)} ({f.derivation})" for f in macro)

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
