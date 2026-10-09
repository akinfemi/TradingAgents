"""Statements for the fact sheet, from SEC EDGAR as filed (REPORT_QUALITY_PLAN R4).

Upstream's ``dataflows/vendors/sec_edgar`` prints statement tables for the
analysts and never derives a figure. The fact sheet needs one value per
line item per quarter, so this module derives the two quarters filers don't
state, and marks every derived value with how it was derived:

- a fourth quarter is the fiscal year less the nine months to date;
- a cash-flow quarter filed only year to date is that year-to-date figure
  less the one to the previous quarter end.

Every value is "as known on ``as_of``": a fact filed later is invisible, and
a period reported more than once takes its latest filing on or before then.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from html import unescape
from pathlib import Path

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.vendors.sec_edgar import _FACTS_URL, _cached_json, cik_for

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# Line items: (concept, kind, tags best first). "flow" items cover a period
# (quarter, year); "stock" items are balances at a date. First tag reporting a
# period wins; values are never summed across tags (double counting).
LINES: list[tuple[str, str, tuple[str, ...]]] = [
    ("revenue", "flow", ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                         "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet")),
    ("cost_of_revenue", "flow", ("CostOfRevenue", "CostOfGoodsAndServicesSold")),
    ("gross_profit", "flow", ("GrossProfit",)),
    ("sga", "flow", ("SellingGeneralAndAdministrativeExpense",)),
    ("ga", "flow", ("GeneralAndAdministrativeExpense",)),
    ("sm", "flow", ("SellingAndMarketingExpense",)),
    ("rnd", "flow", ("ResearchAndDevelopmentExpense",
                     "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost")),
    ("amortization", "flow", ("AmortizationOfIntangibleAssets",)),
    ("dna", "flow", ("DepreciationDepletionAndAmortization", "DepreciationAndAmortization")),
    ("opex", "flow", ("OperatingExpenses",)),
    ("operating_income", "flow", ("OperatingIncomeLoss",)),
    ("non_operating", "flow", ("NonoperatingIncomeExpense",)),
    ("warrant_fair_value", "flow", ("FairValueAdjustmentOfWarrants",)),
    ("net_income", "flow", ("NetIncomeLoss",)),
    ("eps_basic", "flow", ("EarningsPerShareBasic",)),
    ("eps_diluted", "flow", ("EarningsPerShareDiluted",)),
    ("sbc", "flow", ("ShareBasedCompensation", "AllocatedShareBasedCompensationExpense")),
    ("ocf", "flow", ("NetCashProvidedByUsedInOperatingActivities",
                     "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")),
    ("capex", "flow", ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets")),
    ("acquisitions_cash", "flow", ("PaymentsToAcquireBusinessesNetOfCashAcquired",
                                   "PaymentsToAcquireBusinessesGross")),
    ("stock_sold_cash", "flow", ("ProceedsFromIssuanceOfCommonStock",)),
    ("stock_for_acquisitions", "flow", ("StockIssuedDuringPeriodValueAcquisitions",)),
    ("shares_weighted", "flow", ("WeightedAverageNumberOfSharesOutstandingBasic",)),
    ("cash", "stock", ("CashAndCashEquivalentsAtCarryingValue",)),
    # NVIDIA moved from MarketableSecuritiesCurrent to DebtSecuritiesCurrent
    # in fiscal 2027; a missing tag reads as a cash crash (staging, 2026-10-08).
    ("sti", "stock", ("ShortTermInvestments", "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
                      "MarketableSecuritiesCurrent", "DebtSecuritiesCurrent")),
    ("debt", "stock", ("LongTermDebt", "LongTermDebtNoncurrent", "DebtInstrumentCarryingAmount")),
    ("debt_current", "stock", ("LongTermDebtCurrent", "DebtCurrent")),
    # Debt filed by instrument instead of as a total (REITs: Realty Income
    # tags NotesPayable, LoansPayable, CommercialPaper; eval 2026-10-09).
    # Used only when no total above is tagged, so never double counted.
    ("notes_payable", "stock", ("NotesPayable", "SeniorNotes", "UnsecuredDebt")),
    ("loans_payable", "stock", ("LoansPayable",)),
    ("secured_debt", "stock", ("SecuredDebt",)),
    ("commercial_paper", "stock", ("CommercialPaper",)),
    ("credit_line", "stock", ("LineOfCredit", "LongTermLineOfCredit")),
    ("derivative_liabilities", "stock", ("DerivativeLiabilities", "DerivativeLiabilitiesNoncurrent")),
    ("goodwill", "stock", ("Goodwill",)),
    ("intangibles", "stock", ("FiniteLivedIntangibleAssetsNet", "IntangibleAssetsNetExcludingGoodwill")),
    ("equity", "stock", ("StockholdersEquity",
                         "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")),
    ("liabilities", "stock", ("Liabilities",)),
]

# Duration classes by span in days: a quarter, a half, nine months, a year.
_SPANS = {"Q": (60, 115), "H": (150, 200), "9M": (240, 290), "FY": (300, 400)}
# Per-share and share-count items are not additive: never derive their quarters.
NON_ADDITIVE = {"eps_basic", "eps_diluted", "shares_weighted"}


@dataclass
class Value:
    value: float
    unit: str
    filed: str
    accn: str
    tag: str
    derivation: str | None = None


@dataclass
class Statements:
    """One company's filings as known on ``as_of``."""

    cik: str
    as_of: str
    fy_end: tuple[int, int] | None             # (month, day) from submissions
    quarter_ends: list[str]                    # oldest first
    year_ends: list[str]                       # fiscal year ends, oldest first
    quarters: dict[str, dict[str, Value]] = field(default_factory=dict)   # concept → end → value
    years: dict[str, dict[str, Value]] = field(default_factory=dict)
    cover_shares: Value | None = None
    cover_shares_date: str | None = None
    submissions: dict = field(default_factory=dict)


def _span_class(fact: dict) -> str | None:
    if "start" not in fact:
        return None
    days = (date.fromisoformat(fact["end"]) - date.fromisoformat(fact["start"])).days
    return next((name for name, (lo, hi) in _SPANS.items() if lo <= days <= hi), None)


def _known(facts: dict, tag: str, as_of: str) -> tuple[dict[tuple[str | None, str], dict], str]:
    """{(start, end): latest fact filed on or before ``as_of``} for one tag, in
    its first reported unit."""
    units = (facts.get(tag) or {}).get("units") or {}
    if not units:
        return {}, ""
    unit = "USD" if "USD" in units else next(iter(units))
    latest: dict[tuple[str | None, str], dict] = {}
    for fact in units[unit]:
        if fact["filed"] > as_of or not str(fact.get("form", "")).startswith(("10-Q", "10-K")):
            continue
        key = (fact.get("start"), fact["end"])
        seen = latest.get(key)
        if seen is None or fact["filed"] >= seen["filed"]:
            latest[key] = fact
    return latest, unit


def _val(fact: dict, unit: str, tag: str, derivation: str | None = None, value: float | None = None) -> Value:
    return Value(
        value=float(fact["val"] if value is None else value), unit=unit, filed=fact["filed"],
        accn=fact.get("accn", ""), tag=tag, derivation=derivation,
    )


def _prev_quarter_end(end: str, ends: list[str]) -> str | None:
    earlier = [e for e in ends if e < end]
    return earlier[-1] if earlier else None


def _flow_quarter(known: dict, unit: str, tag: str, end: str, quarter_ends: list[str],
                  additive: bool) -> Value | None:
    """The 3-month value ending ``end``: as filed, else derived from to-date figures."""
    by_end = [(start, fact) for (start, e), fact in known.items() if e == end and start]
    direct = [fact for start, fact in by_end if _span_class(fact) == "Q"]
    if direct:
        return _val(max(direct, key=lambda f: f["filed"]), unit, tag)
    if not additive:
        return None
    prev = _prev_quarter_end(end, quarter_ends)
    if prev is None:
        return None
    # A to-date figure ending here, less the same-start figure ending at the
    # previous quarter end. Q4 = FY − 9M is this rule with a fiscal-year start.
    for start, fact in sorted(by_end, key=lambda sf: sf[0]):
        cls = _span_class(fact)
        if cls not in ("H", "9M", "FY"):
            continue
        before = known.get((start, prev))
        if before is None:
            continue
        label = {"H": "6M − 3M", "9M": "9M − 6M", "FY": "FY − 9M"}[cls]
        derived_from = f"{label} (year to {end} less year to {prev})"
        newer = fact if fact["filed"] >= before["filed"] else before
        return _val(newer, unit, tag, derivation=derived_from, value=fact["val"] - before["val"])
    return None


def _flow_year(known: dict, unit: str, tag: str, end: str) -> Value | None:
    years = [fact for (start, e), fact in known.items()
             if e == end and start and _span_class(fact) == "FY" and fact.get("form", "").startswith("10-K")]
    if not years:
        years = [fact for (start, e), fact in known.items() if e == end and start and _span_class(fact) == "FY"]
    return _val(max(years, key=lambda f: f["filed"]), unit, tag) if years else None


def _stock_at(known: dict, unit: str, tag: str, end: str) -> Value | None:
    fact = known.get((None, end))
    return _val(fact, unit, tag) if fact else None


def _period_ends(us_gaap: dict, as_of: str) -> tuple[list[str], list[str]]:
    """Quarter ends and fiscal year ends the filer has reported, from its
    headline flow items (revenue, net income, operating income)."""
    quarter_ends: set[str] = set()
    year_ends: set[str] = set()
    for tag in ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "NetIncomeLoss",
                "OperatingIncomeLoss"):
        known, _ = _known(us_gaap, tag, as_of)
        for (start, end), fact in known.items():
            cls = _span_class(fact) if start else None
            if cls in ("Q", "H", "9M", "FY"):
                quarter_ends.add(end)
            if cls == "FY" and fact.get("form", "").startswith("10-K"):
                year_ends.add(end)
    return sorted(quarter_ends), sorted(year_ends)


def load(ticker: str, as_of: str, n_quarters: int = 8, n_years: int = 2) -> Statements | None:
    """The filer's statements as known on ``as_of``, or None for a non-filer."""
    cik = cik_for(ticker)
    if cik is None:
        return None
    facts = _cached_json(_FACTS_URL.format(cik=cik), f"CIK{cik}.json")
    try:
        submissions = _cached_json(_SUBMISSIONS_URL.format(cik=cik), f"submissions-CIK{cik}.json")
    except Exception:  # noqa: BLE001 — identity and calendar degrade, statements don't
        submissions = {}
    return from_json(cik, facts, submissions, as_of, n_quarters, n_years)


def _fy_end(submissions: dict) -> tuple[int, int] | None:
    raw = str(submissions.get("fiscalYearEnd") or "")
    if len(raw) == 4 and raw.isdigit():
        return int(raw[:2]), int(raw[2:])
    return None


def from_json(cik: str, facts: dict, submissions: dict, as_of: str,
              n_quarters: int = 8, n_years: int = 2) -> Statements | None:
    us_gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    if not us_gaap:
        return None
    all_quarter_ends, all_year_ends = _period_ends(us_gaap, as_of)
    quarter_ends = all_quarter_ends[-n_quarters:]
    year_ends = all_year_ends[-n_years:]
    out = Statements(cik=cik, as_of=as_of, fy_end=_fy_end(submissions), quarter_ends=quarter_ends,
                     year_ends=year_ends, submissions=submissions)
    for concept, kind, tags in LINES:
        quarters: dict[str, Value] = {}
        years: dict[str, Value] = {}
        for tag in tags:
            known, unit = _known(us_gaap, tag, as_of)
            if not known:
                continue
            for end in quarter_ends:
                if end in quarters:
                    continue
                value = (_stock_at(known, unit, tag, end) if kind == "stock"
                         else _flow_quarter(known, unit, tag, end, all_quarter_ends, concept not in NON_ADDITIVE))
                if value is not None:
                    quarters[end] = value
            for end in year_ends:
                if end in years:
                    continue
                value = _stock_at(known, unit, tag, end) if kind == "stock" else _flow_year(known, unit, tag, end)
                if value is not None:
                    years[end] = value
        out.quarters[concept] = quarters
        out.years[concept] = years

    dei = (facts.get("facts") or {}).get("dei") or {}
    cover = [f for f in ((dei.get("EntityCommonStockSharesOutstanding") or {}).get("units") or {}).get("shares", [])
             if f["filed"] <= as_of]
    if cover:
        latest = max(cover, key=lambda f: (f["filed"], f["end"]))
        out.cover_shares = _val(latest, "shares", "dei:EntityCommonStockSharesOutstanding")
        out.cover_shares_date = latest["end"]
        _rescale_thousands(out, [(f["end"], float(f["val"])) for f in cover])
    return out


def _rescale_thousands(st: Statements, cover: list[tuple[str, float]]) -> None:
    """Some filers tag share counts in thousands in some filings (ONDS's Q1 2026
    10-Q: 445,089 weighted shares beside 495.8M on its cover). A weighted count
    under 1% of the nearest cover-page count is read as thousands and scaled,
    with the derivation saying so."""
    def nearest_cover(end: str) -> float | None:
        if not cover:
            return None
        return min(cover, key=lambda c: abs((date.fromisoformat(c[0]) - date.fromisoformat(end)).days))[1]

    for table in (st.quarters, st.years):
        for end, v in (table.get("shares_weighted") or {}).items():
            ref = nearest_cover(end)
            if ref and 0 < v.value < ref / 100:
                v.derivation = f"filed as {v.value:,.0f}; scaled ×1,000 (tagged in thousands by the filer)"
                v.value *= 1000


# ---- fiscal calendar ----------------------------------------------------------


def calendar_quarter(start: date, end: date) -> str:
    """The calendar quarter a fiscal quarter mostly falls in, e.g. "2026Q2":
    NVIDIA's Apr 27 – Jul 26 quarter is calendar 2026Q2, not Q3."""
    mid = start + (end - start) / 2
    return f"{mid.year}Q{(mid.month - 1) // 3 + 1}"


def fiscal_label(end: date, fy_end: tuple[int, int] | None) -> tuple[int, int] | None:
    """(fiscal year, fiscal quarter) of a quarter ending ``end``. The fiscal
    year is named for the calendar year it ends in (NVIDIA's year ending
    January 2027 is FY2027). 52/53-week years end within a week of the
    nominal date."""
    if fy_end is None:
        return None
    month, day = fy_end
    day = min(day, 28) if month == 2 else day
    fy = end.year
    nominal = date(fy, month, day)
    if end > nominal + timedelta(days=7):
        fy += 1
        nominal = date(fy, month, day)
    quarters_left = round((nominal - end).days / 91.3)
    quarter = 4 - quarters_left
    if not 1 <= quarter <= 4:
        return None
    return fy, quarter


def quarter_start(end: str, ends: list[str]) -> date:
    prev = _prev_quarter_end(end, ends)
    if prev is not None:
        return date.fromisoformat(prev) + timedelta(days=1)
    return date.fromisoformat(end) - timedelta(days=90)


def filings(submissions: dict, forms: tuple[str, ...] = ("10-Q", "10-K")) -> list[dict]:
    """Recent periodic filings, newest first: form, filed, period, accession, document."""
    recent = (submissions.get("filings") or {}).get("recent") or {}
    rows = []
    for i, form in enumerate(recent.get("form") or []):
        if form in forms:
            rows.append({
                "form": form,
                "filed": recent["filingDate"][i],
                "period": recent["reportDate"][i],
                "accn": recent["accessionNumber"][i],
                "document": recent["primaryDocument"][i],
            })
    return rows


def get_fundamentals_overview(ticker: str, curr_date: str | None = None) -> str:
    """The ``get_fundamentals`` tool when fundamentals come from SEC EDGAR only
    (REPORT_QUALITY_PLAN § Data sources): the overview is the fact sheet the
    stage already has. Third-party overview stats (consensus, analyst targets,
    profile ratios) are not given to agents."""
    return (
        f"Company overview for {ticker.upper()}: use the fact sheet in your instructions. It holds the "
        "company's identity, its statements as filed with the SEC, and market cap, enterprise value and "
        "multiples computed from them. Use get_income_statement, get_balance_sheet and get_cashflow for "
        "longer filed history. Consensus estimates, analyst price targets and third-party profile "
        "statistics are not available on this platform; do not cite or estimate them."
    )


# ---- business description (10-K Item 1) ---------------------------------------

# Small caps can extract with stray spaces ("ITEM 1. B USINESS", Microsoft).
_ITEM1 = re.compile(
    r"items?\s*1\s*(?:and\s*2\s*)?[.:—–-]?\s*b\s?u\s?s\s?i\s?n\s?e\s?s\s?s\b",  # "Items 1 and 2." (Exxon)
    re.IGNORECASE,
)
_ITEM1A = re.compile(r"item\s*1a\s*[.:—–-]?\s*risk\s+factors", re.IGNORECASE)


def _plain_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|ix:header)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = unescape(text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def item1_excerpt(html: str, limit: int = 2500) -> str | None:
    """The opening of a 10-K's Item 1 (Business), past the table of contents.

    A contents entry ("Item 1. Business 4 Item 1A. Risk Factors 12") is
    followed closely by Item 1A; the real section runs thousands of
    characters before it. The first heading with a long run wins."""
    text = _plain_text(html)
    for m in _ITEM1.finditer(text):
        nxt = _ITEM1A.search(text, m.end())
        run = (nxt.start() if nxt else len(text)) - m.end()
        # Contents pages list the next items within a line or two (page
        # headers repeating "Item 1" inside the section don't count).
        if run < 2000 or re.search(r"\bitem\s*(1a|1b|1c|2)\b", text[m.end(): m.end() + 300], re.IGNORECASE):
            continue
        body = text[m.end(): m.end() + limit * 2].lstrip(" .:-")
        body = re.sub(r"^(Overview|General|Our Company|Company Overview)\b\s*", "", body, flags=re.IGNORECASE)
        cut = body[:limit]
        end = cut.rfind(". ")
        return cut[: end + 1] if end > limit // 2 else cut
    return None


def latest_annual_report(submissions: dict, as_of: str) -> dict | None:
    """The newest 10-K filed on or before ``as_of``."""
    return next((r for r in filings(submissions, ("10-K",)) if r["filed"] <= as_of), None)


def annual_report_url(cik: str, filing: dict) -> str:
    return (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{filing['accn'].replace('-', '')}/{filing['document']}")


def fetch_item1(cik: str, filing: dict) -> str | None:
    """Item 1 excerpt for a filing, cached by accession number."""
    import requests

    from tradingagents.dataflows.vendors.sec_edgar import _user_agent

    path = Path(get_config()["data_cache_dir"]) / "sec_edgar" / f"item1-{filing['accn']}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8") or None
    response = requests.get(annual_report_url(cik, filing), headers={"User-Agent": _user_agent()}, timeout=60)
    response.raise_for_status()
    excerpt = item1_excerpt(response.text) or ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(excerpt, encoding="utf-8")
    return excerpt or None


# ---- segments (dimensional XBRL in the filing's instance document) ----------------

_SEGMENT_AXES = {
    "StatementBusinessSegmentsAxis": "segment",
    "ProductOrServiceAxis": "product",
}
_SEGMENT_CONCEPTS = {
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "OperatingIncomeLoss": "operating_income",
}


def _humanize(member: str) -> str:
    """"nvda:ComputeAndNetworkingMember" → "Compute and Networking"."""
    name = member.split(":")[-1]
    name = re.sub(r"(Segment)?Member$", "", name)
    words = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\b|\d)|[A-Z]?[a-z]+|\d+", name)
    small = {"And": "and", "Of": "of", "For": "for", "The": "the", "In": "in"}
    return " ".join(small.get(w, w) for i, w in enumerate(words)) or name


def parse_labels(label_xml: str) -> dict[str, str]:
    """{"nvda:DataCenterMember": "Data Center"} from a filing's label linkbase."""
    import xml.etree.ElementTree as ET

    xlink = "{http://www.w3.org/1999/xlink}"
    root = ET.fromstring(label_xml)
    locs, arcs, texts = {}, {}, {}
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "loc":
            href = el.get(f"{xlink}href", "")
            locs[el.get(f"{xlink}label")] = href.split("#")[-1].replace("_", ":", 1)
        elif tag == "labelArc":
            arcs.setdefault(el.get(f"{xlink}from"), []).append(el.get(f"{xlink}to"))
        elif tag == "label" and el.text:
            role = el.get(f"{xlink}role", "")
            # The plain label wins over terse/verbose variants.
            if role.endswith("/label") or el.get(f"{xlink}label") not in texts:
                texts[el.get(f"{xlink}label")] = el.text.strip()
    out = {}
    for loc, concept in locs.items():
        for to in arcs.get(loc, []):
            if to in texts:
                out.setdefault(concept, re.sub(r"\s*\[Member\]$", "", texts[to]))
    return out


def parse_segments(instance_xml: str, quarter_ends: list[str], labels: dict | None = None) -> list[dict]:
    """[{axis, member, label, concept, end, value}] for 3-month periods ending
    on ``quarter_ends``, one dimension (a segment or a product line) each."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(instance_xml)
    contexts: dict[str, tuple[str, str, str]] = {}
    for ctx in root.iter():
        if not ctx.tag.endswith("}context"):
            continue
        members = [m for m in ctx.iter() if m.tag.endswith("}explicitMember")]
        # One segment or product line, optionally marked as an operating
        # segment (how NVIDIA tags its segment revenue); anything else is a
        # finer cut (customers, geographies) left out.
        members = [m for m in members if not (
            m.get("dimension", "").endswith("ConsolidationItemsAxis")
            and (m.text or "").strip().endswith("OperatingSegmentsMember"))]
        if len(members) != 1:
            continue
        axis = members[0].get("dimension", "").split(":")[-1]
        if axis not in _SEGMENT_AXES:
            continue
        start = next((p.text for p in ctx.iter() if p.tag.endswith("}startDate")), None)
        end = next((p.text for p in ctx.iter() if p.tag.endswith("}endDate")), None)
        if not start or not end or end not in quarter_ends:
            continue
        days = (date.fromisoformat(end) - date.fromisoformat(start)).days
        span = "Q" if 60 <= days <= 115 else "FY" if 300 <= days <= 400 else None
        if span is None:
            continue
        contexts[ctx.get("id")] = (_SEGMENT_AXES[axis], (members[0].text or "").strip(), end, span)
    rows: dict[tuple, dict] = {}
    for el in root:
        local = el.tag.split("}")[-1]
        concept = _SEGMENT_CONCEPTS.get(local)
        ctx = contexts.get(el.get("contextRef", ""))
        if concept is None or ctx is None or el.text is None:
            continue
        try:
            value = float(el.text)
        except ValueError:
            continue
        axis, member, end, span = ctx
        key = (axis, member, concept, end, span)
        # First tag wins (Revenues before the ASC 606 tag), as in the statements.
        rows.setdefault(key, {"axis": axis, "member": member,
                              "label": (labels or {}).get(member) or _humanize(member),
                              "concept": concept, "end": end, "span": span, "value": value})
    return sorted(rows.values(), key=lambda r: (r["axis"], r["concept"], r["end"], -abs(r["value"])))


def fetch_segments(cik: str, filing: dict, quarter_ends: list[str]) -> list[dict]:
    """Segment rows from a 10-Q/10-K's XBRL instance, cached by accession."""
    import json

    import requests

    from tradingagents.dataflows.vendors.sec_edgar import _user_agent

    path = Path(get_config()["data_cache_dir"]) / "sec_edgar" / f"segments-{filing['accn']}.json"
    if path.exists():
        return [r for r in json.loads(path.read_text(encoding="utf-8")) if r["end"] in quarter_ends]
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{filing['accn'].replace('-', '')}/"
    instance = re.sub(r"\.htm$", "_htm.xml", filing["document"])
    response = requests.get(base + instance, headers={"User-Agent": _user_agent()}, timeout=60)
    response.raise_for_status()
    labels = {}
    try:  # a separate label linkbase, where the filer publishes one
        lab = requests.get(base + re.sub(r"\.htm$", "_lab.xml", filing["document"]),
                           headers={"User-Agent": _user_agent()}, timeout=60)
        if lab.ok:
            labels = parse_labels(lab.text)
    except Exception:  # noqa: BLE001 — names fall back to the member's code
        labels = {}
    rows = parse_segments(response.text, quarter_ends, labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")
    return rows
