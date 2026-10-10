"""The fact sheet against ONDS as filed (REPORT_QUALITY_PLAN R4, Appendix B).

Fixtures are trimmed copies of SEC companyfacts and submissions for CIK
0001646188 fetched 2026-10-08, and the run's OHLCV through 2026-10-02.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from tradingagents.quality import edgar_ext, facts
from tradingagents.quality.session import clock, us_holidays, us_session

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def onds():
    companyfacts = json.loads((FIX / "onds_companyfacts.json").read_text())
    submissions = json.loads((FIX / "onds_submissions.json").read_text())
    st = edgar_ext.from_json("0001646188", companyfacts, submissions, "2026-10-05")
    ohlcv = pd.read_csv(FIX / "onds_ohlcv.csv")
    return facts.build("ONDS", "2026-10-05", "2026-10-05T10:33:00Z", statements=st, ohlcv=ohlcv, offline=True)


def _m(sheet, key):
    return round(sheet.value(key) / 1e6, 1)


@pytest.mark.unit
def test_quarters_reproduce_appendix_b(onds):
    expected = {  # $M, Q2 '25 … Q2 '26
        "revenue": [6.3, 10.1, 30.1, 50.1, 83.8],
        "gross_profit": [3.3, 2.6, 12.7, 24.7, 36.1],
        "rnd": [4.2, 4.5, 8.7, 13.5, 31.0],
        "sm": [2.3, 3.0, 5.5, 10.5, 20.9],
        "ga": [6.1, 10.6, 21.9, 43.3, 128.0],
        "amortization": [1.1, 1.1, 2.6, 5.6, 18.7],
        "operating_income": [-9.2, -15.5, -23.3, -42.7, -162.9],
        "non_operating": [-1.5, 8.3, -77.5, 404.2, 44.2],
        # Q4 '25 is NetIncomeLoss (attributable to Ondas) FY − 9M = −99.9; the
        # appendix's −101.0 used ProfitLoss, which includes minority interests.
        "net_income": [-10.8, -7.5, -99.9, 362.8, -88.4],
        "sbc": [2.2, 5.5, 6.8, 19.7, 69.1],
        "ocf": [-8.4, -11.0, -12.7, -51.3, -86.1],
        "capex": [0.1, 0.2, 1.6, 1.3, 7.8],
    }
    quarters = ["2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2"]
    assert [c.calendar for c in onds.quarters] == quarters
    for concept, values in expected.items():
        assert [_m(onds, f"{concept}.{q}") for q in quarters] == values, concept


@pytest.mark.unit
def test_balances_and_share_issuance(onds):
    assert _m(onds, "cash.2026Q2") == 657.9
    assert _m(onds, "sti.2026Q2") == 726.6
    assert _m(onds, "derivative_liabilities.2026Q2") == 1043.7
    assert _m(onds, "cash_sti.2026Q2") == 1384.5
    assert onds.get("cash.2026Q2").period == "at 2026-06-30"
    assert _m(onds, "stock_for_acquisitions.2026Q2") == 509.0
    assert _m(onds, "stock_sold_cash.2026Q1") == 959.1
    assert onds.get("debt.2026Q2") is None  # no debt tagged: EV counts 0 and says so


@pytest.mark.unit
def test_derived_figures_match_appendix_b(onds):
    assert _m(onds, "ttm_revenue.2026Q2") == 174.1
    assert round(onds.value("revenue_yoy.2026Q2")) == 1235
    assert _m(onds, "fcf.2026Q2") == -93.8
    assert onds.value("shares.cover") == 570552341
    assert "no debt tagged" in onds.get("ev").derivation
    assert round(onds.value("ev_sales.ttm"), 1) == round(onds.value("ev") / 174.1e6, 1)
    assert onds.get("revenue.2025Q4").derivation.startswith("FY − 9M")
    assert onds.get("ocf.2026Q2").derivation.startswith("6M − 3M")
    assert onds.get("revenue.2026Q2").derivation is None  # filed as a quarter


@pytest.mark.unit
def test_share_counts_tagged_in_thousands_are_scaled_and_marked(onds):
    q1 = onds.get("shares_weighted.2026Q1")
    assert 400e6 < q1.value < 500e6
    assert "thousands" in q1.derivation


@pytest.mark.unit
def test_flags_name_the_onds_hazards(onds):
    text = " ".join(onds.flags)
    assert "money moved into investments" in text          # cash −$368M, cash+STI −$89M
    assert "opposite signs" in text                         # TTM: warrant gains vs operating loss
    assert "dilution" in text


@pytest.mark.unit
def test_next_earnings_is_estimated_from_last_years_filing(onds):
    nxt = onds.calendar["earnings_next"]
    assert nxt["status"] == "estimated"
    assert nxt["date"] == "2026-11-12"
    assert nxt["covers"] == "2026Q3"
    assert onds.value("earnings.next") == "2026-11-12"  # citable, as the render cites it
    assert "2026Q2" in nxt["already_reported"]


@pytest.mark.unit
def test_technicals_from_the_runs_own_closes(onds):
    assert onds.value("macd.cross") == "bullish"
    assert onds.value("macd.cross_date") == "2026-09-21"
    assert onds.value("macd.position") == "above signal"
    assert onds.value("sma200.gap_usd") < 0


@pytest.mark.unit
def test_render_carries_keys_and_marks_past_quarters(onds):
    text = facts.render(onds)
    assert "| Revenue [revenue] |" in text
    assert "[F:ev]" in text and "[F:macd.cross]" in text
    assert "PAST results" in text
    assert len(text) < 40_000  # the whole sheet goes to every stage


@pytest.mark.unit
def test_nvidia_quarters_carry_calendar_and_fiscal_labels():
    from datetime import date

    start, end = date(2026, 4, 27), date(2026, 7, 26)
    assert edgar_ext.calendar_quarter(start, end) == "2026Q2"
    assert edgar_ext.fiscal_label(end, (1, 25)) == (2027, 2)
    assert edgar_ext.fiscal_label(date(2026, 1, 25), (1, 25)) == (2026, 4)


@pytest.mark.unit
def test_build_never_raises_without_data():
    sheet = facts.build("ZZZZ", "2026-10-05", None, statements=None, ohlcv=pd.DataFrame(), offline=True)
    assert sheet.unavailable and sheet.facts == []


# ---- session clock ---------------------------------------------------------------


@pytest.mark.unit
def test_onds_was_written_pre_market_monday():
    c = clock("2026-10-05T10:33:00Z", "stock", "2026-10-02", us_listed=True)
    assert c["state"] == "pre-market"
    assert c["last_complete_session"] == "2026-10-02"
    assert "Fri 2026-10-02 close" in c["text"]


@pytest.mark.unit
@pytest.mark.parametrize("utc, state, last", [
    ("2026-10-06T15:00:00Z", "market open", "2026-10-05"),        # Tue 11:00 ET
    ("2026-10-06T21:30:00Z", "after the close", "2026-10-06"),    # Tue 17:30 ET
    ("2026-10-10T16:00:00Z", "market closed (weekend)", "2026-10-09"),
    ("2026-11-26T16:00:00Z", "market closed (holiday)", "2026-11-25"),  # Thanksgiving
    ("2026-04-03T14:00:00Z", "market closed (holiday)", "2026-04-02"),  # Good Friday 2026
])
def test_session_states(utc, state, last):
    s = us_session(datetime.fromisoformat(utc.replace("Z", "+00:00")).astimezone(UTC))
    assert s["state"] == state
    assert s["last_complete"].isoformat() == last


@pytest.mark.unit
def test_holidays_2026():
    from datetime import date

    h = us_holidays(2026)
    for d in ("2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
              "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"):
        assert date.fromisoformat(d) in h, d
    assert len(h) == 10


@pytest.mark.unit
def test_open_market_says_today_is_not_a_completed_bar():
    c = clock("2026-10-06T15:00:00Z", "stock", "2026-10-05", us_listed=True)
    assert "still trading" in c["text"]


@pytest.mark.unit
def test_crypto_has_no_session():
    c = clock("2026-10-10T16:00:00Z", "crypto", "2026-10-09", us_listed=False)
    assert "around the clock" in c["text"]


@pytest.mark.unit
def test_every_stage_gets_the_sheet_and_the_cite_rule(onds):
    from tradingagents.agents.context import get_instrument_context_from_state
    from tradingagents.quality.prompts import CITE_RULE

    state = {"company_of_interest": "ONDS", "instrument_context": "ONDS is Ondas Inc.",
             "fact_sheet_text": facts.render(onds)}
    context = get_instrument_context_from_state(state)
    assert context.startswith("ONDS is Ondas Inc.")
    assert "[F:ev]" in context and CITE_RULE in context
    bare = get_instrument_context_from_state({"company_of_interest": "ONDS", "instrument_context": "x"})
    assert CITE_RULE not in bare


@pytest.mark.unit
def test_the_state_log_keeps_the_fact_sheet(tmp_path):
    """Staging, 2026-10-08: every stage cited the sheet, but the state log
    dropped it, so the report could not render the numbers they cited."""
    from tests.test_cli_display import _bare_graph, _state

    state = {**_state("ONDS"), "fact_sheet": {"version": 1, "facts": [{"key": "ev", "value": 2.75e9}]}}
    _bare_graph(tmp_path)._log_state("2026-10-05", state)
    logged = json.loads(next(tmp_path.rglob("full_states_log*.json")).read_text(encoding="utf-8"))
    assert logged["fact_sheet"]["facts"][0]["key"] == "ev"


# ---- business description (R4b) ------------------------------------------------

_10K = (
    "<html><body><p>Table of Contents</p><p>Item 1. Business 4</p><p>Item 1A. Risk Factors 18</p>"
    "<p>PART I</p><p>ITEM 1. B<span>USINESS</span></p><p>Overview</p>"
    "<p>Ondas Holdings provides private wireless data networks and autonomous drone systems "
    "for defense, homeland security and industrial customers. " + "It sells to rail operators. " * 120
    + "</p><p>Item 1A. Risk Factors</p></body></html>"
)


@pytest.mark.unit
def test_item1_skips_the_contents_page_and_reads_small_caps():
    excerpt = edgar_ext.item1_excerpt(_10K)
    assert excerpt.startswith("Ondas Holdings provides private wireless")
    assert len(excerpt) <= 2500


@pytest.mark.unit
def test_description_is_summarised_once_per_10k(tmp_path, monkeypatch):
    from tradingagents.dataflows.config import get_config, set_config

    config = get_config()
    config["data_cache_dir"] = str(tmp_path)
    set_config(config)
    companyfacts = json.loads((FIX / "onds_companyfacts.json").read_text())
    submissions = json.loads((FIX / "onds_submissions.json").read_text())
    st = edgar_ext.from_json("0001646188", companyfacts, submissions, "2026-10-05")
    monkeypatch.setattr(edgar_ext, "fetch_item1", lambda cik, filing: edgar_ext.item1_excerpt(_10K))
    prompts = []

    def describe(prompt):
        prompts.append(prompt)
        return "Ondas makes private wireless networks and drone systems for defense and rail."

    ohlcv = pd.read_csv(FIX / "onds_ohlcv.csv")
    for _ in range(2):
        sheet = facts.build("ONDS", "2026-10-05", "2026-10-05T10:33:00Z", statements=st, ohlcv=ohlcv,
                            news_text="", describe=describe)
    assert len(prompts) == 1  # cached by accession
    assert "Ondas Holdings provides private wireless" in prompts[0]
    assert sheet.identity["description"].startswith("Ondas makes")
    assert "0001213900-26-035981" in sheet.identity["description_source"]
    assert "Business (from the 10-K" in facts.render(sheet)


# ---- segments (R4b) -------------------------------------------------------------

_INSTANCE = """<?xml version="1.0"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:xbrldi="http://xbrl.org/2006/xbrldi"
  xmlns:us-gaap="http://fasb.org/us-gaap/2025">
 <xbrli:context id="c1"><xbrli:entity><xbrli:segment>
   <xbrldi:explicitMember dimension="us-gaap:StatementBusinessSegmentsAxis">nvda:ComputeAndNetworkingMember</xbrldi:explicitMember>
   <xbrldi:explicitMember dimension="srt:ConsolidationItemsAxis">us-gaap:OperatingSegmentsMember</xbrldi:explicitMember>
 </xbrli:segment></xbrli:entity><xbrli:period><xbrli:startDate>2026-04-27</xbrli:startDate><xbrli:endDate>2026-07-26</xbrli:endDate></xbrli:period></xbrli:context>
 <xbrli:context id="c0"><xbrli:entity><xbrli:segment>
   <xbrldi:explicitMember dimension="us-gaap:StatementBusinessSegmentsAxis">nvda:ComputeAndNetworkingMember</xbrldi:explicitMember>
   <xbrldi:explicitMember dimension="srt:ConsolidationItemsAxis">us-gaap:OperatingSegmentsMember</xbrldi:explicitMember>
 </xbrli:segment></xbrli:entity><xbrli:period><xbrli:startDate>2025-04-28</xbrli:startDate><xbrli:endDate>2025-07-27</xbrli:endDate></xbrli:period></xbrli:context>
 <xbrli:context id="geo"><xbrli:entity><xbrli:segment>
   <xbrldi:explicitMember dimension="us-gaap:StatementBusinessSegmentsAxis">nvda:ComputeAndNetworkingMember</xbrldi:explicitMember>
   <xbrldi:explicitMember dimension="srt:StatementGeographicalAxis">country:US</xbrldi:explicitMember>
 </xbrli:segment></xbrli:entity><xbrli:period><xbrli:startDate>2026-04-27</xbrli:startDate><xbrli:endDate>2026-07-26</xbrli:endDate></xbrli:period></xbrli:context>
 <us-gaap:Revenues contextRef="c1" unitRef="usd" decimals="-6">88299000000</us-gaap:Revenues>
 <us-gaap:Revenues contextRef="c0" unitRef="usd" decimals="-6">41331000000</us-gaap:Revenues>
 <us-gaap:Revenues contextRef="geo" unitRef="usd" decimals="-6">1</us-gaap:Revenues>
</xbrli:xbrl>"""


@pytest.mark.unit
def test_segments_parse_with_year_ago_change():
    from tradingagents.quality.facts import QuarterCol, segment_facts

    rows = edgar_ext.parse_segments(_INSTANCE, ["2026-07-26", "2025-07-27"],
                                    {"nvda:ComputeAndNetworkingMember": "Compute & Networking"})
    assert [(r["label"], r["value"]) for r in rows] == [
        ("Compute & Networking", 41331000000.0), ("Compute & Networking", 88299000000.0)]  # geography cut dropped
    cols = [QuarterCol(end="2025-07-27", calendar="2025Q2", fiscal="Q2 FY2026"),
            QuarterCol(end="2026-07-26", calendar="2026Q2", fiscal="Q2 FY2027")]
    facts_ = {f.key: f.value for f in segment_facts(rows, cols, [], "2026-08-26 x")}
    assert facts_["seg_revenue.compute_networking.2026Q2"] == 88299000000.0
    assert round(facts_["seg_revenue_yoy.compute_networking.2026Q2"]) == 114


@pytest.mark.unit
def test_member_names_without_a_label_file():
    assert edgar_ext._humanize("nvda:ComputeAndNetworkingMember") == "Compute and Networking"
    assert edgar_ext._humanize("onds:ProductRevenueMember") == "Product Revenue"


@pytest.mark.unit
def test_a_missing_investments_tag_is_not_a_cash_crash():
    """Staging NVDA: a renamed tag dropped $39B of investments from one
    quarter; cash + investments must not be computed for that quarter."""
    companyfacts = json.loads((FIX / "onds_companyfacts.json").read_text())
    submissions = json.loads((FIX / "onds_submissions.json").read_text())
    st = edgar_ext.from_json("0001646188", companyfacts, submissions, "2026-10-05")
    del st.quarters["sti"]["2026-06-30"]
    sheet = facts.build("ONDS", "2026-10-05", None, statements=st, ohlcv=pd.read_csv(FIX / "onds_ohlcv.csv"),
                        offline=True)
    assert sheet.get("cash_sti.2026Q1") is not None
    assert sheet.get("cash_sti.2026Q2") is None



@pytest.mark.unit
def test_prices_render_with_cents(onds):
    """Staging ONDS, 2026-10-08: "$7" for a $6.85 close put every stage on
    rounded prices."""
    text = facts.render(onds)
    close = onds.value("price.close")
    assert f"[F:price.close] ${close:,.2f}" in text


@pytest.mark.unit
def test_tangible_equity_is_on_the_sheet(onds):
    gw, intang, eq = (onds.value(f"{c}.2026Q2") for c in ("goodwill", "intangibles", "equity"))
    assert onds.value("goodwill_intangibles.2026Q2") == pytest.approx(gw + intang)
    assert onds.value("tangible_equity.2026Q2") == pytest.approx(eq - gw - intang)
    assert "Tangible equity (equity − goodwill − intangibles) [tangible_equity]" in facts.render(onds)


@pytest.mark.unit
def test_debt_filed_by_instrument_is_summed_into_ev():
    """Realty Income files no total-debt tag; its notes, loans and commercial
    paper (~$29.3B at 2026Q2) were counted as zero debt (eval 2026-10-09)."""
    st = edgar_ext.from_json("0000726728", json.loads((FIX / "o_companyfacts.json").read_text()),
                             json.loads((FIX / "o_submissions.json").read_text()), "2026-10-09")
    days = pd.date_range("2025-09-01", periods=280, freq="B").strftime("%Y-%m-%d")
    ohlcv = pd.DataFrame({"Date": days, "Open": 54.0, "High": 55.0, "Low": 53.0, "Close": 54.17, "Volume": 1e6})
    sheet = facts.build("O", "2026-10-09", "2026-10-09T14:00:00Z", statements=st, ohlcv=ohlcv, offline=True)
    assert 29_000 < _m(sheet, "debt_total.2026Q2") < 29_500
    ev = sheet.value("ev")
    assert ev == pytest.approx(sheet.value("market_cap") - sheet.value("cash_sti.2026Q2") + sheet.value("debt_total.2026Q2"))
    assert "no debt tagged" not in sheet.get("ev").derivation


@pytest.mark.unit
def test_a_debt_total_tag_is_never_summed_with_instruments(onds):
    assert onds.get("debt_total.2026Q2") is None


# ---- release review 2026-10-09: EV without a debt tag, stale cover counts, 20-F filers ----------


QUARTERS = [("2025-01-01", "2025-03-31"), ("2025-04-01", "2025-06-30"), ("2025-07-01", "2025-09-30"),
            ("2025-10-01", "2025-12-31"), ("2026-01-01", "2026-03-31")]


def _companyfacts(liabilities, equity, cover=None, weighted=4.0e9, derivatives=None):
    """A minimal 10-Q filer: five quarters of revenue, cash, liabilities,
    equity and weighted shares; no debt tag; an optional cover count
    (value, date)."""
    def flow(val, unit="USD"):
        return {"units": {unit: [{"start": s, "end": e, "val": val, "filed": "2026-05-01", "form": "10-Q",
                                  "accn": f"q{i}"} for i, (s, e) in enumerate(QUARTERS)]}}

    def stock(val):
        return {"units": {"USD": [{"end": e, "val": val, "filed": "2026-05-01", "form": "10-Q", "accn": f"q{i}"}
                                  for i, (_s, e) in enumerate(QUARTERS)]}}

    us_gaap = {"Revenues": flow(45e9), "NetIncomeLoss": flow(1e9), "CashAndCashEquivalentsAtCarryingValue": stock(20e9),
               "Liabilities": stock(liabilities), "StockholdersEquity": stock(equity),
               "WeightedAverageNumberOfSharesOutstandingBasic": flow(weighted, "shares")}
    if derivatives is not None:
        us_gaap["DerivativeLiabilities"] = stock(derivatives)
    out = {"facts": {"us-gaap": us_gaap}}
    if cover:
        out["facts"]["dei"] = {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"end": cover[1], "val": cover[0], "filed": cover[1], "form": "10-Q", "accn": "c"}]}}}
    return out


def _sheet(companyfacts, as_of="2026-10-08"):
    st = edgar_ext.from_json("0000000002", companyfacts, {"name": "Test Co"}, as_of)
    days = pd.bdate_range(end=as_of, periods=260).strftime("%Y-%m-%d")
    ohlcv = pd.DataFrame({"Date": days, "Open": 12.0, "High": 12.2, "Low": 11.8, "Close": 12.0, "Volume": 5e7})
    return facts.build("TST", as_of, f"{as_of}T14:00:00Z", statements=st, ohlcv=ohlcv, offline=True)


@pytest.mark.unit
def test_no_debt_tag_beside_material_liabilities_gives_no_ev():
    """Ford tags no total debt in its 10-Qs beside $245B of liabilities: EV
    counted that debt as 0 ($14B) (release review 2026-10-09)."""
    sheet = _sheet(_companyfacts(liabilities=245e9, equity=37e9, cover=(4.0e9, "2026-04-25")))
    assert sheet.get("market_cap") is not None
    assert sheet.get("ev") is None and sheet.get("ev_sales.ttm") is None and sheet.get("ev_sales.run_rate") is None
    assert any("no debt is tagged" in u and "no EV" in u for u in sheet.unavailable)


@pytest.mark.unit
def test_no_debt_tag_with_small_liabilities_still_counts_debt_as_zero():
    # Liabilities other than derivatives: 3B − 1B = 2B of 23B assets (9%).
    sheet = _sheet(_companyfacts(liabilities=3e9, equity=20e9, cover=(4.0e9, "2026-04-25"), derivatives=1e9))
    assert "no debt tagged at 2026-03-31, counted as 0" in sheet.get("ev").derivation
    assert not any("enterprise value" in u for u in sheet.unavailable)


@pytest.mark.unit
def test_a_stale_cover_count_is_replaced_by_the_weighted_count():
    """Ford's companyfacts end the dei cover series in 2011 (3.73B shares):
    market cap used it for a 2026 run."""
    sheet = _sheet(_companyfacts(liabilities=3e9, equity=20e9, cover=(3.727e9, "2011-04-28"), weighted=3.991e9))
    assert sheet.get("shares.cover") is None
    assert sheet.value("market_cap") == pytest.approx(3.991e9 * 12.0)
    assert "[shares_weighted.2026Q1]" in sheet.get("market_cap").derivation
    assert any("2011-04-28" in u and "over 400 days old" in u for u in sheet.unavailable)


@pytest.mark.unit
def test_a_cover_count_far_off_the_weighted_count_is_not_used():
    sheet = _sheet(_companyfacts(liabilities=3e9, equity=20e9, cover=(2.0e9, "2026-04-25"), weighted=4.0e9))
    assert sheet.get("shares.cover") is None
    assert sheet.value("market_cap") == pytest.approx(4.0e9 * 12.0)
    assert any("50% off the weighted basic count" in u for u in sheet.unavailable)


@pytest.mark.unit
def test_a_current_cover_count_is_used_and_named(onds):
    assert onds.get("shares.cover") is not None
    assert onds.get("market_cap").derivation.startswith("cover-page shares (")
    assert "[shares.cover]" in onds.get("market_cap").derivation


@pytest.mark.unit
def test_an_annual_only_20f_filer_says_its_statements_are_unavailable():
    companyfacts = {"facts": {"us-gaap": {"Revenues": {"units": {"CNY": [
        {"start": "2025-04-01", "end": "2026-03-31", "val": 9.9e11, "filed": "2026-06-20", "form": "20-F", "accn": "a"}]}}}}}
    submissions = {"name": "Foreign ADR Ltd", "filings": {"recent": {"form": ["20-F"]}}}
    st = edgar_ext.from_json("0000000001", companyfacts, submissions, "2026-10-08")
    ohlcv = pd.DataFrame({"Date": pd.bdate_range(end="2026-10-08", periods=30).strftime("%Y-%m-%d"),
                          "Open": 10.0, "High": 10.5, "Low": 9.5, "Close": 10.0, "Volume": 1e6})
    sheet = facts.build("FADR", "2026-10-08", "2026-10-09T14:00:00Z", statements=st, ohlcv=ohlcv, offline=True)
    assert any(u.startswith("SEC statements unavailable: annual-only 20-F/IFRS filer") for u in sheet.unavailable)
    assert "Unavailable: " in facts.render(sheet)


@pytest.mark.unit
def test_a_share_class_ticker_finds_its_cik(monkeypatch):
    from tradingagents.dataflows.vendors import sec_edgar

    table = {"0": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire Hathaway"}}
    monkeypatch.setattr(sec_edgar, "_cached_json", lambda *_a, **_k: table)
    assert sec_edgar.cik_for("BRK.B") == sec_edgar.cik_for("brk-b") == "0001067983"
    assert sec_edgar.cik_for("BRK.C") is None


# ---- R8: earnings multiples, valuation history, one-off and dilution flags ---------


def _synthetic(n=12, split_after=None, gm_dip=None, warrants_at=None, oi=200e6, ni=None):
    """A company with flat quarterly figures: revenue $1B, gross profit $500M
    (one quarter dipped if asked), 1B basic shares (100M before a 10-for-1
    split if asked), price $100 split-adjusted."""
    ends = [f"{2023 + i // 4}-{(i % 4) * 3 + 3:02d}-{30 if (i % 4) in (1, 2) else 31}" for i in range(n)]
    values: dict[str, dict[str, float]] = {k: {} for k in (
        "revenue", "gross_profit", "operating_income", "net_income", "shares_weighted", "shares_diluted",
        "cash", "sti", "debt", "debt_current", "eps_diluted", "ocf", "capex", "warrants_outstanding")}
    for i, e in enumerate(ends):
        values["revenue"][e] = 1e9
        values["gross_profit"][e] = 5e8 if gm_dip is None or i != gm_dip else 3e8
        values["operating_income"][e] = oi
        values["net_income"][e] = ni if ni is not None else oi * 0.8
        shares = 1e8 if split_after is not None and e <= split_after else 1e9
        values["shares_weighted"][e] = shares
        values["shares_diluted"][e] = shares * 1.01
        values["cash"][e], values["debt"][e] = 2e9, 1e9
        values["eps_diluted"][e] = values["net_income"][e] / (shares * 1.01)
        values["ocf"][e], values["capex"][e] = 3e8, 1e8
    if warrants_at is not None:
        values["warrants_outstanding"][ends[warrants_at]] = 9e7
    st = edgar_ext.Statements(cik="0", as_of=ends[-1], fy_end=(12, 31), quarter_ends=ends, year_ends=[])
    # Counts as filed: each 10-Q about a month after its quarter end, so a
    # count for a quarter before the split was filed before it too.
    from datetime import date as _d, timedelta as _td
    for concept in ("shares_weighted", "shares_diluted"):
        st.quarters[concept] = {
            e: edgar_ext.Value(value=values[concept][e], unit="shares",
                               filed=(_d.fromisoformat(e) + _td(days=30)).isoformat(), accn="x", tag=concept)
            for e in ends}
    frame = pd.DataFrame({"Date": pd.date_range(ends[0], ends[-1], freq="D").strftime("%Y-%m-%d")})
    frame["Close"] = 100.0
    return st, values, frame, ends


@pytest.mark.unit
def test_valuation_history_adjusts_counts_filed_before_a_split():
    st, values, frame, ends = _synthetic(split_after="2024-06-30")
    hist = {f.key: f.value for f in facts._valuation_history(st, values, frame, {"ev_sales": 25.0, "pe": 160.0},
                                                             ends[-1], splits=[("2024-08-15", 10.0)])}
    # $100 × 1B shares − $2B cash + $1B debt = $99B EV on $4B TTM revenue, every quarter.
    assert hist["ev_sales_hist.low"] == hist["ev_sales_hist.high"] == 24.75
    assert hist["ev_sales_hist.percentile"] == 100
    assert hist["pe_hist.median"] == pytest.approx(157.8, abs=0.1)
    # Without the split, the pre-split quarters read 10x too cheap.
    unadjusted = {f.key: f.value for f in facts._valuation_history(st, values, frame, {"ev_sales": 25.0}, ends[-1])}
    assert unadjusted["ev_sales_hist.low"] < 3
    # A count a later filing restated (filed after the split) is not scaled
    # again. (ASC 260: a filing after a split shows restated counts, so the
    # filed date says which basis a count is on.)
    for concept in ("shares_weighted", "shares_diluted"):
        for e, val in st.quarters[concept].items():
            if e <= "2024-06-30":
                val.value *= 10
                val.filed = "2024-09-01"
    restated = {f.key: f.value for f in facts._valuation_history(st, values, frame, {"ev_sales": 25.0}, ends[-1],
                                                                 splits=[("2024-08-15", 10.0)])}
    assert restated["ev_sales_hist.low"] == restated["ev_sales_hist.high"] == 24.75


@pytest.mark.unit
def test_valuation_history_after_a_reverse_split_and_heavy_issuance():
    # 1-for-10 reverse split in mid-2024, then 5x issuance: the old heuristic
    # (nearest to today's count) picked the unscaled count (WKHS, review 2026-10-10).
    st, values, frame, ends = _synthetic()
    for concept in ("shares_weighted", "shares_diluted"):
        for e, val in st.quarters[concept].items():
            # Filed before the split: the old basis; after it: restated.
            val.value = 1e10 if val.filed < "2024-07-15" else (1e9 if e <= "2025-06-30" else 5e9)
    frame["Close"] = 100.0
    hist = {f.key: f.value for f in facts._valuation_history(st, values, frame, {"ev_sales": 25.0}, ends[-1],
                                                             splits=[("2024-07-15", 0.1)])}
    # Pre-split: 1e10 × 0.1 = 1e9 shares, the same as the year after the split.
    assert hist["ev_sales_hist.low"] == 24.75


@pytest.mark.unit
def test_implausible_history_is_withheld():
    st, values, frame, ends = _synthetic()
    for e in ends[3:9]:
        values["revenue"][e] = -5e9           # negative-revenue quarters (a reversal)
    hist = facts._valuation_history(st, values, frame, {"ev_sales": 25.0}, ends[-1])
    assert not any(f.key.startswith("ev_sales_hist") for f in hist)


@pytest.mark.unit
def test_no_pe_on_an_operating_loss():
    st, values, frame, ends = _synthetic(oi=-50e6, ni=40e6)
    cols = [facts._q_label(e, ends, st.fy_end) for e in ends[-5:]]
    derived, _ = facts._derived(st, cols, values, 100.0, ends[-1])
    keys = {f.key for f in derived}
    assert not keys & {"pe.ttm", "pe.run_rate"}
    assert any(k.startswith("ttm_eps.") for k in keys)
    assert facts._valuation_history(st, values, frame, {"pe": None}, ends[-1]) == [] or \
        not any(f.key.startswith("pe_hist") for f in facts._valuation_history(st, values, frame, {}, ends[-1]))


@pytest.mark.unit
def test_pe_and_flags_for_one_offs_net_income_and_dilution():
    st, values, frame, ends = _synthetic(gm_dip=8, warrants_at=10, oi=200e6, ni=260e6)
    st.cover_shares = edgar_ext.Value(value=1e9, unit="shares", filed="2026-01-01", accn="x", tag="dei")
    st.cover_shares_date = ends[-1]
    cols = [facts._q_label(e, ends, st.fy_end) for e in ends[-5:]]
    derived, flags = facts._derived(st, cols, values, 100.0, ends[-1])
    pe = next(f for f in derived if f.key == "pe.ttm")
    assert pe.value == pytest.approx(100 / (1.04e9 / 1.01e9), rel=1e-6)
    text = " ".join(flags)
    dipped = cols[1].calendar                       # shown: ends[7:12]; the dip is ends[8]
    assert f"{dipped} gross margin (30.0%) is 20 points below" in text
    assert "exceeds operating income" in text
    assert f"90.0M warrants or rights outstanding as of {cols[3].calendar}" in text and "9.0% of basic" in text


@pytest.mark.unit
def test_calendar_labels_never_repeat():
    # COST's 12/12/12/16-week quarters: two midpoints fall in calendar Q1.
    ends = ["2025-08-31", "2025-11-23", "2026-02-15", "2026-05-10", "2026-08-30"]
    labels = [facts._q_label(e, ends, (8, 30)).calendar for e in ends]
    assert len(set(labels)) == len(labels) and labels == sorted(labels)
    assert labels[2:4] == ["2026Q1", "2026Q2"]


@pytest.mark.unit
def test_a_successor_registrant_keeps_its_predecessors_history():
    new = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"start": "2026-04-01", "end": "2026-06-30", "val": 5, "filed": "2026-08-03", "form": "10-Q"}]}}}}}
    old = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"start": "2026-01-01", "end": "2026-03-31", "val": 4, "filed": "2026-05-04", "form": "10-Q"},
        {"start": "2026-04-01", "end": "2026-06-30", "val": 9, "filed": "2026-07-01", "form": "10-Q"}]}}}}}
    merged = edgar_ext._merge_predecessor(new, old)
    known, _ = edgar_ext._known(merged["facts"]["us-gaap"], "Revenues", "2026-10-09")
    assert known[("2026-01-01", "2026-03-31")]["val"] == 4      # from the predecessor
    assert known[("2026-04-01", "2026-06-30")]["val"] == 5      # the successor's later filing wins


@pytest.mark.unit
def test_anomaly_flags_say_only_what_the_figures_support():
    st, values, frame, ends = _synthetic()
    cols = [facts._q_label(e, ends, st.fy_end) for e in ends[-5:]]
    last, prev = ends[-1], ends[-2]
    # A bond moving to current: long-term −40%, total unchanged → no debt flag.
    values["debt"][last], values["debt_current"] = 0.6e9, {last: 0.4e9, prev: 0.0}
    values["debt"][prev] = 1.0e9
    # Cash jumps with FCF unknown → no "not operating cash" claim.
    values["cash"][last] = 9e9
    del values["ocf"][last]
    # A large non-operating LOSS.
    values["non_operating"] = {last: -5e9}
    values["net_income"][last] = 1e9
    _, flags = facts._derived(st, cols, values, 100.0, last)
    text = " ".join(flags)
    assert "Total debt moved" not in text and "Long-term debt moved" not in text
    assert "is not operating cash" not in text
    assert "Non-operating loss" in text and "understate" in text
