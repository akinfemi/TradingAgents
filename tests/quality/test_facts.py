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
