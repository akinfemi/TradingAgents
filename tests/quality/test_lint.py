"""The linter against the ONDS reference failure (REPORT_QUALITY_PLAN
Appendix A): every seeded error is caught, and correct text is clean."""

import json
from pathlib import Path

import pandas as pd
import pytest

from tradingagents.quality import edgar_ext, facts
from tradingagents.quality.lint import Facts, check_target, figures, lint_state, lint_text

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def sheet():
    st = edgar_ext.from_json("0001646188", json.loads((FIX / "onds_companyfacts.json").read_text()),
                             json.loads((FIX / "onds_submissions.json").read_text()), "2026-10-05")
    built = facts.build("ONDS", "2026-10-05", "2026-10-05T10:33:00Z", statements=st,
                        ohlcv=pd.read_csv(FIX / "onds_ohlcv.csv"), offline=True)
    return Facts(built.model_dump(mode="json"))


def kinds(text, sheet, sources=None, field="bear"):
    return {f.kind for f in lint_text(text, sheet, "bear_researcher", field, sources) if f.blocking}


FUNDAMENTALS = ("G&A was $128.0M [F:ga.2026Q2], S&M $20.9M [F:sm.2026Q2] and R&D $31.0M [F:rnd.2026Q2]; "
                "operating cash flow was −$86.1M [F:ocf.2026Q2].")


# ---- Appendix A, one test per error class ------------------------------------------

@pytest.mark.unit
def test_invented_quote_from_the_fundamentals_report(sheet):
    text = 'From the fundamental analysis: "SG&A of $179.8M is 4.98x gross profit".'
    assert "misattributed" in kinds(text, sheet, {"fundamentals": FUNDAMENTALS})


@pytest.mark.unit
def test_wrong_rnd_figure_cited(sheet):
    assert "cited_mismatch" in kinds("R&D was $23.4M [F:rnd.2026Q2], 26% of opex.", sheet)


@pytest.mark.unit
def test_total_cash_change_presented_as_operating_cash_flow(sheet):
    assert "cited_mismatch" in kinds("Operating cash flow was negative $372.03M [F:ocf.2026Q2].", sheet)


@pytest.mark.unit
def test_annual_figure_compared_with_a_quarter(sheet):
    text = "Cash burn is worsening from −$38.7M [F:ocf.FY2025] to −$86.1M [F:ocf.2026Q2]."
    assert "period_mismatch" in kinds(text, sheet)


@pytest.mark.unit
def test_total_opex_passed_off_as_ga(sheet):
    text = "G&A rose from $43.3M [F:ga.2026Q1] to $199.1M [F:opex.2026Q2] in one quarter."
    assert "concept_mismatch" in kinds(text, sheet)


@pytest.mark.unit
def test_bearish_cross_when_the_cross_was_bullish(sheet):
    assert "direction" in kinds("The MACD printed a bearish cross and the signal is declining.", sheet)


@pytest.mark.unit
def test_price_side_of_the_200_day(sheet):
    assert "direction" in kinds("The stock trades above its 200-day SMA, a healthy uptrend.", sheet)


@pytest.mark.unit
def test_percent_change_that_does_not_compute(sheet):
    assert "arithmetic" in kinds("Revenue grew from $50.1M to $83.8M (+40%) in a quarter.", sheet)


@pytest.mark.unit
def test_short_interest_is_not_used(sheet):
    assert "data_policy" in kinds("With 42% of the float shorted, a squeeze is possible.", sheet)
    assert "data_policy" not in kinds("Short interest is not available on this platform.", sheet)


@pytest.mark.unit
def test_target_on_the_wrong_side_of_the_price(sheet):
    flags = check_target({"rating": "Underweight", "price_target": 9.5}, sheet)
    assert [f.kind for f in flags] == ["target_direction"]
    assert check_target({"rating": "Underweight", "price_target": 5.5}, sheet) == []


@pytest.mark.unit
def test_unknown_key(sheet):
    assert "unknown_key" in kinds("The 10-year yield is 4.5% [F:10y_treasury].", sheet)


# ---- false positives -------------------------------------------------------------------

CLEAN = [
    "Revenue was $83.8M [F:revenue.2026Q2], up 1,235% year on year [F:revenue_yoy.2026Q2].",
    "Revenue grew from $50.1M [F:revenue.2026Q1] to $83.8M [F:revenue.2026Q2] (+67%).",
    "Gross margins of 42-49% [F:gross_margin.2026Q2] over the year held up.",
    "Operating cash flow was −$86.1M [F:ocf.2026Q2] against FCF of −$93.8M [F:fcf.2026Q2].",
    "The MACD crossed bullish on 2026-09-21 [F:macd.cross_date] and sits above its signal line.",
    "RSI of 46.6 [F:rsi14.value] is not oversold (<30).",
    "A break below the 50-day average would weaken the setup.",
    "Set the stop near $5 [F:52w.low], the 52-week low.",
    "Stock issued for acquisitions was $741.5M ([F:stock_for_acquisitions.2026Q1]+[F:stock_for_acquisitions.2026Q2]).",
    "Quarterly revenue of $83.8M Q2, with $165M in new orders to convert [F:revenue.2026Q2].",
    "Cash plus short-term investments of $1.38B [F:cash_sti.2026Q2] covers 14.8 quarters [F:runway_quarters.2026Q2].",
]


@pytest.mark.unit
@pytest.mark.parametrize("text", CLEAN)
def test_correct_text_is_clean(sheet, text):
    blocking = [f for f in lint_text(text, sheet, "bull_researcher", "bull") if f.blocking]
    assert blocking == [], [(f.kind, f.expected) for f in blocking]


@pytest.mark.unit
def test_figures_read_units_ranges_and_precision():
    figs = figures("from $5 to $7.24, a 42-49% margin, 5.5x opex, −$162.9M and $1.38B")
    assert [(round(f.value, 2), f.kind) for f in figs] == [
        (5.0, "usd"), (7.24, "usd"), (49.0, "pct"), (5.5, "x"), (-162.9e6, "usd"), (1.38e9, "usd")]
    assert figs[0].step == 1.0 and figs[2].range_low == 42.0


@pytest.mark.unit
def test_full_pass_reports_counts(sheet):
    state = {"fact_sheet": sheet.sheet, "market_report": "MACD shows a bearish cross.",
             "investment_debate_state": {"bull_history": "", "bear_history": ""}, "risk_debate_state": {},
             "final_trade_decision": "", "portfolio_decision": {"rating": "Buy", "price_target": 6.0}}
    report = lint_state(state)
    assert report["has_fact_sheet"] and report["blocking"] == 2
    assert {f["kind"] for f in report["flags"]} == {"direction", "target_direction"}


@pytest.mark.unit
def test_the_target_is_derived_and_inside_its_cases(sheet):
    ok = {"rating": "Underweight", "price_target": 6.08, "bear_case_value": 4.1, "bull_case_value": 8.0,
          "target_math": "12x × $174.1M = $2.09B EV; + $1.38B cash = $3.47B; ÷ 570.6M shares = $6.08"}
    assert check_target(ok, sheet) == []
    off = {**ok, "target_math": "… ÷ 570.6M shares = $5.10"}
    assert [f.kind for f in check_target(off, sheet)] == ["target_math"]
    outside = {**ok, "bull_case_value": 5.9}
    assert [f.kind for f in check_target(outside, sheet)] == ["target_range"]


@pytest.mark.unit
def test_rendered_decision_carries_the_valuation():
    from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating, render_pm_decision

    md = render_pm_decision(PortfolioDecision(
        rating=PortfolioRating.UNDERWEIGHT, executive_summary="s", investment_thesis="t", price_target=6.08,
        valuation_method="EV/Sales on TTM revenue", valuation_inputs=["ev_sales.ttm", "ttm_revenue.2026Q2"],
        target_math="12x × $174.1M = $2.09B; ÷ 570.6M = $6.08", bear_case_value="$4.10", bull_case_value=8))
    for line in ("**Valuation Method**: EV/Sales on TTM revenue", "**Valuation Inputs**: ev_sales.ttm, ttm_revenue.2026Q2",
                 "**Target Math**: 12x", "**Bear Case Value**: 4.1", "**Bull Case Value**: 8.0"):
        assert line in md


@pytest.mark.unit
def test_an_unverified_social_claim_cannot_carry_a_load_bearing_point(sheet):
    text = "A $50M order from the Army (unverified social claim, n=1) underpins the bull case."
    assert "social_claim" in {f.kind for f in lint_text(text, sheet, "digest", "digest.bull_thesis") if f.blocking}
    assert "social_claim" not in {f.kind for f in lint_text(text, sheet, "sentiment_analyst", "sentiment_report")}
