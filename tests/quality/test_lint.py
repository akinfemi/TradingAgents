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


@pytest.mark.unit
@pytest.mark.parametrize("text", [
    # R7 staging/eval: correct derivations the RM and PM were told to make.
    "Tangible equity: equity of $1.57B [F:equity.2026Q2] less goodwill of $661.4M [F:goodwill.2026Q2] less "
    "intangibles of $583.3M [F:intangibles.2026Q2] is about $325M.",
    "Base: 13x × $174.1M TTM revenue [F:ttm_revenue.2026Q2] = $2,263M EV.",
    "Cash was $657.9M [F:cash.2026Q2] and short-term investments $726.6M [F:sti.2026Q2]. Together that is $1,384.5M.",
    "The $56M European order is headline-only and unverified; do not size on it.",
    "Do not use short-interest data, consensus estimates or price targets in sizing.",
    "The bull's 'deeply oversold base': wrong.",
])
def test_derivations_and_compliance_are_clean(sheet, text):
    blocking = [f for f in lint_text(text, sheet, "research_manager", "rm") if f.blocking or f.severity == "load_bearing"]
    assert blocking == [], [(f.kind, f.quote[:60]) for f in blocking]


@pytest.mark.unit
def test_a_wrong_derivation_is_still_caught(sheet):
    text = "Tangible equity: equity of $1.57B [F:equity.2026Q2] less goodwill [F:goodwill.2026Q2] is about $1.30B."
    assert [f.kind for f in lint_text(text, sheet, "research_manager", "rm") if f.severity == "load_bearing"]



@pytest.mark.unit
@pytest.mark.parametrize("text", [
    # Staging ONDS, 2026-10-08 (R7 calibration, second round).
    "FCF went from −$52.6M to −$93.8M [F:fcf.2026Q1/2026Q2].",
    "Reassess re-adding only after a confirmed daily close above the 50-day SMA.",
])
def test_combined_keys_and_conditional_reentry_are_clean(sheet, text):
    flags = [f for f in lint_text(text, sheet, "portfolio_manager", "pm") if f.blocking]
    assert flags == [], [(f.kind, f.expected) for f in flags]


@pytest.mark.unit
def test_a_key_cited_for_a_bare_number_is_not_matched_to_a_dollar_figure(sheet):
    atr = sheet.value("atr14.usd")
    text = f"The $7.12 stop sits about 0.17 ATR below the $7.19 close ((7.19-7.12)/{atr:.4f} [F:atr14.usd])."
    assert [f for f in lint_text(text, sheet, "portfolio_manager", "pm") if f.kind == "cited_mismatch"] == []



@pytest.mark.unit
def test_more_citation_and_volume_forms(sheet):
    text = "Price is below the 10-day EMA and 50-day SMA [F:ema10.value/F:sma50.value]."
    assert [f for f in lint_text(text, sheet, "research_manager", "rm") if f.kind == "unknown_key"] == []
    close = sheet.value("price.close")
    vol = "61.0M shares"
    text2 = f"Average volume of {vol} at the ${close:.2f} close is about ${61.0e6 * close / 1e6:,.0f}M a day."
    assert [f for f in lint_text(text2, sheet, "portfolio_manager", "pm") if f.severity == "load_bearing"] == []


@pytest.mark.unit
def test_tangible_equity_and_labelled_news_figures_are_not_unsupported(sheet):
    """Staging ONDS, 2026-10-09: both held a report."""
    text = ("The balance-sheet point (tangible equity ~$0.33B against $1.24B of acquisition-related intangibles "
            "and $1.04B derivative liabilities) is real. Note: the news-reported $165M/$56M order figures are "
            "not fact-sheet items and are excluded from this thesis entirely.")
    flags = [f for f in lint_text(text, sheet, "research_manager", "rm") if f.kind == "unsupported"]
    assert flags == [], [f.quote for f in flags]
    assert [f.kind for f in lint_text("Backlog of $165M supports the call.", sheet, "research_manager", "rm")] == ["unsupported"]


@pytest.mark.unit
def test_two_figures_in_one_sentence_are_one_finding(sheet):
    state = {"fact_sheet": sheet.sheet, "investment_plan": "Orders of $165M and $56M support the call."}
    flags = [f for f in lint_state(state)["flags"] if f["kind"] == "unsupported"]
    assert len(flags) == 1


# ---- review vocabulary in the digest (MSFT, 2026-10-09) ------------------------------------

MSFT_HEADLINE = "Underweight: quality verified, but spending is outrunning cash conversion"


def process_flags(text, field="digest.headline", stage="digest"):
    return [f for f in lint_text(text, Facts(None), stage, field) if f.kind == "process_language"]


@pytest.mark.unit
def test_review_vocabulary_in_the_digest_headline_is_flagged():
    flags = process_flags(MSFT_HEADLINE)
    assert len(flags) == 1 and flags[0].blocking and flags[0].severity == "load_bearing"
    for text in ("Hold: margins corrected per errata E5", "Buy: the fact sheet shows net cash",
                 "Sell: unverified partnership claims", "Hold: the previous draft overstated FCF",
                 "Hold: lint-clean figures", "Hold: as fixed in E12"):
        assert process_flags(text, "digest.bull_thesis"), text


@pytest.mark.unit
def test_normal_digest_prose_is_not_flagged():
    for text in (
        "Overweight: operating strength outweighs capex risk",
        "Hold: the FDA review of the lead asset is due in Q1; revision of guidance is possible",
        "Underweight: spending is outrunning cash conversion; EPS fell 4% and EBITDA margin narrowed",
        "Exit if quarterly FCF falls below $20B; currently $24.6B",
    ):
        assert process_flags(text) == [], text


@pytest.mark.unit
def test_research_manager_labels_are_not_flagged_outside_the_digest():
    text = "**Verified** (derived): FCF of $24.6B covers capex. The unverified news claim is excluded (errata E5)."
    assert process_flags(text, field="rm", stage="research_manager") == []
    state = {"fact_sheet": {"facts": []}, "investment_plan": text,
             "report_digest": {"headline": "Hold: fine",
                               "bull_points": [{"title": "Cash", "detail": "Verified FCF covers capex."}]}}
    flags = [f for f in lint_state(state)["flags"] if f["kind"] == "process_language"]
    assert [(f["stage"], f["field"]) for f in flags] == [("digest", "digest.bull_points")]
    assert flags[0]["severity"] == "load_bearing" and flags[0]["blocking"]
