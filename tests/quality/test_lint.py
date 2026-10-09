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


@pytest.mark.unit
def test_target_math_with_bear_and_bull_cases_after_the_base(sheet):
    """Eval 2026-10-09: the last '=' was the bull case, not the target."""
    close = sheet.value("price.close")
    target = round(close * 0.9, 2)
    decision = {"rating": "Underweight", "price_target": target,
                "target_math": f"Base: 28x × EPS = ${target:,.2f}. Bear: 25x = ${target * 0.8:,.2f}. "
                               f"Bull: 32x = ${target * 1.15:,.2f}."}
    assert [f.kind for f in check_target(decision, sheet) if f.kind == "target_math"] == []
    wrong = {**decision, "price_target": round(target * 0.95, 2)}
    assert [f.kind for f in check_target(wrong, sheet) if f.kind == "target_math"] == ["target_math"]


@pytest.mark.unit
def test_chained_valuation_math_is_supported(sheet):
    """Eval 2026-10-09 (GPT deep tier, NVDA): annualised revenue derived once,
    then reused in the bear and bull cases."""
    q = sheet.value("revenue.2026Q2")
    ann = 4 * q
    text = (f"Annualized revenue = 4 × ${q / 1e6:,.1f}M [F:revenue.2026Q2] = ${ann / 1e6:,.1f}M. "
            f"Base assumed 16x: 16 × ${ann / 1e6:,.1f}M = ${16 * ann / 1e6:,.1f}M EV. "
            + "Reasoning about margins, dilution and the cash runway fills this gap between the cases. " * 4
            + f"Bear assumed 12x: 12 × ${ann / 1e6:,.1f}M = ${12 * ann / 1e6:,.1f}M EV.")
    flags = [f for f in lint_text(text, sheet, "portfolio_manager", "pm") if f.kind == "unsupported"]
    assert flags == [], [f.quote for f in flags]
    assert [f.kind for f in lint_text("Backlog of $912.0M supports the call.", sheet, "portfolio_manager", "pm")] == ["unsupported"]


# ---- code review of the linter, 2026-10-09 (one test per finding) ---------------------------


def lb_kinds(text, sheet, stage="research_manager", field="rm"):
    return [f.kind for f in lint_text(text, sheet, stage, field) if f.blocking or f.severity == "load_bearing"]


@pytest.mark.unit
def test_1_to_is_not_a_range(sheet):
    """'rose 67% to $95.0M', 'from $50.1M to $95.0M' and 'in 2025 to $40M' made
    a range low out of whatever number came before 'to'."""
    for text in ("Revenue rose 67% to $95.0M [F:revenue.2026Q2] in Q2.",
                 "Revenue rose from $50.1M to $95.0M [F:revenue.2026Q2].",
                 "Revenue climbed in 2025 to $40M [F:revenue.FY2025]."):
        assert "cited_mismatch" in lb_kinds(text, sheet), text
    assert figures("Revenue climbed in 2025 to $40M")[0].range_low is None
    # Explicit ranges still read as ranges, the low end in its own unit.
    assert figures("between $80M and $90M")[1].range_low == 80e6
    assert figures("$500K–$2M")[1].range_low == 500e3
    for text in ("Revenue rose 67% to $83.8M [F:revenue.2026Q2] in Q2.",
                 "Revenue between $80M and $90M [F:revenue.2026Q2] in Q2."):
        assert lb_kinds(text, sheet) == [], text


@pytest.mark.unit
def test_2_a_dash_between_money_figures_is_a_range_and_a_year_is_not_a_figure(sheet):
    low, high = figures("Revenue of $80M-$90M")
    assert (low.in_range, high.value, high.range_low, high.signed) == (True, 90e6, 80e6, False)
    assert lb_kinds("Revenue of $80M-$90M [F:revenue.2026Q2] in Q2.", sheet) == []
    # A year or quarter label before a minus sign is not the left end of a range.
    assert [f.value for f in figures("Q2 2026 −$86.1M")] == [-86.1e6]
    assert [f.value for f in figures("FY2025 −$38.7M")] == [-38.7e6]
    # A subtraction is not a range, and its operand is not negative.
    sub = figures("$83.8M - $31.0M = $52.8M")
    assert sub[1].value == 31.0e6 and sub[1].range_low is None
    # Still a range check: a cited value outside the written range is flagged.
    assert "cited_mismatch" in lb_kinds("Revenue of $90M-$95M [F:revenue.2026Q2] in Q2.", sheet)


@pytest.mark.unit
def test_3_a_wrong_cited_figure_is_not_excused_as_a_sum_of_its_neighbours(sheet):
    """'R&D of $51.9M' passed as R&D + S&M: no arithmetic is shown, and the
    figure's own key is never one of its operands."""
    assert "cited_mismatch" in lb_kinds("R&D of $51.9M [F:rnd.2026Q2] in Q2; S&M $20.9M [F:sm.2026Q2].", sheet)
    q1, q2 = sheet.value("cash_sti.2026Q1"), sheet.value("cash_sti.2026Q2")
    per_share = sheet.value("market_cap") / sheet.value("shares.cover")
    atr = sheet.value("atr14.usd")
    for text in (
        # Arithmetic on the figure's own keys, as the stages are told to write it.
        f"Cash plus short-term investments fell ${(q1 - q2) / 1e6:.1f}M [F:cash_sti.2026Q1][F:cash_sti.2026Q2].",
        f"Market value is ${per_share:.2f} a share (market cap ÷ shares) [F:market_cap][F:shares.cover].",
        "R&D and S&M combined were $51.9M [F:rnd.2026Q2][F:sm.2026Q2].",
        f"A 2-ATR stop sits ${2 * atr:.2f} below the close [F:atr14.usd].",
        # Arithmetic written out in the sentence.
        "Opex excluding R&D was $168.1M ($199.1M [F:opex.2026Q2] less $31.0M [F:rnd.2026Q2]).",
        # A change in the one line it cites, the change word governing the figure.
        f"Cash plus short-term investments fell ${(q1 - q2) / 1e6:.1f}M in Q2 [F:cash_sti.2026Q2].",
    ):
        assert lb_kinds(text, sheet) == [], text
    # Not a change: the same number stated as the level is wrong.
    assert "cited_mismatch" in lb_kinds(
        f"Cash plus short-term investments were ${(q1 - q2) / 1e6:.1f}M at quarter end [F:cash_sti.2026Q2].", sheet)


@pytest.mark.unit
def test_4_swapped_citations_are_caught(sheet):
    text = "R&D fell to $13.5M [F:rnd.2026Q2] from $31.0M [F:rnd.2026Q1]."
    assert lb_kinds(text, sheet).count("cited_mismatch") == 2
    assert lb_kinds("R&D rose to $31.0M [F:rnd.2026Q2] from $13.5M [F:rnd.2026Q1].", sheet) == []
    # A citation written before its figure belongs to that figure.
    upper, lower = sheet.value("boll.upper"), sheet.value("boll.lower")
    text = f"Bands: [F:boll.upper] ${upper:.2f} and [F:boll.lower] ${lower:.2f}, midpoint ${(upper + lower) / 2:.2f}."
    assert lb_kinds(text, sheet, "portfolio_manager", "pm") == []


@pytest.mark.unit
def test_5_a_sign_that_contradicts_the_fact_is_flagged(sheet):
    for text in ("Q2 net income was a profit of $88.4M [F:net_income.2026Q2].",
                 "Operating cash flow turned positive at +$86.1M [F:ocf.2026Q2]."):
        assert "cited_mismatch" in lb_kinds(text, sheet), text
    for text in ("Q2 net loss was $88.4M [F:net_income.2026Q2].",
                 "Q2 net income was −$88.4M [F:net_income.2026Q2].",
                 "Operating cash flow was negative $86.1M [F:ocf.2026Q2].",
                 # Stored negative, written as a gain: not a signed line item.
                 "The warrant fair-value gain was $15.2M [F:warrant_fair_value.2026Q2].",
                 # Costs are stored positive and often written as outflows.
                 "Capex was −$7.8M [F:capex.2026Q2]."):
        assert lb_kinds(text, sheet) == [], text


@pytest.mark.unit
def test_6_an_uncited_small_figure_must_round_to_a_fact(sheet):
    """'$6.2M' sat within the ±$0.1M floor of revenue.2025Q2 ($6.273M)."""
    assert lb_kinds("Backlog of $6.2M supports the call.", sheet) == ["unsupported"]
    assert lb_kinds("Operating cash flow of +$86.1M supports the call.", sheet) == ["unsupported"]
    for text in ("Revenue of $6.3M a year earlier supports the call.",
                 "Capex of $7.8M supports the call.",
                 "Operating cash flow of −$86.1M weighs on the call."):
        assert lb_kinds(text, sheet) == [], text


@pytest.mark.unit
def test_7_only_disclosure_phrases_label_a_figure_unverified(sheet):
    for text in ("Cash dropped to $212M by quarter end, funding just two quarters.",
                 "Quarterly revenue reported by the company was $412M."):
        assert lb_kinds(text, sheet) == ["unsupported"], text
    for text in ("The $212M order was dropped from the thesis as news-only.",
                 "A $56M order, as reported by MT Newswires, is not on the fact sheet.",
                 "A $56M European contract reported in the news is a catalyst, not a figure we size on."):
        assert lb_kinds(text, sheet) == [], text


@pytest.mark.unit
def test_8_target_math_without_a_dollar_result_is_read(sheet):
    base = {"rating": "Underweight", "price_target": 6.08}
    for math in ("$2.09B EV + $1.38B cash = $3.47B equity; / 570.6M shares = 6.08 per share",
                 "12x × $174.1M = $2.09B; + $1.38B = $3.47B; ÷ 570.6M ≈ $6.08",
                 "Equity value $3.47B / 570.6M shares → $6.08",
                 "Base 12x × $174.1M = $2.09B ($6.08/share)"):
        assert check_target({**base, "target_math": math}, sheet) == [], math
    assert [f.kind for f in check_target({**base, "target_math": "… ÷ 570.6M ≈ $5.10"}, sheet)] == ["target_math"]
    # Math with no stated result is not flagged for a per-share input.
    no_result = {**base, "price_target": 306.0, "target_math": "TTM EPS is $24.48 per share; apply 12.5x."}
    assert [f for f in check_target(no_result, sheet) if f.kind == "target_math"] == []


@pytest.mark.unit
def test_9_one_average_against_another_is_not_a_price_claim():
    sheet = Facts({"facts": [{"key": "sma200.gap_pct", "value": -4.0, "unit": "pct"},
                             {"key": "sma50.gap_pct", "value": -6.0, "unit": "pct"}]})
    for text in ("The 50-day SMA remains above the 200-day SMA, so the long-term trend is intact.",
                 "The 50-day moving average sits above the 200-day moving average.",
                 "A daily close below the 50-day SMA cancels the add program."):
        assert lb_kinds(text, sheet) == [], text
    assert lb_kinds("The stock trades above its 200-day SMA.", sheet) == ["direction"]
    assert lb_kinds("Price remains above the 50-day SMA.", sheet) == ["direction"]


@pytest.mark.unit
def test_10_short_of_a_high_and_ordinary_names_are_not_policy_or_process_language(sheet):
    assert lb_kinds("The stock still trades 47% short of its 52-week high [F:52w.from_high_pct].", sheet) == []
    assert "data_policy" in lb_kinds("With 42% of the float shorted, a squeeze is possible.", sheet)
    for text in ("Buy: E2 jet deliveries are accelerating", "Hold: E7 Wedgetail award is the swing factor",
                 "Buy: FDA-verified endpoints de-risk the launch", "Hold: Series E3 funding round closed",
                 "Hold: the trial data were independently verified", "Hold: no catalyst can be verified this week"):
        assert process_flags(text) == [], text
    for text in ("Hold: margin restated (E5)", "Hold: see E2", "Buy: Verified (derived) cash covers capex",
                 "Hold: bear won on verified arithmetic", "Buy: the (verified) backlog"):
        assert process_flags(text), text


@pytest.mark.unit
def test_11_a_patch_the_relint_would_hold_is_not_applied_and_digest_points_are_linted(sheet):
    from tradingagents.quality import editor
    from tradingagents.quality.lint import LOAD_BEARING_FIELDS

    digest = {"headline": "Hold: old", "bull_points": [{"title": "Cash", "detail": "Liquidity is ample."}]}
    out, applied = editor.apply_patch(digest, [{"field": "headline", "action": "replace",
                                                "value": "Hold: backlog of $412M covers two years of revenue"}], sheet)
    assert out["headline"] == "" and applied == [{"field": "headline", "action": "delete"}]
    out, _ = editor.apply_patch(digest, [{"field": "headline", "action": "replace",
                                          "value": "Hold: revenue of $83.8M grew fast"}], sheet)
    assert out["headline"] == "Hold: revenue of $83.8M grew fast"
    # The points get the figure checks too (minor: they restate derived figures uncited).
    state = {"fact_sheet": sheet.sheet,
             "report_digest": {"headline": "Hold: fine",
                               "bull_points": [{"title": "R&D", "detail": "R&D was $23.4M [F:rnd.2026Q2]."}]}}
    flags = [f for f in lint_state(state)["flags"] if f["field"] == "digest.bull_points"]
    assert [(f["kind"], f["severity"]) for f in flags] == [("cited_mismatch", "minor")]
    assert "key_numbers" not in LOAD_BEARING_FIELDS


@pytest.mark.unit
def test_12_citation_forms_brackets_and_long_text(sheet):
    import time

    for text in ("R&D was $23.4M [F:rnd.2026Q2, F:rnd.2026Q1].", "R&D was $23.4M [F: rnd.2026Q2].",
                 "R&D was $23.4M [F:rnd.2026Q2; F:sm.2026Q2]."):
        assert "cited_mismatch" in lb_kinds(text, sheet), text
    assert lb_kinds("R&D was $31.0M [F: rnd.2026Q2] and S&M $20.9M [F:sm.2026Q2 , F:rnd.2026Q1].", sheet) == []
    # Parentheses are a negative only in a table.
    assert figures("Revenue ($83.8M) beat")[0].value == 83.8e6
    assert figures("| Q2 | ($86.1M) |")[0].value == -86.1e6
    # A run-on text without sentence breaks is linted in linear time (was 9.1s at 40k chars).
    text = "".join(f"G&A $128.0M [F:ga.2026Q2], S&M $20.9M, backlog ${100 + i}.{i % 10}M; " for i in range(750))
    started = time.monotonic()
    lint_text(text, sheet, "research_manager", "rm")
    assert len(text) > 40_000 and time.monotonic() - started < 5
