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
    close = sheet.value("price.close")
    assert check_target({"rating": "Underweight", "price_target": round(close * 0.92, 2)}, sheet) == []


@pytest.mark.unit
def test_target_outside_the_rating_band(sheet):
    # AMD, 2026-10-10: -15.3% rated Underweight ("modestly underperform").
    close = sheet.value("price.close")
    flags = check_target({"rating": "Underweight", "price_target": round(close * 0.847, 2)}, sheet)
    assert [f.kind for f in flags] == ["target_tier"] and "Sell" in flags[0].expected
    assert check_target({"rating": "Sell", "price_target": round(close * 0.847, 2)}, sheet) == []
    assert [f.kind for f in check_target({"rating": "Buy", "price_target": round(close * 1.05, 2)}, sheet)] \
        == ["target_tier"]
    # A point of slack at the band's edge.
    assert check_target({"rating": "Hold", "price_target": round(close * 1.045, 2)}, sheet) == []


@pytest.mark.unit
def test_reader_voice_in_the_digest(sheet):
    def voice(text, field="digest.risk_aggressive"):
        return [f for f in lint_text(text, sheet, "digest", field) if f.kind == "voice"]
    assert voice("I'd reconsider the trim stance if the stock stabilizes.")
    assert voice("This lens adds a post-event range test.")
    assert voice("The plan trims exposure above the investor's modest target.")
    assert voice("Stage trims after the weekend using limit orders.")
    assert voice("Favors waiting for the report.") == []
    assert voice("The Phase I trial reads out in the US next quarter.") == []
    assert voice("I'd add.", field="market_report") == []      # analyst text is not reader text
    assert all(f.severity == "minor" for f in voice("We would trim.", field="pm"))


@pytest.mark.unit
def test_rating_words_and_margin_wording(sheet):
    state = {"fact_sheet": None, "portfolio_decision": {"rating": "Underweight"},
             "report_digest": {"headline": "Underweight: demanding expectations",
                               "ruling": "The bear case won narrowly on valuation.", "conviction": 62,
                               "risk_neutral": {"stance": "Favors restraint.",
                                                "summary": "Would revisit the SELL/trim stance. Turns Overweight on a beat."},
                               "exit_triggers": [{"title": "Momentum", "detail": "A reclaim of the 10-day EMA pauses trims."}]}}
    flags = lint_state(state)["flags"]
    got = {(f["kind"], f["field"]) for f in flags}
    assert ("rating_word", "digest.risk_neutral") in got
    assert sum(1 for f in flags if f["kind"] == "rating_word") == 2   # SELL and Overweight
    assert ("margin_wording", "digest.ruling") in got
    assert ("trigger_mix", "digest.exit_triggers") in got
    assert not any(f["kind"] == "rating_word" and f["field"] == "digest.headline" for f in flags)
    state["report_digest"].update(conviction=38, exit_triggers=[
        {"title": "Data Center slows", "detail": "Data Center growth below 60%; currently 107%."}])
    got = {f["kind"] for f in lint_state(state)["flags"]}
    assert "margin_wording" not in got and "trigger_mix" not in got


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
    ok = {"rating": "Sell", "price_target": 6.08, "bear_case_value": 4.1, "bull_case_value": 8.0,
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
    base = {"rating": "Sell", "price_target": 6.08}
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
    # Not applied, and not emptied: the original stays, so the relint still sees it.
    assert out["headline"] == "Hold: old" and applied == []
    out, applied = editor.apply_patch(digest, [{"field": "bull_points[0].detail", "action": "replace",
                                                "value": "R&D was $23.4M [F:rnd.2026Q2]."}], sheet)
    assert out["bull_points"] == [] and applied == [{"field": "bull_points[0].detail", "action": "delete"}]
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
    # A run-on text is read in one pass: sentence bounds are computed once, not
    # scanned per figure (was 9.1s at 40k chars), and a figure's "sentence" is
    # a bounded window.
    from tradingagents.quality import lint as lint_module

    text = "".join(f"G&A $128.0M [F:ga.2026Q2], S&M $20.9M, backlog ${100 + i}.{i % 10}M; " for i in range(750))

    def no_rescan(*_a, **_k):
        raise AssertionError("figures() rescanned the text per figure")

    original = lint_module._sentence_at
    lint_module._sentence_at = no_rescan
    try:
        figs = figures(text)
    finally:
        lint_module._sentence_at = original
    assert len(text) > 40_000 and len(figs) == 2250
    assert max(len(f.sentence) for f in figs) <= lint_module._MAX_SENTENCE + 20
    assert time.monotonic()  # (no wall-clock bound: CI timing is not a test)


# ---- second review of the linter, 2026-10-09 (negative tests) -------------------------------


@pytest.mark.unit
def test_r2_1_a_rejected_text_replacement_keeps_the_original(sheet):
    from tradingagents.quality import editor

    digest = {"headline": "Hold: the original headline"}
    out, applied = editor.apply_patch(digest, [{"field": "headline", "action": "replace",
                                                "value": "Hold: backlog of $412M covers two years"}], sheet)
    assert out["headline"] == "Hold: the original headline" and applied == []


@pytest.mark.unit
def test_r2_2_a_trailing_citation_is_not_the_next_figures(sheet):
    for text in ("| Revenue | $83.8M [F:revenue.2026Q2] | $83.8M [F:revenue.2026Q1] |",
                 "R&D was $31.0M [F:rnd.2026Q2], $31.0M [F:rnd.2026Q1] a quarter earlier."):
        assert "cited_mismatch" in lb_kinds(text, sheet), text
    # Cite-first throughout is still read as cite-first.
    text = (f"[F:rnd.2026Q1] ${sheet.value('rnd.2026Q1') / 1e6:.1f}M, "
            f"[F:rnd.2026Q2] ${sheet.value('rnd.2026Q2') / 1e6:.1f}M and [F:sm.2026Q2] $20.9M.")
    assert lb_kinds(text, sheet) == []


@pytest.mark.unit
def test_r2_3_a_level_is_not_a_change_and_the_direction_must_agree(sheet):
    for text in ("Cash fell to $368.1M [F:cash.2026Q2].", "Revenue rose to $33.7M [F:revenue.2026Q2].",
                 "Operating cash flow improved by $34.8M [F:ocf.2026Q1][F:ocf.2026Q2].",
                 "Cash rose by $368.1M [F:cash.2026Q2]."):
        assert "cited_mismatch" in lb_kinds(text, sheet), text
    for text in ("Cash fell by $368.1M [F:cash.2026Q2].", "Revenue rose $33.7M quarter on quarter [F:revenue.2026Q2].",
                 "Operating cash flow worsened by $34.8M [F:ocf.2026Q1][F:ocf.2026Q2]."):
        assert lb_kinds(text, sheet) == [], text


@pytest.mark.unit
def test_r2_4_a_negative_percentage_is_not_excused_by_its_own_digits(sheet):
    for text in ("TTM revenue growth was −979.3% [F:ttm_revenue_yoy.2026Q2].",
                 "The stock is −41.3% from its 52-week low [F:52w.from_low_pct].",
                 "Share count changed −232.4% YoY [F:shares_yoy.2026Q2]."):
        assert "cited_mismatch" in lb_kinds(text, sheet), text
    assert lb_kinds("TTM revenue growth was 979.3% [F:ttm_revenue_yoy.2026Q2].", sheet) == []


@pytest.mark.unit
def test_r2_5_a_per_share_input_is_not_the_result(sheet):
    decision = {"rating": "Sell", "price_target": 6.08,
                "target_math": "Book value of $6.08 per share × 1.5x P/B = $9.12 per share"}
    assert [f.kind for f in check_target(decision, sheet)] == ["target_math"]


@pytest.mark.unit
def test_r2_6_no_range_across_a_citation(sheet):
    text = "Gross profit of $36.1M [F:gross_profit.2026Q2] − $250.0M [F:opex.2026Q2] opex."
    assert "cited_mismatch" in lb_kinds(text, sheet)
    assert figures(text)[1].range_low is None


@pytest.mark.unit
def test_r2_7_a_line_cited_without_its_period(sheet):
    """Matches a period: a minor format note. Matches none: a mismatch. Not a line: unknown."""
    flags = lint_text("Revenue grew 1,235.4% [F:revenue_yoy] in Q2.", sheet, "research_manager", "rm")
    assert [(f.kind, f.severity) for f in flags] == [("citation_format", "minor")]
    assert lb_kinds("Revenue grew 900% [F:revenue_yoy] in Q2.", sheet) == ["cited_mismatch"]
    assert lb_kinds("Margin of 5% [F:nonsense_line].", sheet) == ["unknown_key"]


@pytest.mark.unit
def test_r2_8_a_spaced_dash_is_punctuation(sheet):
    assert figures("Q1 net income – $362.8M [F:net_income.2026Q1] – came")[0].value == 362.8e6
    assert lb_kinds("Q1 net income – $362.8M [F:net_income.2026Q1] – came from a warrant gain.", sheet) == []
    assert figures("OCF was −$86.1M")[0].value == -86.1e6


@pytest.mark.unit
def test_r2_9_a_net_figure_does_not_double_count(sheet):
    assert "cited_mismatch" in lb_kinds("Net cash (derived) of $2.1B [F:cash.2026Q2][F:sti.2026Q2].", sheet)
    cash_sti = sheet.value("cash_sti.2026Q2")
    assert lb_kinds(f"Net cash (derived) of ${cash_sti / 1e9:.2f}B [F:cash.2026Q2][F:sti.2026Q2].", sheet) == []


@pytest.mark.unit
def test_r2_10_figures_are_frozen_and_pathological_input_is_bounded(sheet):
    import dataclasses

    from tradingagents.quality import lint as lint_module

    fig = figures("$83.8M")[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        fig.value = 1.0
    text = "Values " + ", ".join(f"${i}.3M [F:rnd.2026Q2]" for i in range(1500)) + " less $5M."
    figs = figures(text)
    assert max(len(f.sentence) for f in figs) <= lint_module._MAX_SENTENCE + 20
    assert len(figures("$1M " * (lint_module._MAX_FIGURES + 10))) == lint_module._MAX_FIGURES


@pytest.mark.unit
def test_reader_voice_ignores_ordinary_financial_prose(sheet):
    # Review of the AMD fixes (2026-10-10): each of these is ordinary
    # third-person prose or quoted material and must not cost a revision.
    clean = [
        "The I/O die moves to 3nm next year.",
        "The Phase I trial reads out in the US next quarter; Fund I raised $2B.",
        "King Charles I and Schedule I filings are irrelevant here.",
        "Shares fell 6% on Monday after the downgrade.",
        "Earnings are due on Tuesday, November 4.",
        "Weekend sales of the console beat the prior launch; Thanksgiving weekend demand held.",
        "The investor, Elliott Management, disclosed a stake before the investor day.",
        "Management said \"we expect gross margin near 54%\" and the CEO said “I think demand is durable.”",
        "My Size Inc. and Our Next Energy are not covered; the 'For You' feed grew.",
    ]
    for text in clean:
        for field in ("digest.headline", "digest.news_excerpt"):
            assert not [f for f in lint_text(text, sheet, "digest", field) if f.kind == "voice"], (text, field)
    # Excerpts and lenses are leads, not holds.
    flags = [f for f in lint_text("I'd trim into strength.", sheet, "digest", "digest.risk_neutral") if f.kind == "voice"]
    assert flags and all(f.severity == "minor" for f in flags)


@pytest.mark.unit
def test_rating_words_and_margin_words_ignore_other_meanings(sheet):
    def kinds_for(digest, rating="Overweight"):
        state = {"fact_sheet": None, "portfolio_decision": {"rating": rating}, "report_digest": digest}
        return [(f["kind"], f["field"]) for f in lint_state(state)["flags"]
                if f["kind"] in ("rating_word", "margin_wording")]
    assert kinds_for({"news_excerpt": "Morgan Stanley cut it to Underweight on valuation.",
                      "headline": "Overweight: BUY-side demand outruns SELL-side caution",
                      "ruling": "Analysts at one broker moved the stock to Underweight; the bull case still won."}) == []
    assert kinds_for({"headline": "HOLD: valuation balances growth"}, rating="Hold") == []
    margin = {"conviction": 60, "ruling": ("AMD has a narrow moat in client chips and gross margin expanded "
                                           "slightly; the decisive factor is MI400 timing, a clear-cut gap.")}
    assert kinds_for(margin) == []
    margin["ruling"] = "The bull case won decisively on cash generation."
    assert kinds_for(margin) == [("margin_wording", "digest.ruling")]
    margin.update(conviction=30, ruling="The bear side narrowly carried the debate.")
    assert kinds_for(margin) == []


@pytest.mark.unit
def test_check_8_second_review_cases(sheet):
    def voice(text, field="digest.ruling"):
        return [f for f in lint_text(text, sheet, "digest", field) if f.kind == "voice"]
    for text in ("Overall I think the bear wins.", "Here I see limited upside.", "Stage trims on Monday.",
                 "I'd add.", "The bull point: I would add."):
        assert voice(text, "digest.bull_points" if "point" in text else "digest.ruling"), text
    assert all(f.severity == "load_bearing" for f in voice("I would add.", "digest.bull_points"))

    def flagged(digest, rating="Underweight"):
        state = {"fact_sheet": None, "portfolio_decision": {"rating": rating}, "report_digest": digest}
        return {f["kind"] for f in lint_state(state)["flags"]}
    for headline in ("AMD beat Q3 estimates slightly, but guidance disappoints.", "Earnings beat marginally.",
                     "The bull side cites slightly firmer margins.", "Margins favor a slightly lower multiple.",
                     "The bear won; margins slightly lower.", "The bear wins: guidance barely covers capex."):
        assert "margin_wording" not in flagged({"conviction": 62, "headline": headline}), headline
    assert "margin_wording" in flagged({"conviction": 62, "ruling": "The bear case carried the debate narrowly."})
    assert "rating_word" in flagged({"ruling": "SELL into strength as guidance was cut."})
    assert "rating_word" in flagged({"ruling": "The stock moved; SELL now."})
    assert "rating_word" not in flagged({"ruling": "One broker cut the stock to Overweight from Buy."})


@pytest.mark.unit
def test_trigger_headroom_and_target_debt():
    from tradingagents.quality.lint import check_target_debt, check_trigger_headroom

    digest = {"exit_triggers": [
        {"title": "Margin slips", "detail": "d", "metric": "Operating margin", "current": "34.0%", "threshold": "below 34.0%"},
        {"title": "Cloud slows", "detail": "d", "metric": "Cloud growth", "current": "32%", "threshold": "below 25%"},
    ]}
    flags = check_trigger_headroom(digest)
    assert [f.kind for f in flags] == ["trigger_headroom"] and "Operating margin" in flags[0].quote
    sheet = Facts({"quarters": [{"calendar": "2026Q2", "end": "2026-06-30"}], "facts": [
        {"key": "debt.2026Q2", "value": 98.17e9, "unit": "usd"},
        {"key": "debt_current.2026Q2", "value": 2.0e9, "unit": "usd"}]})
    lt_only = {"target_math": "10x × $400B = $4,000B EV; + $155B cash − $98.17B long-term debt = $4,056.83B; "
                              "÷ 12.1B shares = $335.27"}
    assert [f.kind for f in check_target_debt(lt_only, sheet)] == ["target_debt"]
    total = {"target_math": lt_only["target_math"].replace("$98.17B long-term debt", "$100.17B debt")}
    assert check_target_debt(total, sheet) == []


@pytest.mark.unit
def test_the_digest_gets_the_fact_sheet_flags():
    from tradingagents.graph.digest import build_digest_prompt

    prompt = build_digest_prompt({"company_of_interest": "GOOGL", "trade_date": "2026-10-09",
                                  "fact_sheet": {"flags": ["Cash + short-term investments moved $115.63B"],
                                                 "unavailable": ["10-year Treasury yield unavailable"]}})
    assert "Fact-sheet flags and data gaps" in prompt and "$115.63B" in prompt and "Unavailable: 10-year" in prompt


@pytest.mark.unit
def test_a_flagged_one_off_quarter_used_as_a_base_is_a_lead():
    from tradingagents.quality.lint import check_one_off_bases

    state = {"fact_sheet": {"flags": ["2025Q2 gross margin (39.8%) is 13 points below the median of the other quarters "
                                      "shown (53.3%): likely a one-off (an inventory charge)."]},
             "final_trade_decision": "Data Center operating income improved from a loss of $155M in 2025Q2 to $2.1B.",
             "report_digest": {"bull_thesis": "Margins recovered from 2025Q2's inventory charge."}}
    flags = check_one_off_bases(state)
    assert [f.field for f in flags] == ["pm"] and flags[0].kind == "one_off_base"


@pytest.mark.unit
def test_an_explicit_growth_formula_is_checked():
    from tradingagents.quality.lint import check_formulas

    bad = "Revenue jumped: ($1.02B / $145.0M − 1) × 100 = 650.0% sequential growth."
    assert [f.kind for f in check_formulas(bad, "fundamentals_analyst", "fundamentals_report")] == ["arithmetic"]
    good = "Revenue jumped: ($1.016B / $145.0M − 1) × 100 = 600.7% sequential growth."
    assert check_formulas("($1.02B / $145.0M − 1) × 100 = 603.4%", "pm", "pm") == []   # right for its inputs
    # Rounded inputs: the exact figures behind them give the stated result.
    assert check_formulas("($1.02B / $145.0M − 1) × 100 = 600.7%", "pm", "pm") == []
    assert check_formulas("($8.3B / $7.6B − 1) × 100 = 8.8%", "pm", "pm") == []
    assert check_formulas(good, "fundamentals_analyst", "fundamentals_report") == []
    assert check_formulas("($93.20B [F:x] / $77.67B − 1) = 20.0%", "pm", "pm") == []
