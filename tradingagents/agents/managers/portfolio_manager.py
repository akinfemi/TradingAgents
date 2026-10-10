"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call. Its rating is the run's
``final_rating``, and the decision is rendered to markdown as
``final_trade_decision`` for the memory log, CLI display and saved reports.
When a provider does not expose structured output, the agent falls back to
free-text generation and the rating is read from that text.
"""

from __future__ import annotations

from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
    rating_horizon,
)
from tradingagents.agents.rating import band_text, parse_rating
from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.structured import NO_EXTERNAL_TOOLS, bind_structured, invoke_structured
from tradingagents.dataflows.config import get_config


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        portfolio_context = get_portfolio_context_from_state(state)
        benchmark, horizon_days, horizon_words = rating_horizon(
            str(state["company_of_interest"]), get_config()
        )

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

{portfolio_context}

---

**Rating horizon:** rate the instrument's expected performance relative to {benchmark} over the next {horizon_words} ({horizon_days} trading days). The call is graded on exactly that: its return minus {benchmark}'s over the window. A price target, if you give one, is for the end of that window. How to get in or out (over how many sessions, at what levels) is execution timing, not the horizon.

**Valuation:** derive the price target from the fact sheet, never assert it. Pick one method the figures support (EV/Sales on TTM or run-rate revenue for a company without stable earnings; P/E on TTM EPS where earnings are stable and positive), state the multiple you apply and argue for its level: applying today's multiple only reproduces today's price, so say whether the multiple should expand or compress over the horizon and why, from the fact sheet's growth, margins, cash runway and dilution (for example, decelerating growth and widening losses argue for compression). Anchor it in the stock's own history when the fact sheet gives one (Valuation history: low, median, high and today's percentile): say where today's multiple sits in that range and where your target multiple sits, and why it belongs there. For a company with operating profit, state the P/E on the fact sheet as a cross-check even when your method is EV/Sales. Keep one basis: every multiple you quote for the target, the cases and "today" is on your method's basis (if the method uses annualized latest-quarter revenue, today's multiple is the run-rate one, not the TTM one); the history on the fact sheet is on TTM figures, so compare like with like and say so. The bear case uses the stock's own history (its median or low) or says why that history no longer applies (a changed business mix, a different growth rate). The target is for the end of the window, after the next report: apply the multiple to the rolled-forward base on the fact sheet (ttm_next.*: TTM with the next quarter at the latest quarter's level) or to your own explicit next-quarter scenario, never to today's TTM held constant. Where the fact sheet flags a likely one-off quarter or potential dilution, account for it in the growth you cite and in the share count. Show the arithmetic from cited keys to the per-share target: enterprise value, plus cash and short-term investments, minus debt, divided by the share count the sheet's market cap uses (cover-page shares, or the weighted count it names when the cover count isn't usable). Without an EV on the sheet (debt not tagged), use a method that doesn't need one, such as P/E or price-to-book from cited figures. The target is exactly the number that arithmetic produces, to the cent; never restate it "with rounding" as a different number. Work each case once, start to finish: don't compute a value and then replace it with another. Give bear and bull case values with the same method and a different multiple each, stating the assumption behind each; the bear case is below the target and the bull case above it. The rating's direction and the target must agree (a buy-side rating has a target above the price; a sell-side one below). If the figures can't support a target, leave it empty and say why.

{"**This is a revision.** The review errata at the top list what failed review. Fix every item that concerns the decision (the rating, target, stop, exit or sizing): restate each corrected value explicitly, and do not repeat a figure the errata mark wrong." if state.get("review_errata") else ""}

**Rating Scale** (use exactly one). Ratings are expected performance against {benchmark} over the horizon, as the report's disclosure defines them; the price target's move from the last close must sit in the rating's band:
- **Buy**: significantly outperform; target move {band_text("Buy")}
- **Overweight**: modestly outperform; target move {band_text("Overweight")}
- **Hold**: perform broadly in line; target move {band_text("Hold")}
- **Underweight**: modestly underperform; target move {band_text("Underweight")}
- **Sell**: significantly underperform; target move {band_text("Sell")}
If the target lands outside the rating's band, change one of them so they agree; never leave a modest rating on a large move or the reverse.

**Catalyst:** a call on a {horizon_words} horizon needs a reason the price moves inside the window. Name the dated catalyst inside it (the next earnings date on the fact sheet, or a dated event in the news) and what you expect it to show, in figures where you can (growth, margin, guidance against the latest quarter). A case that rests on valuation alone, with no catalyst view, is a Hold: valuations rarely correct to a target within {horizon_words} without news. For that Hold, set the target from the multiple you expect by the end of the window (near today's, since nothing in it moves the stock), so the target stays inside the Hold band, or leave the target empty and say why.

**Price levels:** no entry or exit band narrower than the 14-day ATR on the fact sheet; levels finer than the stock's daily range are false precision. Exit thresholds sit with deliberate headroom from today's value (a margin trigger at today's margin fires on any decline), unless you say it is meant as a tripwire.

**Thesis:** open the investment thesis with the investment case itself (what drives the stock over the window, with its numbers), never with which analysts argued better or with sizing mechanics; leave data notes (which share count, which tag) out of the thesis. When your target adds cash and investments, use the fact sheet's split: add cash and debt securities, and treat marketable equity or other stakes separately and say how; use the same debt (long-term plus current) as the fact sheet's EV.

**Voice:** this is a published research note, not advice to one person. Write in the third person about the stock and the call ("the plan trims exposure in stages"), never "I", "we", "you" or "the investor", and never refer to the investor's own target, holdings or loss budget. Say when in market terms ("over the next five sessions", "before the November 4 report"), never relative to today ("after the weekend", "tomorrow", "Monday"). When your valuation multiple uses a different basis from one the debate cites (annualized latest quarter against TTM, say), name both and say why you chose yours. In "what would change it", give at least one fundamental threshold with its current value from the fact sheet (segment growth, gross or operating margin, free cash flow, guidance), not only price levels. These are conditions that would make the rating wrong and move it, not ones that confirm it (for an Underweight, what would move it to Hold or above). Check any year-on-year threshold against the next report's comparison base on the fact sheet: a threshold the business clears by standing still tests nothing. The "if flat" figures are hurdles for tests, not your base case; say what the base case expects separately. Compare quarterly thresholds like with like: a seasonal metric (margin, revenue, cash flow) against the same quarter a year earlier or the TTM, not against the latest quarter.

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}

---

Ground every conclusion in specific evidence from the analysts. The risk debate always contains conflicting stances; deciding which is stronger is the job, so conflict alone is not a reason to Hold. Commit to the stronger case, sized by how decisively it wins. Choose Hold only when the evidence is still balanced after that weighing, or too thin to support a call; do not force a direction to appear decisive. Weigh the analysts on their merits, independent of speaking order.

## Output

Write these sections, in this order, starting with the rating on its own line:

- **Rating**: exactly one of Buy / Overweight / Hold / Underweight / Sell
- **Executive Summary**: the call and how to act on it
- **Valuation**: the method, the inputs (fact keys), the target math, and the bear and bull case values
- **Execution Timing**: how to enter or exit, e.g. "build over 3-5 sessions"
- **Investment Thesis**: the evidence that decided it, and what would change it

{NO_EXTERNAL_TOOLS}{get_language_instruction()}"""

        # The typed rating is the decision; the rendered text only carries it.
        # Read back from text, a rating the thesis quotes could replace it.
        decision = invoke_structured(structured_llm, prompt, "Portfolio Manager")
        if decision is not None:
            final_trade_decision = render_pm_decision(
                decision, f"{horizon_words} ({horizon_days} trading days) vs {benchmark}"
            )
            final_rating = decision.rating.value
        else:
            final_trade_decision = llm.invoke(prompt).content
            final_rating = parse_rating(final_trade_decision)

        new_risk_debate_state = {
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
            "final_rating": final_rating,
            # The typed decision, kept for report surfaces (tickeragent.ai).
            "portfolio_decision": decision.model_dump(mode="json") if decision is not None else None,
        }

    return portfolio_manager_node
