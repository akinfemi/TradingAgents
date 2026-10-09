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
from tradingagents.agents.rating import parse_rating
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

**Valuation:** derive the price target from the fact sheet, never assert it. Pick one method the figures support (EV/Sales on TTM or run-rate revenue for a company without stable earnings; P/E on TTM EPS where earnings are stable and positive), state the multiple you apply and argue for its level: applying today's multiple only reproduces today's price, so say whether the multiple should expand or compress over the horizon and why, from the fact sheet's growth, margins, cash runway and dilution (for example, decelerating growth and widening losses argue for compression). Show the arithmetic from cited keys to the per-share target: enterprise value, plus cash and short-term investments, minus debt, divided by the share count the sheet's market cap uses (cover-page shares, or the weighted count it names when the cover count isn't usable). Without an EV on the sheet (debt not tagged), use a method that doesn't need one, such as P/E or price-to-book from cited figures. The target is exactly the number that arithmetic produces, to the cent; never restate it "with rounding" as a different number. Work each case once, start to finish: don't compute a value and then replace it with another. Give bear and bull case values with the same method and a different multiple each, stating the assumption behind each; the bear case is below the target and the bull case above it. The rating's direction and the target must agree (a buy-side rating has a target above the price; a sell-side one below). If the figures can't support a target, leave it empty and say why.

{"**This is a revision.** The review errata at the top list what failed review. Fix every item that concerns the decision (the rating, target, stop, exit or sizing): restate each corrected value explicitly, and do not repeat a figure the errata mark wrong." if state.get("review_errata") else ""}

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

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
