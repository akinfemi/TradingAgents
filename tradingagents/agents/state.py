import operator
from typing import Annotated

from langgraph.graph import MessagesState
from typing_extensions import TypedDict


# Researcher team state
class InvestDebateState(TypedDict):
    bull_history: Annotated[
        str, "Bullish Conversation history"
    ]
    bear_history: Annotated[
        str, "Bearish Conversation history"
    ]
    history: Annotated[str, "Conversation history"]
    current_response: Annotated[str, "Latest response"]
    count: Annotated[int, "Length of the current conversation"]


# Risk management team state
class RiskDebateState(TypedDict):
    aggressive_history: Annotated[
        str, "Aggressive Agent's Conversation history"
    ]
    conservative_history: Annotated[
        str, "Conservative Agent's Conversation history"
    ]
    neutral_history: Annotated[
        str, "Neutral Agent's Conversation history"
    ]
    history: Annotated[str, "Conversation history"]
    latest_speaker: Annotated[str, "Analyst that spoke last"]
    current_aggressive_response: Annotated[
        str, "Latest response by the aggressive analyst"
    ]
    current_conservative_response: Annotated[
        str, "Latest response by the conservative analyst"
    ]
    current_neutral_response: Annotated[
        str, "Latest response by the neutral analyst"
    ]
    count: Annotated[int, "Length of the current conversation"]


class AgentState(MessagesState):
    company_of_interest: Annotated[str, "Company that we are interested in trading"]
    asset_type: Annotated[str, "Asset type under analysis such as stock or crypto"]
    instrument_context: Annotated[str, "Deterministic ticker identity resolved at run start"]
    fact_sheet: Annotated[dict | None, "The run's fact sheet (quality.facts.FactSheet as JSON), built in code at run start"]
    fact_sheet_text: Annotated[str, "The fact sheet rendered for prompts, with a key on every figure"]
    # R5 stage gates: errors still open after a stage's fix-up turn, shown to
    # every later stage; and one record per gate. Appended to, never replaced
    # (the analysts' gates run side by side).
    open_errata: Annotated[list, operator.add]
    # R6 revisions: the review errata every re-run stage sees first, and the
    # previous pass's outputs for stages before the restart (returned as-is).
    review_errata: Annotated[str, "Rendered errata from the failed review, empty on a first pass"]
    kept: Annotated[dict, "Stage group -> the previous pass's output, for stages before the restart"]
    quality_gates: Annotated[list, operator.add]
    trade_date: Annotated[str, "The analysis date; data is served as of it"]

    # research step
    market_report: Annotated[str, "Report from the Market Analyst"]
    extra_sentiment_blocks: Annotated[
        list,
        "(source_name, block_text) pairs from user-connected sources, fetched once "
        "before the run and shown to the Sentiment Analyst as extra data blocks",
    ]
    sentiment_report: Annotated[str, "Report from the Sentiment Analyst"]
    sentiment_structured: Annotated[
        dict | None, "The Sentiment Analyst's typed report as a JSON dict; None on free-text fallback"
    ]
    news_report: Annotated[str, "Report from the News Analyst on company and world news"]
    fundamentals_report: Annotated[str, "Report from the Fundamentals Analyst"]

    # researcher team discussion step
    investment_debate_state: Annotated[
        InvestDebateState, "Current state of the debate on if to invest or not"
    ]
    investment_plan: Annotated[str, "Investment plan from the Research Manager"]

    trader_investment_plan: Annotated[str, "Transaction proposal from the Trader"]

    # risk management team discussion step
    risk_debate_state: Annotated[
        RiskDebateState, "Current state of the debate on evaluating risk"
    ]
    final_trade_decision: Annotated[str, "Final decision from the Portfolio Manager"]
    final_rating: Annotated[str, "The Portfolio Manager's 5-tier rating, or REVIEW when it has none"]
    portfolio_decision: Annotated[
        dict | None, "The Portfolio Manager's typed decision as a JSON dict; None on free-text fallback"
    ]
    past_context: Annotated[str, "Memory log context for the Portfolio Manager (same-ticker decisions + cross-ticker lessons), written by the Memory Log step"]
    memory_note: Annotated[str, "What the Memory Log step could not settle or read this run, for the report; empty when all went well"]
    portfolio_context: Annotated[str, "Caller-supplied holdings and cash, rendered at run start; empty when not provided"]
