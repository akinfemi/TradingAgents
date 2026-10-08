"""The linter (REPORT_QUALITY_PLAN R5, Layer 2): pure functions over stage
text and the run's fact sheet. No model calls.

Checks (numbered as in the plan):

1. number match: a figure cited by key must match that key within
   tolerance; an uncited figure must match some fact, or it is unsupported;
2. arithmetic: "from A to B (+p%)" must reproduce p;
3. period and concept guard: "from A to B" must compare one concept over
   one kind of period (a quarter against a quarter, not a fiscal year);
4. direction words: "bearish cross", "above the 200-day", "oversold" must
   agree with the computed technicals;
5. quote provenance: figures quoted "from the fundamentals report" must be
   in that report;
6. plan coherence: the target's side of the price agrees with the rating;
7. sanitiser and data policy: emoji, sign-offs, truncated fields, and
   data the platform may not use (short interest, consensus, targets).

Severity: ``load_bearing`` when the location is load-bearing (the
headline, the PM's summary or thesis, the RM ruling, bull and bear key
points, exit triggers, the price target, Key Numbers); otherwise
``minor``; ``style`` for sanitiser findings. Whether a flag BLOCKS a stage
(the gate's fix-up turn) depends on its kind, not its severity: an error
in an analyst report is the one every later stage repeats.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

# ---- the flag ----------------------------------------------------------------------

BLOCKING_KINDS = {"cited_mismatch", "unknown_key", "direction", "misattributed", "arithmetic",
                  "period_mismatch", "concept_mismatch", "data_policy", "target_direction"}

LOAD_BEARING_FIELDS = {
    "digest.headline", "digest.bull_thesis", "digest.bear_thesis", "digest.ruling",
    "digest.exit_triggers", "pm", "pm.executive_summary", "pm.investment_thesis",
    "pm.price_target", "rm", "key_numbers",
}


@dataclass
class LintFlag:
    severity: str          # load_bearing | minor | style
    kind: str
    stage: str
    field: str
    quote: str
    expected: str | None = None
    fact_keys: list[str] = field(default_factory=list)

    @property
    def blocking(self) -> bool:
        return self.kind in BLOCKING_KINDS

    def as_dict(self) -> dict:
        out = asdict(self)
        out["blocking"] = self.blocking
        return out


def _severity(field_name: str, kind: str) -> str:
    if kind in ("style",):
        return "style"
    return "load_bearing" if field_name in LOAD_BEARING_FIELDS else "minor"


# ---- facts -----------------------------------------------------------------------------


class Facts:
    """The fact sheet as the linter reads it: {key: fact dict}."""

    def __init__(self, sheet: dict | None):
        self.sheet = sheet or {}
        self.by_key = {f["key"]: f for f in self.sheet.get("facts") or []}

    def __bool__(self) -> bool:
        return bool(self.by_key)

    def get(self, key: str) -> dict | None:
        return self.by_key.get(key)

    def value(self, key: str):
        f = self.by_key.get(key)
        return None if f is None else f.get("value")

    def numeric(self):
        for f in self.by_key.values():
            if isinstance(f.get("value"), (int, float)) and not isinstance(f.get("value"), bool):
                yield f


# ---- figures in text ---------------------------------------------------------------------

_SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mm": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9,
          "billion": 1e9, "t": 1e12, "trillion": 1e12}
_NUM = r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_FIGURE = re.compile(
    r"(?P<neg>[−\-–]|\()?\s?"
    r"(?:(?P<cur>\$)\s?" + _NUM + r"\s?(?P<scale>trillion|billion|million|thousand|bn|mm|[kmbt])?\b"
    r"|" + _NUM.replace("num", "num2") + r"\s?(?P<unit>%|pp|x|×)(?![\w])"
    r")",
    re.IGNORECASE,
)
_CITE = re.compile(r"\[F:([^\]\s]+)\]")


@dataclass
class Figure:
    value: float
    kind: str              # usd | pct | x
    raw: str
    start: int
    end: int
    keys: list[str]
    sentence: str
    step: float = 0.0             # the precision written: "$5" → 1.0, "$7.00" → 0.01, "$83.8M" → 0.1e6
    range_low: float | None = None  # "42–49%": this figure is 49 and the range starts at 42
    at_least: bool = False        # "1,200%+": a floor, not a value


def _sentence_at(text: str, start: int, end: int) -> str:
    left = max(text.rfind(". ", 0, start), text.rfind("\n", 0, start))
    right_candidates = [i for i in (text.find(". ", end), text.find("\n", end)) if i != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1: right + 1].strip()


def figures(text: str) -> list[Figure]:
    """Money, percentages and multiples in ``text``, each with the [F:…] keys
    cited right after it (before the next figure)."""
    out: list[Figure] = []
    matches = list(_FIGURE.finditer(text or ""))
    for i, m in enumerate(matches):
        num = m.group("num") or m.group("num2")
        try:
            value = float(num.replace(",", ""))
        except (TypeError, ValueError):
            continue
        if m.group("cur"):
            kind = "usd"
            value *= _SCALE.get((m.group("scale") or "").lower(), 1.0)
        else:
            unit = m.group("unit").lower()
            kind = "pct" if unit in ("%", "pp") else "x"
        neg = m.group("neg")
        # "42-49%": a dash right after a number joins a range; it is not a sign.
        range_dash = bool(neg) and neg.strip() in ("-", "–", "−") and bool(re.search(r"\d\s?%?\s?$", text[:m.start()]))
        if neg and neg.strip() in ("−", "-", "–", "(") and not range_dash:
            value = -value
        nxt = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        tail = text[m.end(): min(nxt, m.end() + 80)]
        keys = _CITE.findall(tail)
        decimals = len(num.split(".")[1]) if "." in num else 0
        step = 10 ** -decimals
        if kind == "usd":
            step *= _SCALE.get((m.group("scale") or "").lower(), 1.0)
        low = None
        before = text[max(0, m.start() - 16): m.end() - len(m.group(0).lstrip("−-–( "))] if range_dash \
            else text[max(0, m.start() - 16): m.start()]
        rng = re.search(r"\$?(\d[\d,]*(?:\.\d+)?)\s?(?:%|[kmbt])?\s?(?:[-–—]|to)\s?$", before, re.I)
        if rng:
            try:
                low = float(rng.group(1).replace(",", "")) * (
                    _SCALE.get((m.group("scale") or "").lower(), 1.0) if kind == "usd" else 1.0)
            except ValueError:
                low = None
        out.append(Figure(value=value, kind=kind, raw=m.group(0).strip(), start=m.start(), end=m.end(),
                          keys=keys, sentence=_sentence_at(text, m.start(), m.end()), step=step,
                          range_low=low, at_least=text[m.end(): m.end() + 1] == "+"))
    return out


def _fact_as(f: dict, kind: str) -> float | None:
    """A fact's value in a figure's terms (usd, pct, x), or None if the units differ."""
    v = f.get("value")
    unit = f.get("unit")
    if not isinstance(v, (int, float)):
        return None
    if kind == "usd" and unit in ("usd", "usd_per_share"):
        return float(v)
    if kind == "pct" and unit == "pct":
        return float(v)
    if kind == "x" and unit in ("x", "ratio"):
        return float(v)
    return None


def _close(a: float, b: float, kind: str) -> bool:
    """Tolerances (plan § Layer 2): money ±0.5% or ±$0.1M; percentages ±0.2pp
    (or ±1% relative for large percentages, which are printed rounded);
    multiples ±0.1x."""
    if kind == "usd":
        return abs(a - b) <= max(abs(b) * 0.005, 0.1e6 if abs(b) >= 1e6 else 0.005)
    if kind == "pct":
        return abs(a - b) <= max(0.2, abs(b) * 0.01)
    return abs(a - b) <= 0.1


def _matches_value(fig: Figure, target: float) -> bool:
    for a, b in ((fig.value, target), (abs(fig.value), abs(target))):  # signs are often written in words
        if _close(a, b, fig.kind) or abs(a - b) <= fig.step / 2 + 1e-9:   # rounded as written
            return True
        if fig.range_low is not None and min(fig.range_low, a) - fig.step / 2 <= b <= max(fig.range_low, a) + fig.step / 2:
            return True
        if fig.at_least and b >= a - fig.step / 2:
            return True
    return False


def _matches_fact(fig: Figure, f: dict) -> bool:
    target = _fact_as(f, fig.kind)
    return target is not None and _matches_value(fig, target)


# ---- checks ----------------------------------------------------------------------------------


def check_numbers(text: str, facts: Facts, stage: str, field_name: str) -> list[LintFlag]:
    """Check 1. Cited figures must match their key; a cited key must exist."""
    flags: list[LintFlag] = []
    all_figs = figures(text)
    sentence_figs: dict[str, list[Figure]] = {}
    for g in all_figs:
        sentence_figs.setdefault(g.sentence, []).append(g)
    for fig in all_figs:
        if not fig.keys:
            continue
        known = [k for k in fig.keys if facts.get(k)]
        for k in fig.keys:
            if not facts.get(k):
                flags.append(LintFlag(_severity(field_name, "unknown_key"), "unknown_key", stage, field_name,
                                      fig.sentence[:300], f"no fact {k} on the sheet", [k]))
        if known and not any(_matches_fact(fig, facts.get(k)) for k in known):
            # Keys of another unit (a % cited after a $ figure) don't contradict it.
            comparable = [k for k in known if _fact_as(facts.get(k), fig.kind) is not None]
            # A sum of cited keys ("$741.5M ([F:a]+[F:b])").
            if len(comparable) > 1 and _matches_value(fig, sum(_fact_as(facts.get(k), fig.kind) for k in comparable)):
                continue
            # A citation placed at the end of a sentence may belong to another
            # figure in it ("$83.8M in Q2, with $221M in orders [F:revenue.2026Q2]").
            siblings = [g for g in sentence_figs.get(fig.sentence, []) if g is not fig]
            comparable = [k for k in comparable
                          if not any(_matches_fact(g, facts.get(k)) for g in siblings)]
            if comparable:
                expected = "; ".join(f"{k} = {facts.value(k)}" for k in comparable)
                flags.append(LintFlag(_severity(field_name, "cited_mismatch"), "cited_mismatch", stage,
                                      field_name, fig.sentence[:300], expected, comparable))
    return flags


def unsupported_figures(text: str, facts: Facts, stage: str, field_name: str) -> list[LintFlag]:
    """Check 1, second half: uncited money figures that match no fact. Not
    blocking (news and tools carry real figures the sheet doesn't), but
    load-bearing in load-bearing places."""
    flags: list[LintFlag] = []
    numeric = list(facts.numeric())
    for fig in figures(text):
        if fig.keys or fig.kind != "usd" or abs(fig.value) < 1e6:
            continue
        if any(_matches_fact(fig, f) for f in numeric):
            continue
        flags.append(LintFlag(_severity(field_name, "unsupported"), "unsupported", stage, field_name,
                              fig.sentence[:300], "not on the fact sheet and not cited"))
    return flags


_FROM_TO = re.compile(
    r"from\s+(?P<a>[^,;]{0,40}?\$?[−\-–]?\$?\d[\d,.]*\s?(?:%|x|[kmbt]|million|billion)?)"
    r"(?P<akeys>(?:\s*\[F:[^\]]+\])*)"
    r"[^;]{0,60}?\bto\s+(?P<b>\$?[−\-–]?\$?\d[\d,.]*\s?(?:%|x|[kmbt]|million|billion)?)"
    r"(?P<bkeys>(?:\s*\[F:[^\]]+\])*)"
    r"(?P<pct>\s*\(\s*[+−\-–]?\d[\d,.]*\s?%\s*\))?",
    re.IGNORECASE,
)


def _period_type(key: str) -> str | None:
    period = key.rsplit(".", 1)[-1]
    if key.startswith("ttm_"):
        return "TTM"
    if re.fullmatch(r"FY\d{4}", period):
        return "FY"
    if re.fullmatch(r"\d{4}Q[1-4]", period):
        return "Q"
    return None


def _concept(key: str) -> str:
    base = key.split(".", 1)[0]
    return base[4:] if base.startswith("ttm_") else base


def check_comparisons(text: str, facts: Facts, stage: str, field_name: str) -> list[LintFlag]:
    """Checks 2 and 3 on "from A to B" comparisons."""
    flags: list[LintFlag] = []
    for m in _FROM_TO.finditer(text or ""):
        a_keys = _CITE.findall(m.group("akeys") or "")
        b_keys = _CITE.findall(m.group("bkeys") or "")
        quote = _sentence_at(text, m.start(), m.end())[:300]
        if a_keys and b_keys:
            ka, kb = a_keys[0], b_keys[0]
            pa, pb = _period_type(ka), _period_type(kb)
            if pa and pb and pa != pb and _concept(ka) == _concept(kb):
                flags.append(LintFlag(_severity(field_name, "period_mismatch"), "period_mismatch", stage,
                                      field_name, quote, f"{ka} is {pa}, {kb} is {pb}", [ka, kb]))
            elif pa and pb and _concept(ka) != _concept(kb) and \
                    (facts.get(ka) or {}).get("unit") == (facts.get(kb) or {}).get("unit") == "usd":
                flags.append(LintFlag(_severity(field_name, "concept_mismatch"), "concept_mismatch", stage,
                                      field_name, quote, f"{_concept(ka)} compared with {_concept(kb)}", [ka, kb]))
        if m.group("pct"):
            a_figs, b_figs = figures(m.group("a")), figures(m.group("b"))
            pct_figs = figures(m.group("pct"))
            if a_figs and b_figs and pct_figs and a_figs[0].kind == b_figs[0].kind == "usd" and a_figs[0].value:
                a, b, p = a_figs[0].value, b_figs[0].value, pct_figs[0].value
                actual = (b / a - 1) * 100 if a > 0 else None
                if actual is not None and not _close(p, actual, "pct") and not _close(abs(p), abs(actual), "pct"):
                    flags.append(LintFlag(_severity(field_name, "arithmetic"), "arithmetic", stage, field_name,
                                          quote, f"from {a:,.0f} to {b:,.0f} is {actual:+.1f}%"))
    return flags


_DIRECTION_RULES = [
    # (pattern, fact key, predicate on the fact value -> True when the claim is WRONG, expected text)
    (re.compile(r"\bbearish\s+(?:MACD\s+)?cross(?:over)?\b", re.I), "macd.cross",
     lambda v: v == "bullish", "the last MACD cross was bullish"),
    (re.compile(r"\bbullish\s+(?:MACD\s+)?cross(?:over)?\b", re.I), "macd.cross",
     lambda v: v == "bearish", "the last MACD cross was bearish"),
    (re.compile(r"\bMACD\b[^.]{0,30}\bbelow\s+(?:its\s+|the\s+)?signal", re.I), "macd.position",
     lambda v: v == "above signal", "MACD is above its signal line"),
    (re.compile(r"\bMACD\b[^.]{0,30}\babove\s+(?:its\s+|the\s+)?signal", re.I), "macd.position",
     lambda v: v == "below signal", "MACD is below its signal line"),
    # A claim about now ("is oversold", "in overbought territory"), not a
    # threshold, a history or a denial ("not yet oversold (<30)").
    (re.compile(r"\b(?:is|are|remains?|now|currently|already|deeply|technically)\s+(?:in\s+)?oversold\b|"
                r"\boversold\s+(?:territory|conditions?|setup|reading)\b(?![^.]{0,20}<)", re.I), "rsi14.value",
     lambda v: isinstance(v, (int, float)) and v > 40, "RSI is not in oversold territory"),
    (re.compile(r"\b(?:is|are|remains?|now|currently|already|deeply|technically)\s+(?:in\s+)?overbought\b|"
                r"\boverbought\s+(?:territory|conditions?|setup|reading)\b(?![^.]{0,20}>)", re.I), "rsi14.value",
     lambda v: isinstance(v, (int, float)) and v < 60, "RSI is not in overbought territory"),
]
_MA_SIDE = re.compile(
    r"\b(?P<side>above|below)\s+(?:its\s+|the\s+)?(?P<n>10|50|200)[-\s]?(?:day|d)?[\s-]*(?:simple\s+|exponential\s+)?"
    r"(?:moving\s+average|SMA|EMA|MA)\b",
    re.I,
)
_HEDGES = re.compile(r"\b(if|would|could|should|unless|risk of|watch for|a break|breaks?|fall|falls|falling|"
                     r"drop|drops|reclaim|reclaims|until|when|previous|prior|earlier|before|not|nor|neither|"
                     r"approaching|threshold|potential|stop|stop-loss|target|entry|level)\b|[<>]", re.I)
# Whose position "above/below the 50-day" describes: the price, not a stop or a target.
_PRICE_SUBJECT = re.compile(r"\b(price|stock|shares|it|trades?|trading|closed?|closes|sits?|is|remains?)\b"
                            r"[^.$\d]{0,30}$", re.I)


def check_directions(text: str, facts: Facts, stage: str, field_name: str) -> list[LintFlag]:
    """Check 4: technical-state words against the computed states. Sentences
    about what might happen ("a break below the 50-day would…") are not
    claims about now and are skipped."""
    flags: list[LintFlag] = []
    for rx, key, wrong, expected in _DIRECTION_RULES:
        value = facts.value(key)
        if value is None:
            continue
        for m in rx.finditer(text or ""):
            sentence = _sentence_at(text, m.start(), m.end())
            if _HEDGES.search(sentence):
                continue
            if wrong(value):
                flags.append(LintFlag(_severity(field_name, "direction"), "direction", stage, field_name,
                                      sentence[:300], expected, [key]))
    for m in _MA_SIDE.finditer(text or ""):
        key = {"10": "ema10", "50": "sma50", "200": "sma200"}[m.group("n")]
        gap = facts.value(f"{key}.gap_pct")
        if not isinstance(gap, (int, float)):
            continue
        sentence = _sentence_at(text, m.start(), m.end())
        if _HEDGES.search(sentence) or not _PRICE_SUBJECT.search(text[max(0, m.start() - 40): m.start()]):
            continue
        claims_above = m.group("side").lower() == "above"
        if claims_above != (gap > 0) and abs(gap) > 0.3:
            flags.append(LintFlag(_severity(field_name, "direction"), "direction", stage, field_name,
                                  sentence[:300],
                                  f"price is {'above' if gap > 0 else 'below'} the {m.group('n')}-day "
                                  f"({gap:+.1f}%)", [f"{key}.gap_pct"]))
    return flags


_ATTRIBUTION = re.compile(
    r"(?:from|per|according to|citing|as)\s+the\s+(?P<src>fundamental|fundamentals|market|technical|news|"
    r"sentiment|social)\s+(?:analysis|analyst|report|data)[^.:]{0,20}[:,]?\s*[\"“](?P<quote>[^\"”]{10,400})[\"”]",
    re.I,
)
_SRC = {"fundamental": "fundamentals", "fundamentals": "fundamentals", "market": "market",
        "technical": "market", "news": "news", "sentiment": "sentiment", "social": "sentiment"}


def check_provenance(text: str, sources: dict[str, str], stage: str, field_name: str) -> list[LintFlag]:
    """Check 5: a quote attributed to an analyst report must be in it, by
    its figures (every figure in the quote appears in the source) or, with
    no figures, by fuzzy match."""
    flags: list[LintFlag] = []
    for m in _ATTRIBUTION.finditer(text or ""):
        source_text = sources.get(_SRC[m.group("src").lower()]) or ""
        if not source_text:
            continue
        quote = m.group("quote")
        quoted_figs = figures(quote)
        if quoted_figs:
            source_figs = figures(source_text)
            missing = [q.raw for q in quoted_figs
                       if not any(q.kind == s.kind and _close(q.value, s.value, q.kind) for s in source_figs)]
            if missing:
                flags.append(LintFlag(_severity(field_name, "misattributed"), "misattributed", stage, field_name,
                                      m.group(0)[:300],
                                      f"{', '.join(missing)} not in the {_SRC[m.group('src').lower()]} report"))
        else:
            ratio = SequenceMatcher(None, quote.lower(), source_text.lower()).find_longest_match(
                0, len(quote), 0, len(source_text)).size / max(len(quote), 1)
            if ratio < 0.6:
                flags.append(LintFlag(_severity(field_name, "misattributed"), "misattributed", stage, field_name,
                                      m.group(0)[:300],
                                      f"quote not found in the {_SRC[m.group('src').lower()]} report"))
    return flags


_POLICY = [
    (re.compile(r"\bshort\s+(?:interest|float|squeeze\s+fuel)\b|\bdays[\s-]to[\s-]cover\b|"
                r"\d[\d.]*\s?%\s+(?:of\s+(?:the\s+)?float\s+)?(?:short(?:ed)?|sold\s+short)\b|"
                r"\bshorted\b", re.I), "short interest"),
    (re.compile(r"\bconsensus\s+(?:EPS|estimate|estimates|revenue|forecast)\b", re.I), "consensus estimates"),
    (re.compile(r"\b(?:analyst|street|consensus|average)\s+(?:price\s+)?targets?\b(?:\s+(?:of|at)\s+\$)?", re.I),
     "analyst price targets"),
]
# Saying the data is NOT used is fine ("short interest is not available").
_POLICY_NEGATED = re.compile(r"\b(not\s+(?:available|used|provided|cited|considered)|unavailable|"
                             r"no\s+(?:data|figures?)\s+on|excluded|missing)\b", re.I)


def check_policy(text: str, stage: str, field_name: str) -> list[LintFlag]:
    """Check 7, data policy: figures the platform may not use."""
    flags: list[LintFlag] = []
    seen: set[str] = set()
    for rx, what in _POLICY:
        for m in rx.finditer(text or ""):
            sentence = _sentence_at(text, m.start(), m.end())
            if _POLICY_NEGATED.search(sentence) or sentence in seen:
                continue
            seen.add(sentence)
            flags.append(LintFlag(_severity(field_name, "data_policy"), "data_policy", stage, field_name,
                                  sentence[:300], f"{what} may not be used"))
    return flags


_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF]")
_SIGNOFF = re.compile(r"(?:\b(?:bull|bear|neutral|aggressive|conservative)\b[^.?!]{0,40}\?\s*$|"
                      r"\b(?:over to you|your move|let me know|I rest my case)\b)", re.I)


def check_style(text: str, stage: str, field_name: str) -> list[LintFlag]:
    """Check 7, sanitiser findings (style severity, never blocking)."""
    flags: list[LintFlag] = []
    if _EMOJI.search(text or ""):
        flags.append(LintFlag("style", "emoji", stage, field_name, _EMOJI.search(text).group(0)))
    tail = (text or "").rstrip()
    if tail.endswith(("…", "...")):
        flags.append(LintFlag("style", "truncated", stage, field_name, tail[-120:]))
    if _SIGNOFF.search(tail[-200:]):
        flags.append(LintFlag("style", "sign_off", stage, field_name, tail[-200:]))
    return flags


def check_target(decision: dict | None, facts: Facts) -> list[LintFlag]:
    """Check 6 (part): the target sits on the rating's side of the price."""
    if not decision:
        return []
    target, rating = decision.get("price_target"), (decision.get("rating") or "").title()
    close = facts.value("price.close")
    if not isinstance(target, (int, float)) or not isinstance(close, (int, float)) or not rating:
        return []
    up = target > close
    if (rating in ("Buy", "Overweight") and not up) or (rating in ("Sell", "Underweight") and up):
        return [LintFlag("load_bearing", "target_direction", "portfolio_manager", "pm.price_target",
                         f"{rating} with a target of {target} against a close of {close}",
                         "a buy-side target above the price, a sell-side one below", ["price.close"])]
    return []


# ---- running it --------------------------------------------------------------------------------


def lint_text(text: str, facts: Facts, stage: str, field_name: str,
              sources: dict[str, str] | None = None) -> list[LintFlag]:
    """Checks 1–5 and 7 on one stage output (what the gates run)."""
    if not text:
        return []
    flags = [
        *check_numbers(text, facts, stage, field_name),
        *check_comparisons(text, facts, stage, field_name),
        *check_directions(text, facts, stage, field_name),
        *check_provenance(text, sources or {}, stage, field_name),
        *check_policy(text, stage, field_name),
        *check_style(text, stage, field_name),
    ]
    if field_name in LOAD_BEARING_FIELDS:
        flags.extend(unsupported_figures(text, facts, stage, field_name))
    return flags


def stage_texts(state: dict) -> list[tuple[str, str, str]]:
    """(stage, field, text) for every stage output in a final state."""
    debate = state.get("investment_debate_state") or {}
    risk = state.get("risk_debate_state") or {}
    out = [
        ("market_analyst", "market_report", state.get("market_report")),
        ("sentiment_analyst", "sentiment_report", state.get("sentiment_report")),
        ("news_analyst", "news_report", state.get("news_report")),
        ("fundamentals_analyst", "fundamentals_report", state.get("fundamentals_report")),
        ("bull_researcher", "bull", debate.get("bull_history")),
        ("bear_researcher", "bear", debate.get("bear_history")),
        ("research_manager", "rm", state.get("investment_plan")),
        ("trader", "trader", state.get("trader_investment_plan")),
        ("risk_debate", "risk", risk.get("history")),
        ("portfolio_manager", "pm", state.get("final_trade_decision")),
    ]
    digest = state.get("report_digest") or {}
    for key in ("headline", "bull_thesis", "bear_thesis", "ruling"):
        if isinstance(digest.get(key), str):
            out.append(("digest", f"digest.{key}", digest[key]))
    triggers = digest.get("exit_triggers")
    if triggers:
        out.append(("digest", "digest.exit_triggers", str(triggers)))
    return [(s, f, t) for s, f, t in out if t]


def lint_state(state: dict) -> dict:
    """The full pass over a finished run: every stage, the digest and the plan."""
    facts = Facts(state.get("fact_sheet"))
    sources = {
        "market": state.get("market_report") or "", "sentiment": state.get("sentiment_report") or "",
        "news": state.get("news_report") or "", "fundamentals": state.get("fundamentals_report") or "",
    }
    flags: list[LintFlag] = []
    for stage, field_name, text in stage_texts(state):
        # An analyst's own report is not checked against itself for provenance.
        own = {k: v for k, v in sources.items() if not stage.startswith(k)}
        flags.extend(lint_text(text, facts, stage, field_name, own))
    flags.extend(check_target(state.get("portfolio_decision"), facts))
    counts: dict[str, int] = {}
    for f in flags:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    return {
        "version": 1,
        "has_fact_sheet": bool(facts),
        "counts": counts,
        "blocking": sum(1 for f in flags if f.blocking),
        "flags": [f.as_dict() for f in flags],
    }
