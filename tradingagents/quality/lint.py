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
   data the platform may not use (short interest, consensus, targets),
   and review vocabulary ("verified", "errata", "fact sheet") in the
   reader-facing digest;
8. reader voice: first person, "the investor", time relative to today,
   the trader's BUY/SELL or another tier's name in reader text, ruling
   words that disagree with the debate margin, and exit triggers with no
   fundamental threshold.

Severity: ``load_bearing`` when the location is load-bearing (the
headline, the PM's summary or thesis, the RM ruling, the digest's bull
and bear theses, exit triggers, the price target); otherwise
``minor``; ``style`` for sanitiser findings. Whether a flag BLOCKS a stage
(the gate's fix-up turn) depends on its kind, not its severity: an error
in an analyst report is the one every later stage repeats.
"""

from __future__ import annotations

import itertools
import re
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass, field, replace
from difflib import SequenceMatcher
from functools import lru_cache

from tradingagents.agents.rating import RATING_BANDS, band_text, tier_for_move

# ---- the flag ----------------------------------------------------------------------

BLOCKING_KINDS = {"cited_mismatch", "unknown_key", "direction", "misattributed", "arithmetic",
                  "period_mismatch", "concept_mismatch", "data_policy", "target_direction", "target_math",
                  "target_range", "target_tier", "social_claim", "process_language"}

LOAD_BEARING_FIELDS = {
    "digest.headline", "digest.bull_thesis", "digest.bear_thesis", "digest.ruling",
    "digest.exit_triggers", "pm", "pm.executive_summary", "pm.investment_thesis",
    "pm.price_target", "rm",
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

    def periods_of(self, line: str) -> list[dict]:
        """The facts of one line across its periods ("revenue_yoy" → revenue_yoy.2026Q2, ...)."""
        return [f for k, f in self.by_key.items() if k.startswith(line + ".")]

    # Scenario figures ("TTM if the next quarter is flat", hurdles, history
    # ranges): citable by key, but never the source of an uncited figure —
    # ONDS's rolled TTM net income (+$86.14M) "supported" a wrong-signed
    # "+$86.1M" of operating cash flow.
    SCENARIO_PREFIXES = ("ttm_next.", "next_base.", "next_hurdle.", "seg_hurdle.")

    def numeric(self, actual_only: bool = False):
        for f in self.by_key.values():
            if not isinstance(f.get("value"), (int, float)) or isinstance(f.get("value"), bool):
                continue
            if actual_only and (f["key"].startswith(self.SCENARIO_PREFIXES) or "_hist." in f["key"]):
                continue
            yield f


# ---- figures in text ---------------------------------------------------------------------

_SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mm": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9,
          "billion": 1e9, "t": 1e12, "trillion": 1e12}
_NUM = r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_SCALES = r"trillion|billion|million|thousand|bn|mm|[kmbt]"
_FIGURE = re.compile(
    r"(?P<neg>[−\-–+]|\()?\s?"
    r"(?:(?P<cur>\$)\s?" + _NUM + r"\s?(?P<scale>" + _SCALES + r")?\b"
    r"|" + _NUM.replace("num", "num2") + r"\s?(?P<unit>%|pp|x|×)(?![\w])"
    r")",
    re.IGNORECASE,
)
# A citation, also "[F: a]", "[F:a, F:b]" and "[F:a; F:b]".
_CITE_RAW = re.compile(r"\[F:\s*([^\[\]]{1,200}?)\s*\]")
_CITE_KEY = re.compile(r"[A-Za-z0-9_][\w.\-]*")


class _Cites:
    """[F:key] citations, with a combined "[F:fcf.2026Q1/2026Q2]" read as the
    two keys it names (fcf.2026Q1 and fcf.2026Q2), and a list "[F:a, F:b]"
    as its keys."""

    def findall(self, text: str) -> list[str]:
        out = []
        for raw in _CITE_RAW.findall(text or ""):
            base = None
            for group in re.split(r"\s*[,;]\s*", raw):
                group = group.strip().removeprefix("F:").strip()
                if group.startswith(".") and base:      # "[F:revenue_yoy.2025Q3, .2025Q4]"
                    group = base + group
                parts = [p.strip().removeprefix("F:").strip() for p in re.split(r"/|→|->", group)]
                parts = [p for p in parts if p]
                if not parts or not all(_CITE_KEY.fullmatch(p) for p in parts):
                    continue                            # prose in brackets, not a key
                base = parts[0].split(".", 1)[0] if "." in parts[0] else base
                if len(parts) > 1 and all("." in p for p in parts):
                    out.extend(parts)                   # [F:ema10.value/F:sma50.value]
                elif len(parts) > 1 and "." in parts[0]:
                    base = parts[0].split(".", 1)[0]   # [F:fcf.2026Q1/2026Q2]
                    out.append(parts[0])
                    out.extend(f"{base}.{p}" for p in parts[1:])
                else:
                    out.append("/".join(parts))
        return out


_CITE = _Cites()


@dataclass(frozen=True)
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
    signed: bool = False          # written with an explicit sign ("−$86.1M", "+$5M")
    in_range: bool = False        # the low end of a range ("$80M" in "$80M–$90M")
    lead: str = ""                # the words just before it in its clause ("a profit of")
    sent_start: int = 0           # where its sentence starts in the text
    scale: float = 1.0            # "$71.61B" → 1e9


class _Sentences:
    """Sentence bounds of one text, computed once (a run-on text of 40k
    characters made per-figure scans quadratic)."""

    def __init__(self, text: str):
        self.text = text
        self.dots = [m.start() for m in re.finditer(r"\. ", text)]
        self.nls = [m.start() for m in re.finditer(r"\n", text)]

    def bounds(self, start: int, end: int) -> tuple[int, int]:
        """(left, right) as ``text[left + 1: right + 1]`` is the sentence."""
        i = bisect_right(self.dots, start - 2) - 1
        j = bisect_right(self.nls, start - 1) - 1
        left = max(self.dots[i] if i >= 0 else -1, self.nls[j] if j >= 0 else -1)
        i = bisect_left(self.dots, end)
        j = bisect_left(self.nls, end)
        rights = [x for x in (self.dots[i] if i < len(self.dots) else None,
                              self.nls[j] if j < len(self.nls) else None) if x is not None]
        return left, (min(rights) if rights else len(self.text))

    def at(self, start: int, end: int) -> str:
        left, right = self.bounds(start, end)
        return self.text[left + 1: right + 1].strip()


def _sentence_at(text: str, start: int, end: int) -> str:
    left = max(text.rfind(". ", 0, start), text.rfind("\n", 0, start))
    right_candidates = [i for i in (text.find(". ", end), text.find("\n", end)) if i != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1: right + 1].strip()


# The figure (or bare number) just before a dash or "and": "$80M" in
# "$80M–$90M", "42" in "42–49%", "$5" in "between $5 and $7".
_LEFT = (r"(?<![\w.,\-–−/:$])(?P<lcur>\$)?\s?(?P<lnum>\d[\d,]*(?:\.\d+)?)\s?"
         r"(?P<lunit>%|pp|x|×|" + _SCALES + r")?")
_LEFT_OF_DASH = re.compile(_LEFT + r"(?:\s*\[F:[^\[\]]*\])*\s*\)?\s*$", re.I)
_LEFT_OF_AND = re.compile(r"\bbetween\s+" + _LEFT + r"\s+and\s+$", re.I)
_PERIOD_LABEL = re.compile(r"(?:\bQ[1-4]|\bFY|\bH[12]|\bCY|\bfiscal)\s*$", re.I)


def _left_figure(m: re.Match | None, before: str, kind: str, scale: float):
    """(joined, low): whether a dash joins this figure to a figure on its
    left (a range or a subtraction, so the dash is not a sign), and the
    range's low end when it is a range of one kind of figure. A year or a
    quarter label ("Q2 2026 −$86.1M") is not a figure."""
    if not m:
        return False, None
    lcur, lnum, lunit = m.group("lcur"), m.group("lnum"), (m.group("lunit") or "").lower()
    if not lcur and not lunit and (re.fullmatch(r"(?:19|20)\d\d", lnum)
                                   or _PERIOD_LABEL.search(before[:m.start("lnum")])):
        return False, None
    lkind = ("pct" if lunit in ("%", "pp") else "x" if lunit in ("x", "×")
             else "usd" if lcur or lunit in _SCALE else kind)
    if lkind != kind:
        return True, None
    try:
        low = float(lnum.replace(",", ""))
    except ValueError:
        return True, None
    if kind == "usd":
        low *= _SCALE.get(lunit, scale)
    return True, low


def _cite_run(text: str, start: int, stop: int, window: int = 80) -> str:
    """The citations after a figure: those that start within ``window``
    characters of it, and the run of citations that continues them
    ("[F:a.2026Q1][F:a.2026Q2]", "[F:a] and [F:b]"), up to the next figure."""
    out, last = [], None
    line_end = text.find("\n", start, stop)
    stop = line_end if line_end != -1 else stop      # the next line's citations are its own
    for c in _CITE_RAW.finditer(text, start, stop):
        if c.start() - start < window or (last is not None and _RUN_GAP.fullmatch(text[last: c.start()])):
            out.append(c.group(0))
            last = c.end()
        else:
            break
    return " ".join(out)


_RUN_GAP = re.compile(r"[\s,;/+&→\-–−]*(?:and|or|vs\.?|to)?[\s,;/+&→\-–−]*")
_CITES_BEFORE = re.compile(r"((?:\[F:[^\[\]]*\][\s,;/+&]*(?:and\s+)?)+)[\s(~≈*:=|—–-]{0,8}$")
_AT_LEAST = re.compile(r"\b(?:over|more than|at least|above|north of|upwards of|in excess of)\s*~?$", re.I)


# Pathological input (a 36k-character sentence with 1,500 figures took 35s).
_MAX_FIGURES = 5000
_MAX_SENTENCE = 1500
_MAX_SIBLINGS = 30


@lru_cache(maxsize=2048)
def _figures(text: str) -> tuple[Figure, ...]:
    out: list[Figure] = []
    sentences = _Sentences(text)
    matches = list(itertools.islice(_FIGURE.finditer(text), _MAX_FIGURES))
    for i, m in enumerate(matches):
        num = m.group("num") or m.group("num2")
        try:
            value = float(num.replace(",", ""))
        except (TypeError, ValueError):
            continue
        scale = 1.0
        if m.group("cur"):
            kind = "usd"
            scale = _SCALE.get((m.group("scale") or "").lower(), 1.0)
            value *= scale
        else:
            unit = m.group("unit").lower()
            kind = "pct" if unit in ("%", "pp") else "x"
        core = m.start("cur") if m.group("cur") else m.start("num2")
        neg = m.group("neg")
        sign_at = m.start("neg") if neg else core
        before = text[max(0, sign_at - 40): sign_at]
        signed, low, joined = False, None, False
        if neg in ("-", "–", "−"):
            # "42-49%", "$80M-$90M", "$83.8M - $31.0M": a dash after a figure
            # joins a range or subtracts; it is not a sign. A sign is attached
            # ("−$86.1M"); "net income – $362.8M [F:…] – came from" is punctuation.
            left_match = _LEFT_OF_DASH.search(before)
            joined, low = _left_figure(left_match, before, kind, scale)
            if left_match and "[F:" in left_match.group(0):
                low = None                     # "$36.1M [F:a] − $250.0M": a subtraction, never a range
            if not joined and m.end("neg") == core:
                value, signed = -value, True
        elif neg == "+":
            signed = m.end("neg") == core      # "+$86.1M", not "$83.8M + $5M"
        elif neg == "(":
            # Accounting brackets are a negative only for money in a table
            # ("| ($86.1M) |"); in prose "($83.8M)", or "67.2% (69.0%)", it is an aside.
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_end = text.find("\n", m.end())
            line = text[line_start: line_end if line_end != -1 else len(text)]
            if kind == "usd" and "|" in line and text[m.end(): m.end() + 1] == ")":
                value, signed = -value, True
        else:
            joined, low = _left_figure(_LEFT_OF_AND.search(before), before, kind, scale)
        if low is not None and not low <= abs(value):
            low = None                         # "$83.8M - $31.0M": a subtraction, not a range
        if low is not None and out and 0 <= sign_at - out[-1].end <= 6:
            out[-1] = replace(out[-1], in_range=True)
        nxt = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        keys = _CITE.findall(_cite_run(text, m.end(), nxt))
        decimals = len(num.split(".")[1]) if "." in num else 0
        step = 10 ** -decimals
        if kind == "usd":
            step *= scale
        left, right = sentences.bounds(m.start(), m.end())
        sent_start = left + 1
        if right - left > _MAX_SENTENCE:      # a run-on "sentence": read a window around the figure
            left = max(left, m.start() - _MAX_SENTENCE // 2)
            right = min(right, m.end() + _MAX_SENTENCE // 2)
        lead = text[max(left + 1, sign_at - 40): sign_at]
        lead = re.split(r"[;:,(\]]", lead)[-1]
        out.append(Figure(value=value, kind=kind, raw=m.group(0).strip(), start=m.start(), end=m.end(),
                          keys=keys, sentence=text[left + 1: right + 1].strip(), step=step,
                          range_low=low, signed=signed,
                          at_least=text[m.end(): m.end() + 1] == "+" or bool(_AT_LEAST.search(text[max(0, sign_at - 20): sign_at])),
                          lead=lead, sent_start=sent_start, scale=scale))
    return tuple(out)


def figures(text: str) -> list[Figure]:
    """Money, percentages and multiples in ``text``, each with the [F:…] keys
    cited right after it (before the next figure)."""
    return list(_figures(text or ""))


def _fact_as(f: dict, kind: str) -> float | None:
    """A fact's value in a figure's terms (usd, pct, x), or None if the units differ."""
    v = f.get("value")
    unit = f.get("unit")
    if not isinstance(v, (int, float)) or isinstance(v, bool):
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


# Line items that are signed by nature (a profit or a loss, an inflow or an
# outflow); costs such as capex are stored positive and often written "−$7.8M".
_SIGNED_CONCEPT = re.compile(r"income|ocf|fcf|eps|margin|non_operating|cash_flow|earnings|ebit|_yoy|gap|macd|"
                             r"from_high|from_low|change|return", re.I)
_SAYS_POSITIVE = re.compile(r"\b(?:profit(?:able)?|positive|inflows?|gains?|surplus|generat(?:ed|ing|es))\b", re.I)
_SAYS_NEGATIVE = re.compile(r"\b(?:loss(?:es)?|negative|outflows?|deficit)\b", re.I)
# The polarity word must govern the figure: "a profit of $88.4M", "turned
# positive at +$86.1M", not "deep losses and 2.6% share growth".
_GOVERNS = r"(?:\s+(?:of|at|to|was|were|is|reached|totall?ing|about|approximately|roughly|nearly|around|some|a|an|the|net|"
_GOVERNS += r"quarterly|annual|operating|free|cash|flow|income))*\s*[:~≈]?\s*$"
_SAYS_POSITIVE_AT = re.compile(_SAYS_POSITIVE.pattern + _GOVERNS, re.I)
_SAYS_NEGATIVE_AT = re.compile(_SAYS_NEGATIVE.pattern + _GOVERNS, re.I)
_SAYS_NOT = re.compile(r"\b(?:not|no|never|yet|until|before|instead|from|versus|vs\.?|rather)\b", re.I)


def _sign_conflict(fig: Figure, f: dict, target: float) -> bool:
    """The figure's written sign, or a polarity word just before it ("a
    profit of $88.4M"), contradicts the fact's sign. An unsigned figure is
    not a conflict: losses are usually written "a loss of $88.4M"."""
    if not target or not fig.value:
        return False
    fact_negative = target < 0
    signed_concept = bool(_SIGNED_CONCEPT.search(f.get("key") or ""))
    if fig.signed and fig.value > 0 and fact_negative:
        return True
    if fig.signed and fig.value < 0 and not fact_negative and signed_concept:
        return True
    if fig.signed or _SAYS_NOT.search(fig.lead):
        return False
    # Only line items signed by nature: a warrant "gain of $15.2M" is filed as −15.25M.
    if fact_negative and signed_concept and _SAYS_POSITIVE_AT.search(fig.lead) and not _SAYS_NEGATIVE.search(fig.lead):
        return True
    return (not fact_negative and signed_concept and bool(_SAYS_NEGATIVE_AT.search(fig.lead))
            and not _SAYS_POSITIVE.search(fig.lead))


def _matches_fact(fig: Figure, f: dict) -> bool:
    target = _fact_as(f, fig.kind)
    return target is not None and not _sign_conflict(fig, f, target) and _matches_value(fig, target)


def _matches_fact_uncited(fig: Figure, f: dict) -> bool:
    """An uncited money figure against one fact on the sheet. Small figures
    must round to the fact as written: with some 200 money facts on a sheet,
    the ±$0.1M floor let a third of invented figures match one."""
    if not _matches_fact(fig, f):
        return False
    if fig.kind == "usd" and abs(fig.value) < 20e6 and fig.range_low is None and not fig.at_least:
        return abs(abs(fig.value) - abs(_fact_as(f, "usd"))) <= fig.step / 2 + 1e-9
    return True


def _derivable(fig: Figure, values: list[float], multiples: list[float] = (), max_terms: int = 3) -> bool:
    """Whether ``fig`` is arithmetic on ``values``: a signed sum of two to
    ``max_terms`` of them, or a product or quotient of two. The stages are
    told to derive figures from cited keys ("equity $1.57B less goodwill
    $661M less intangibles $583M = $325M"); such a result is on no sheet by
    definition."""
    from itertools import combinations, product

    # Callers put the figure's own sentence first; keep the nearest values.
    seen: list[float] = []
    for v in values:
        if isinstance(v, (int, float)) and v and v not in seen:
            seen.append(v)
    vals = seen[:8]

    def hit(x: float) -> bool:
        return any(_close(a, b, fig.kind) or abs(a - b) <= fig.step / 2 + 1e-9
                   for a, b in ((fig.value, x), (abs(fig.value), abs(x)))) or (
            fig.range_low is not None and min(fig.range_low, abs(fig.value)) - fig.step / 2 <= abs(x)
            <= max(fig.range_low, abs(fig.value)) + fig.step / 2)

    for n in range(2, min(max_terms, len(vals)) + 1):
        for combo in combinations(vals, n):
            for signs in product((1, -1), repeat=n - 1):
                if hit(combo[0] + sum(sg * v for sg, v in zip(signs, combo[1:], strict=True))):
                    return True
    for a, b in combinations(vals, 2):
        for x in (a * b, a / b, b / a):
            if hit(x):
                return True
    # A multiple applied to a figure ("13x × $174.1M TTM revenue = $2,263M").
    return any(hit(v * m) for v in vals for m in multiples if m)


def _previous_sentence(text: str, fig: Figure) -> str:
    if fig.sent_start <= 1:
        return ""
    return _Sentences(text[max(0, fig.sent_start - 1500): fig.sent_start]).at(
        min(1500, fig.sent_start) - 1, min(1500, fig.sent_start) - 1)


def _context_values(text: str, fig: Figure, facts: Facts) -> list[float]:
    """Values ``fig`` may be derived from: its own sentence first, then the
    sentence before it (a result is often stated a sentence after its
    inputs: "…$49.3B. That is a fall of $23.65B")."""
    return (_sentence_values(fig.sentence, facts, fig.kind, fig)
            + _sentence_values(_previous_sentence(text, fig), facts, fig.kind))


def _wide_context_values(text: str, fig: Figure, facts: Facts) -> list[float]:
    """For an uncited figure: its sentence, then the 300 characters before
    it (a result two sentences after its inputs: "…is $27.32B. In 2025Q3 it
    was … $50.97B. That is a fall of $23.65B")."""
    before = text[max(0, fig.start - 300): fig.start]
    nearest_first = [g.value for g in reversed(figures(before)) if g.kind == fig.kind]
    cited = [v for k in reversed(_CITE.findall(before)) if (v := _fact_as(facts.get(k) or {}, fig.kind)) is not None]
    return _sentence_values(fig.sentence, facts, fig.kind, fig) + nearest_first + cited


# A sentence that says it is doing arithmetic. Only there may a cited figure
# that does not match its key be a result derived from other figures.
_ARITHMETIC = re.compile(r"=|≈|×|÷|\s[−\-–+*/]\s|\]\s*[+−\-–/]\s*\[|\b(?:less|plus|minus|combined|total(?:s|led|ing)?|"
                         r"sum(?:s|med)?|per[\s-]share|of which|net of|times|divided by|multiplied by)\b", re.I)


def _multiples(context: str) -> list[float]:
    out = [g.value for g in figures(context) if g.kind == "x"]
    out += [float(a or b) for a, b in _BARE_MULTIPLIER.findall(context)]
    return out


# A bare multiplier beside a times sign ("4 × $96.22B", "16 × $384.88B"):
# annualising a quarter, or applying a multiple written without its "x".
_BARE_MULTIPLIER = re.compile(r"(?<![\w.$])(\d{1,3}(?:\.\d+)?)\s*[×*]|[×*]\s*(\d{1,3}(?:\.\d+)?)(?![\d.,]*\s*[%$BMKbmk])")


def _sentence_values(sentence: str, facts: Facts, kind: str, exclude: Figure | None = None) -> list[float]:
    """Values a figure in ``sentence`` can be derived from: the facts cited in
    it (in the figure's unit) and the other figures written beside it."""
    out = [_fact_as(facts.get(k), kind) for k in _CITE.findall(sentence) if facts.get(k)]
    out = [v for v in out if v is not None]
    out += [g.value for g in figures(sentence) if g.kind == kind and (exclude is None or g.raw != exclude.raw)]
    # Per-share results divide by a share count, and dollar volume multiplies
    # one ("61.0M shares at the $6.85 close"): share counts are not money figures.
    if kind == "usd":
        out += [float(f["value"]) for k in _CITE.findall(sentence) if (f := facts.get(k)) and f.get("unit") == "shares"]
        out += [float(n.replace(",", "")) * _SCALE[u.lower()]
                for n, u in re.findall(r"(\d[\d,]*(?:\.\d+)?)\s?(M|B|million|billion)\s+shares", sentence, re.I)]
    return out


def _own_values(fig: Figure, keys: list[str], facts: Facts) -> list[float]:
    out = []
    for k in keys:
        f = facts.get(k) or {}
        v = _fact_as(f, fig.kind)
        if v is None and fig.kind == "usd" and f.get("unit") == "shares" and isinstance(f.get("value"), (int, float)):
            v = float(f["value"])
        if v is not None:
            out.append(v)
    return out


def _bare_results(sentence: str) -> list[float]:
    """The value of each expression written out with bare numbers."""
    out = []
    for e in _BARE_EXPR.finditer(sentence):
        tokens = re.findall(r"\d[\d,]*(?:\.\d+)?|[+−\-–×÷*/]", e.group(0))
        nums = [float(t.replace(",", "")) for t in tokens[::2]]
        ops = [{"−": "-", "–": "-", "×": "*", "÷": "/"}.get(t, t) for t in tokens[1::2]]
        terms, signs = [nums[0]], [1]          # products and quotients first, then the sum
        for op, n in zip(ops, nums[1:], strict=False):
            if op in "*/":
                if op == "/" and not n:
                    break
                terms[-1] = terms[-1] * n if op == "*" else terms[-1] / n
            else:
                terms.append(n)
                signs.append(1 if op == "+" else -1)
        else:
            out.append(sum(sg * t for sg, t in zip(signs, terms, strict=True)))
    return out


def _bare_operands(sentence: str, fig: Figure) -> list[float]:
    """Numbers written without units in an expression ("$71.61B (136.16 −
    64.55)", "(2.85+1.04)"), read in the figure's scale."""
    if fig.kind != "usd":
        return []
    return [float(n.replace(",", "")) * fig.scale for e in _BARE_EXPR.finditer(sentence)
            for n in re.findall(r"\d[\d,]*(?:\.\d+)?", e.group(0))]


# "3-5 sessions" and "2025-2026" are ranges; a spaced or true minus is arithmetic.
_BARE_EXPR = re.compile(r"(?<![\w.$])\d[\d,]*(?:\.\d+)?(?:(?:\s?[+−×÷*/]\s?|\s[-–]\s)\d[\d,]*(?:\.\d+)?)+(?![\w.%])")


_NET_POSITION = re.compile(r"\bnet\s+(?:cash|debt)\b|\bderived\b", re.I)
_CHANGE = re.compile(r"\b(?:rose|rise[sn]?|fell|fall(?:s|en)?|increas\w*|decreas\w*|declin\w*|grew|grow\w*|drop\w*|"
                     r"jump\w*|chang\w*|swung|swing|added|adds?|lost|shrank|eased|climb\w*|up|down|gain\w*|"
                     r"widen\w*|narrow\w*)\b", re.I)


_UP = re.compile(r"\b(?:rose|rise[sn]?|increas\w*|grew|grow\w*|jump\w*|climb\w*|up|gain\w*|added|adds?|improv\w*)\b", re.I)
_DOWN = re.compile(r"\b(?:fell|fall(?:s|en)?|decreas\w*|declin\w*|drop\w*|down|lost|shrank|eased|worsen\w*|deteriorat\w*)\b",
                   re.I)
_LEVEL_AFTER_CHANGE = re.compile(r"\b(?:to|at|was|were|is|reached|stood|stands|hit|near|around)\s*(?:about|roughly|~|≈)?\s*$",
                                 re.I)


def _change_direction(fig: Figure, text: str) -> int | None:
    """+1 / −1 when a change word governs the figure (0 when the word has no
    direction: "changed", "swung"), None when none does or the figure is the
    level it changed to ("fell to $368.1M")."""
    window = f"{fig.lead} {text[fig.end: fig.end + 30]}"
    up, down = _UP.search(window), _DOWN.search(window)
    if not (up or down or _CHANGE.search(window)) or _LEVEL_AFTER_CHANGE.search(fig.lead):
        return None
    return 1 if up and not down else -1 if down and not up else 0


def _period_changes(key: str, facts: Facts, kind: str) -> list[float]:
    """The quarter-on-quarter changes (later minus earlier) of a cited line
    into and out of the cited quarter."""
    concept, _, period = key.partition(".")
    m = re.fullmatch(r"(\d{4})Q([1-4])", period)
    here = _fact_as(facts.get(key) or {}, kind)
    if not m or here is None:
        return []
    y, q = int(m.group(1)), int(m.group(2))
    prev = f"{y - 1}Q4" if q == 1 else f"{y}Q{q - 1}"
    nxt = f"{y + 1}Q1" if q == 4 else f"{y}Q{q + 1}"
    out = []
    if (v := _fact_as(facts.get(f"{concept}.{prev}") or {}, kind)) is not None:
        out.append(here - v)
    if (v := _fact_as(facts.get(f"{concept}.{nxt}") or {}, kind)) is not None:
        out.append(v - here)
    return out


def _matches_change(fig: Figure, keys: list[str], facts: Facts, direction: int = 0) -> bool:
    """A change in one cited line, or in the sum of up to three of them ("cash
    + short-term investments declined $89.4M [F:cash.2026Q2], [F:sti.2026Q2]"),
    in the direction the text says; or a difference of lines ("net cash fell
    $19B [F:debt.2026Q2] [F:cash_sti.2026Q2]"), whose direction the text
    cannot show without saying which line is subtracted."""
    from itertools import combinations, product

    changes = [c for c in (_period_changes(k, facts, fig.kind) for k in keys[:3]) if c]
    for n in range(1, len(changes) + 1):
        for lines in combinations(changes, n):
            for picks in product(*lines):
                for signs in product((1, -1), repeat=n):
                    if len(set(signs)) == 1 and signs[0] == -1:
                        continue                # a line or a sum keeps its own sign
                    total = sum(sg * d for sg, d in zip(signs, picks, strict=True))
                    if not _matches_value(fig, total):
                        continue
                    if len(set(signs)) > 1 or not direction or direction * total > 0:
                        return True
    return False


def _period_order(period: str):
    if m := re.fullmatch(r"(\d{4})Q([1-4])", period):
        return ("Q", int(m.group(1)), int(m.group(2)))
    if m := re.fullmatch(r"FY(\d{4})", period):
        return ("FY", int(m.group(1)), 0)
    return None


def _later_minus_earlier(keys: list[str], facts: Facts, kind: str) -> float | None:
    """For two keys of one line in two periods of one kind, later minus earlier."""
    if len(keys) != 2:
        return None
    (c1, _, p1), (c2, _, p2) = (k.partition(".") for k in keys)
    o1, o2 = _period_order(p1), _period_order(p2)
    if c1 != c2 or o1 is None or o2 is None or o1[0] != o2[0] or o1 == o2:
        return None
    early, late = (keys[0], keys[1]) if o1 < o2 else (keys[1], keys[0])
    a, b = _fact_as(facts.get(early) or {}, kind), _fact_as(facts.get(late) or {}, kind)
    return None if a is None or b is None else b - a


# Lines that extend a cited line in a net-cash or net-debt figure (the
# sheet's debt is filed by instrument); they are only ever subtracted.
_DEBT_EXTENSIONS = {"debt": ("debt_current", "commercial_paper", "notes_payable", "loans_payable",
                             "secured_debt", "credit_line")}


def _related_values(keys: list[str], facts: Facts, kind: str) -> list[float]:
    """Same-period debt lines a net figure leaves uncited ("net cash $27.32B
    [F:cash_sti.2026Q2], [F:debt.2026Q2]" also takes debt_current.2026Q2)."""
    out = []
    for k in keys:
        concept, _, period = k.partition(".")
        for extra in _DEBT_EXTENSIONS.get(concept, ()):
            other = f"{extra}.{period}"
            v = _fact_as(facts.get(other) or {}, kind)
            if v is not None and other not in keys:
                out.append(v)
    return out


def _cited_derivation(text: str, fig: Figure, known: list[str], facts: Facts) -> bool:
    """Whether a cited figure that matches none of its keys is arithmetic the
    text shows: on its own keys ("fell $1.43B [F:cash_sti.2026Q1][F:cash_sti.2026Q2]",
    "$741.5M ([F:a]+[F:b])", "3x ATR14 ($10.53) [F:atr14.usd]", "13.9M sh at
    [F:price.close]"), or, where its sentence says it is computing ("=",
    "less", "÷", "combined"...), on the figures of that sentence and the one
    before, never the value of a single key it cites (eval review 2026-10-09:
    "R&D of $51.9M [F:rnd.2026Q2]; S&M $20.9M" passed as R&D + S&M)."""
    own = _own_values(fig, known, facts)
    multiples = _multiples(fig.sentence)
    shares = [float(n.replace(",", "")) * _SCALE[u.lower()]
              for n, u in re.findall(r"(\d[\d,]*(?:\.\d+)?)\s?(M|B|million|billion)\s+(?:[\w-]+\s+)?sh(?:ares)?\b",
                                     fig.sentence, re.I)]
    if fig.kind == "usd":      # share counts cited in the sentence ("570.6M shares [F:shares.cover] at $7.42")
        shares += [float(f["value"]) for k in _CITE.findall(fig.sentence)
                   if (f := facts.get(k)) and f.get("unit") == "shares" and isinstance(f.get("value"), (int, float))]
    # "Operating cash flow improved by $34.8M [F:ocf.2026Q1][F:ocf.2026Q2]" when it worsened.
    direction = _change_direction(fig, text)
    delta = _later_minus_earlier(known, facts, fig.kind)
    if direction and delta and _matches_value(fig, delta) and direction * delta < 0:
        return False
    if (own and _derivable(fig, own + shares, multiples, max_terms=max(2, len(own)))
            and (len(own) >= 2 or shares or any(_matches_value(fig, o * m) for o in own for m in multiples if m))):
        return True
    # The mean of its keys ("$449.30 [F:boll.upper] [F:boll.lower]", "midpoint of ...").
    if len(own) >= 2 and _matches_value(fig, sum(own) / len(own)):
        return True
    # A change in the cited line ("Cash fell by $11.17B in 2026Q2 [F:cash.2026Q2]",
    # "Equity rose $28.02B over the quarter [F:equity.2026Q1]").
    # The change word governs the figure: "fell by $11.17B", "a $36.13B jump";
    # "fell to $368.1M" states the level, not the change.
    if (fig.kind in ("usd", "pct") and direction is not None
            and _matches_change(fig, known, facts, direction)):
        return True
    # Net of a related line the text leaves uncited ("Net cash (derived) $27.32B
    # [F:cash_sti.2026Q2], [F:debt.2026Q2]" also takes debt_current.2026Q2).
    if len(own) >= 2 and _NET_POSITION.search(fig.sentence):
        from itertools import product

        for x in _related_values(known, facts, fig.kind):
            for signs in product((1, -1), repeat=len(own) - 1):
                if _matches_value(fig, own[0] + sum(sg * o for sg, o in zip(signs, own[1:], strict=True)) - x):
                    return True
    if not (_ARITHMETIC.search(fig.sentence) or _BARE_EXPR.search(fig.sentence)):
        return False
    # The expression written out in full ("−$171.8M (11.1 + 14.3 + 52.6 + 93.8)").
    if fig.kind == "usd" and any(_matches_value(fig, v * fig.scale) for v in _bare_results(fig.sentence)):
        return True
    prev = _previous_sentence(text, fig)
    operands = _context_values(text, fig, facts)
    if len(own) == 1:
        operands = [v for v in operands if not _close(abs(v), abs(own[0]), fig.kind)]
    # Numbers the text writes out in an expression stand ("$34.37B (33.37 + 1.00
    # [F:debt_current.2026Q2])"), its own key's value included.
    operands = _bare_operands(fig.sentence, fig) + operands
    multiples = _multiples(f"{prev} {fig.sentence}")
    return _derivable(fig, operands, multiples) or any(
        _matches_value(fig, v / m) for v in operands for m in multiples if m)


# A figure the text itself marks as not from the filings (R7: the RM lists
# news figures as unverified) is a disclosure, not a claim. Only disclosure
# phrases: "Cash dropped to $212M" and "reported by the company" are claims.
_LABELLED_UNVERIFIED = re.compile(
    r"\b(unverified|not on the fact[- ]sheet|not (?:a )?fact[- ]sheet (?:keys?|figures?|items?)|"
    r"not (?:on the|a) fact[- ]sheet|headline[- ]only|news[- ](?:reported|only)|not verified|cannot be verified|"
    r"per (?:the )?news|as reported by|(?:reported|cited) (?:in|by) (?:the )?(?:news|press|media)|"
    r"(?:dropped|excluded|removed) from (?:the|this|our) (?:thesis|analysis|case|sizing|valuation|call)|"
    r"disregard(?:ed)? (?:it|this|that|them|the (?:figure|claim|report)))\b", re.I)


# Only where the text says it is adding periods up.
_SUMS_PERIODS = re.compile(r"\b(H[12]|first half|second half|half[- ]year|combined|together|cumulative|"
                           r"year[- ]to[- ]date|YTD|9M|nine months|TTM|trailing|over (?:the )?(?:last )?"
                           r"(?:two|three|four) quarters|across)\b", re.I)


def _period_sum(fig: Figure, facts: Facts) -> bool:
    """A sum of two to four consecutive quarters of one line item ("$741.5M
    in H1" = stock issued for acquisitions in 2026Q1 + 2026Q2)."""
    by_concept: dict[str, list[tuple[str, float]]] = {}
    for f in facts.numeric():
        key = f["key"]
        concept, _, period = key.partition(".")
        if re.fullmatch(r"\d{4}Q[1-4]", period) and f.get("unit") == "usd":
            by_concept.setdefault(concept, []).append((period, float(f["value"])))
    for series in by_concept.values():
        vals = [v for _, v in sorted(series)]
        for n in (2, 3, 4):
            for i in range(len(vals) - n + 1):
                if _matches_value(fig, sum(vals[i:i + n])):
                    return True
    return False


# ---- checks ----------------------------------------------------------------------------------


# "($1.02B / $145.0M − 1) × 100 = 603.4%": an explicit growth formula whose
# result is wrong (eval, 2026-10-10: MRNA's was 600.7%).
_SCALE_WORD = {"": 1.0, "k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}
_FORMULA = re.compile(
    r"\(\s*\$?\s?(\d[\d,]*(?:\.\d+)?)\s?([KMBT])?\s*(?:\[[^\]]*\]\s*)?/\s*\$?\s?(\d[\d,]*(?:\.\d+)?)\s?([KMBT])?"
    r"\s*(?:\[[^\]]*\]\s*)?[−-]\s*1\s*\)\s*(?:[×x*]\s*100\s*)?=\s*([−-]?\d[\d,]*(?:\.\d+)?)\s?%", re.I)


def check_formulas(text: str, stage: str, field_name: str) -> list[LintFlag]:
    """Check 2 (part): an explicit (A / B − 1) × 100 = p% reproduces p."""
    flags = []
    for m in _FORMULA.finditer(text or ""):
        def value_and_half_unit(num: str, scale: str):
            digits = num.replace(",", "")
            decimals = len(digits.split(".")[1]) if "." in digits else 0
            mult = _SCALE_WORD[(scale or "").lower()]
            return float(digits) * mult, 0.5 * 10 ** -decimals * mult

        a, da = value_and_half_unit(m.group(1), m.group(2))
        b, db = value_and_half_unit(m.group(3), m.group(4))
        raw = m.group(5).replace("−", "-").replace(",", "")
        stated = float(raw)
        stated_half = 0.5 * 10 ** -(len(raw.split(".")[1]) if "." in raw else 0)
        if b - db <= 0:
            continue
        actual = (a / b - 1) * 100
        # The inputs are rounded as written: the true result can be anywhere
        # their last digits allow ("$1.02B / $145.0M" spans 599.5–607.4%).
        low = ((a - da) / (b + db) - 1) * 100
        high = ((a + da) / (b - db) - 1) * 100
        if not low - stated_half - 0.05 <= stated <= high + stated_half + 0.05:
            flags.append(LintFlag(_severity(field_name, "arithmetic"), "arithmetic", stage, field_name,
                                  _sentence_at(text, m.start(), m.end())[:300],
                                  f"the formula gives {actual:.1f}%, not {stated:g}%"))
    return flags


def check_numbers(text: str, facts: Facts, stage: str, field_name: str) -> list[LintFlag]:
    """Check 1. Cited figures must match their key; a cited key must exist."""
    flags: list[LintFlag] = []
    all_figs = figures(text)
    sentence_figs: dict[int, list[Figure]] = {}
    for g in all_figs:
        sentence_figs.setdefault(g.sent_start, []).append(g)
    pre_cited = _pre_citations(text, all_figs)
    for fig in all_figs:
        if not fig.keys:
            continue
        known = [k for k in fig.keys if facts.get(k)]
        pre_keys = [k for k in pre_cited.get(fig.start, []) if facts.get(k)]
        # Siblings: the other figures of its sentence, the nearest few.
        siblings = sorted((g for g in sentence_figs.get(fig.sent_start, []) if g is not fig),
                          key=lambda g: abs(g.start - fig.start))[:_MAX_SIBLINGS]
        lines = [k for k in fig.keys if not facts.get(k) and "." not in k and facts.periods_of(k)]
        for k in fig.keys:
            if not facts.get(k) and k not in lines:
                flags.append(LintFlag(_severity(field_name, "unknown_key"), "unknown_key", stage, field_name,
                                      fig.sentence[:300], f"no fact {k} on the sheet", [k]))
        if lines:
            flags.extend(_check_line_citation(fig, lines, siblings, facts, stage, field_name))
        # "[F:boll.upper] $238.43 and [F:boll.lower] $209.95": the citation
        # comes first, and the one after the figure is the next figure's.
        if pre_keys and (any(_matches_fact(fig, facts.get(k)) for k in pre_keys)
                         or (len(pre_keys) >= 2 and _derivable(fig, _own_values(fig, pre_keys, facts),
                                                                max_terms=len(pre_keys)))):
            continue
        if known and fig.value and not any(_matches_fact(fig, facts.get(k)) for k in known):
            # Keys of another unit (a % cited after a $ figure) don't contradict it.
            comparable = [k for k in known if _fact_as(facts.get(k), fig.kind) is not None]
            # Arithmetic on the cited keys, or shown in the sentence.
            if _cited_derivation(text, fig, known, facts):
                continue
            # A citation placed at the end of a sentence may belong to another
            # figure in it ("$83.8M in Q2, with $221M in orders [F:revenue.2026Q2]"),
            # but only to one that carries no citation of its own: "fell to $13.5M
            # [F:rnd.2026Q2] from $31.0M [F:rnd.2026Q1]" swaps the quarters.
            comparable = [k for k in comparable
                          if not any(_matches_fact(g, facts.get(k)) and (not g.keys or k in pre_cited.get(g.start, []))
                                     for g in siblings)]
            # A citation for the total of this figure and an uncited sibling ("cash
            # was $657.9M and short-term investments $726.6M [F:cash_sti.2026Q2]").
            comparable = [k for k in comparable
                          if not any(not g.keys and g.kind == fig.kind
                                     and any(_close(fig.value + sg * g.value, _fact_as(facts.get(k), fig.kind), fig.kind)
                                             for sg in (1, -1))
                                     for g in siblings)]
            # Keys that together derive a sibling with no citation of its own
            # ("~$1.75B across 2025Q3–2026Q1 … [F:a.2025Q3][F:a.2025Q4][F:a.2026Q1]").
            own = _own_values(fig, comparable, facts)
            if len(own) >= 2 and any(not g.keys and g.kind == fig.kind and _derivable(g, own, max_terms=len(own))
                                     for g in siblings):
                comparable = []
            # An ATR key cited for a distance in ATRs ("0.17 ATR below the $7.19
            # close, per [F:atr14.usd]") measures the distance, not this figure.
            if re.search(r"(?:\d(?:\.\d+)?|\b(?:one|two|three|half|an?))[\s-]*(?:x\s*|times\s+)?ATR(?:s|14)?\b",
                         fig.sentence, re.I):
                comparable = [k for k in comparable if not k.startswith("atr14")]
            # A key cited for a bare number in the sentence ("(7.19 − 7.12) / 0.4076
            # [F:atr14.usd]") belongs to that number, not to this figure.
            bare = [float(n.replace(",", "")) for n in re.findall(r"(?<![\w.$])\d[\d,]*\.\d+|(?<![\w.$])\d{2,}(?![\d.])",
                                                                  _without_figures(fig.sentence))]
            comparable = [k for k in comparable
                          if not any(abs(b - float(facts.value(k))) <= max(abs(float(facts.value(k))) * 0.005, 0.0005)
                                     for b in bare if isinstance(facts.value(k), (int, float)))]
            if comparable:
                expected = f"{fig.raw.lstrip('(')} is written; " + "; ".join(f"{k} = {facts.value(k)}" for k in comparable)
                flags.append(LintFlag(_severity(field_name, "cited_mismatch"), "cited_mismatch", stage,
                                      field_name, fig.sentence[:300], expected, comparable))
    return flags


def _without_figures(sentence: str) -> str:
    """The sentence with its figures blanked: a bare number is one written
    without a unit, not the digits of "−979.3%" itself."""
    chars = list(sentence)
    for g in figures(sentence):
        chars[g.start: g.end] = " " * (g.end - g.start)
    return "".join(chars)


def _pre_citations(text: str, figs: list[Figure]) -> dict[int, list[str]]:
    """{figure start: keys cited just before it}, only in a consistent
    cite-first style ("[F:boll.upper] $238.43 and [F:boll.lower] $209.95").
    A citation run that directly follows a figure citing after itself is that
    figure's ("$31.0M [F:rnd.2026Q2], $31.0M [F:rnd.2026Q1]")."""
    out: dict[int, list[str]] = {}
    previous_pre = False
    for i, g in enumerate(figs):
        window_start = max(0, g.start - 200)
        pre = _CITES_BEFORE.search(text[window_start: g.start])
        keys: list[str] = []
        if pre:
            run_start = window_start + pre.start(1)
            prev = figs[i - 1] if i else None
            follows = (prev is not None and prev.end <= run_start
                       and re.fullmatch(r"[\s,;:)|*]*", text[prev.end: run_start]) is not None)
            # "[F:cash_sti.2026Q2] ($76.84B)": a value glossed right after its citation.
            gloss = g.raw.startswith("(") or text[max(0, g.start - 1): g.start] == "("
            if not follows or previous_pre or gloss:
                keys = _CITE.findall(pre.group(1))
        out[g.start] = keys
        previous_pre = bool(keys)
    return out


def _check_line_citation(fig: Figure, lines: list[str], siblings: list[Figure], facts: Facts,
                         stage: str, field_name: str) -> list[LintFlag]:
    """A citation of a line without its period ("[F:revenue_yoy]"): resolved
    against the line's periods. A figure one of them matches, or that the
    lines derive in one period, gets a minor note to add the period; one none
    of them supports is a mismatch."""
    from itertools import product

    periods = {k: facts.periods_of(k) for k in lines}
    candidates = [f for k in lines for f in periods[k] if _fact_as(f, fig.kind) is not None]
    note = LintFlag("minor", "citation_format", stage, field_name, fig.sentence[:300],
                    f"cite a period: [F:{lines[0]}.<period>], not [F:{lines[0]}]", list(lines))
    if not candidates:
        return [note]                           # a line of another unit: not this figure's
    if any(_matches_fact(fig, f) for f in candidates):
        return [note]
    # The line belongs to an uncited figure beside it.
    if any(not g.keys and any(_matches_fact(g, f) for f in candidates) for g in siblings):
        return [note]
    # The mean of two members of a line that has no periods ("$170.57 (Boll
    # Mid) [F:boll]" is the middle of boll.upper and boll.lower).
    values = [_fact_as(f, fig.kind) for f in candidates if _period_order(f["key"].partition(".")[2]) is None]
    if any(_matches_value(fig, (a + b) / 2) for i, a in enumerate(values) for b in values[i + 1:]):
        return [note]
    # A net figure across the lines in one period ("net debt ~$2.87B [F:cash_sti, F:debt, F:commercial_paper]").
    shared = set.intersection(*({f["key"].partition(".")[2] for f in periods[k]} for k in lines))
    for period in shared:
        vals = [v for k in lines if (v := _fact_as(facts.get(f"{k}.{period}") or {}, fig.kind)) is not None]
        if len(vals) >= 2 and any(_matches_value(fig, vals[0] + sum(sg * v for sg, v in zip(signs, vals[1:], strict=True)))
                                  for signs in product((1, -1), repeat=len(vals) - 1)):
            return [note]
    return [LintFlag(_severity(field_name, "cited_mismatch"), "cited_mismatch", stage, field_name, fig.sentence[:300],
                     f"{fig.raw.lstrip('(')} is written; no period of {', '.join(lines)} is that", list(lines))]


def unsupported_figures(text: str, facts: Facts, stage: str, field_name: str) -> list[LintFlag]:
    """Check 1, second half: uncited money figures that match no fact. Not
    blocking (news and tools carry real figures the sheet doesn't), but
    load-bearing in load-bearing places."""
    flags: list[LintFlag] = []
    numeric = list(facts.numeric(actual_only=True))
    # Money figures already established earlier in this text (cited, on the
    # sheet or derived): a figure derived once and reused below ("annualised
    # revenue $384.88B" in the base, bear and bull cases) is not re-derived.
    established: list[float] = []
    for fig in figures(text):
        if fig.kind != "usd":
            continue
        if fig.keys:
            established.append(fig.value)
            continue
        if abs(fig.value) < 1e6 or fig.in_range:
            continue
        if fig.at_least and not fig.raw.endswith("+"):
            fig = replace(fig, at_least=False)  # "over $500M" is a claim, not every larger figure
        if (any(_matches_fact_uncited(fig, f) for f in numeric)
                or any(_matches_value(fig, v) for v in established)
                or _derivable(fig, _wide_context_values(text, fig, facts),
                              _multiples(text[max(0, fig.start - 300): fig.end + 60]), max_terms=4)
                or (_SUMS_PERIODS.search(fig.sentence) and _period_sum(fig, facts))):
            established.append(fig.value)
            continue
        if _LABELLED_UNVERIFIED.search(fig.sentence):
            continue
        flags.append(LintFlag(_severity(field_name, "unsupported"), "unsupported", stage, field_name,
                              fig.sentence[:300], f"{fig.raw.lstrip('(')} is not on the fact sheet and not cited"))
    return flags


_FROM_TO = re.compile(
    r"from\s+(?P<a>[^,;]{0,40}?\$?[−\-–]?\$?\d[\d,.]*\s?(?:%|x|[kmbt]|million|billion)?)"
    r"(?P<akeys>(?:\s*\[F:[^\]]+\])*)"
    r"(?:(?!\.\s)[^;—]){0,60}?\bto\s+(?P<b>\$?[−\-–]?\$?\d[\d,.]*\s?(?:%|x|[kmbt]|million|billion)?)"
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
_HEDGES = re.compile(r"\b(wrong|incorrect|false|misread|not supported|refuted?|only after|re-?add|re-?enter|"
                     r"reassess|confirmed|once|needs? to|must|"
                     r"if|would|could|should|unless|risk of|watch for|a break|breaks?|fall|falls|falling|"
                     r"drop|drops|reclaim|reclaims|until|when|previous|prior|earlier|before|not|nor|neither|"
                     r"approaching|threshold|potential|stop|stop-loss|target|entry|level|next|cancels?|"
                     r"invalidates?|triggers?)\b|\ba\s+(?:daily\s+|weekly\s+|sustained\s+|confirmed\s+)?close\b|[<>]", re.I)
# Whose position "above/below the 50-day" describes: the price, not a stop,
# a target or another average ("the 50-day SMA remains above the 200-day").
_PRICE_SUBJECT = re.compile(r"(?:\b(?:price|stock|shares|it|trades?|trading|closed?|closes|closing)\b|"
                            r"\b(?!SMA\b|EMA\b|DMA\b|MACD\b|RSI\b|ATR\b)[A-Z]{2,5}\b)[^.$\d]{0,30}$")
_OTHER_AVERAGE = re.compile(r"\b(?:\d{1,3}[-\s]?(?:day|d|DMA|week|wk)\b|SMA|EMA|DMA|moving\s+average|MA\b)", re.I)


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
        lead = text[max(0, m.start() - 40): m.start()]
        lead = lead[lead.rfind(". ") + 1:] if ". " in lead else lead
        subject = _PRICE_SUBJECT.search(lead) or _PRICE_SUBJECT.search(lead.lower())
        if _HEDGES.search(sentence) or not subject or _OTHER_AVERAGE.search(lead[subject.start():]):
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
                r"\d[\d.]*\s?%\s+(?:of\s+(?:the\s+)?float\s+)?(?:short(?:ed)?(?!\s+of\b)|sold\s+short)\b|"
                r"\bshorted\b", re.I), "short interest"),
    (re.compile(r"\bconsensus\s+(?:EPS|estimate|estimates|revenue|forecast)\b", re.I), "consensus estimates"),
    (re.compile(r"\b(?:analyst|street|consensus|average)\s+(?:price\s+)?targets?\b(?:\s+(?:of|at)\s+\$)?", re.I),
     "analyst price targets"),
]
# Saying the data is NOT used is fine ("short interest is not available").
_POLICY_NEGATED = re.compile(r"\b(not\s+(?:available|used|provided|cited|considered)|unavailable|"
                             r"no\s+(?:data|figures?)\s+on|excluded|missing|do\s+not\s+use|don't\s+use|"
                             r"never\s+use|not\s+to\s+use|without\s+using|may\s+not\s+be\s+used|avoid)\b", re.I)


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


# The review's own vocabulary in reader text (MSFT, 2026-10-09: the RM's
# "Verified (derived)" labels became a digest headline, "quality verified",
# which a reader takes for the platform's Verified badge). The stages'
# adjudication labels are for the record; the digest states the substance.
# Bare "review" is ordinary prose and is not flagged.
_PROCESS = re.compile(r"\bunverified\b|\berrat(?:a|um)\b|\blint\b|\bfact[\s-]sheets?\b|"
                      r"\bprevious\s+draft\b", re.I)
# "Verified" as the review's label ("**Verified**", "quality verified",
# "verified quality", "(verified)", "Verified (derived)"), not "FDA-verified"
# or "independently verified".
_VERIFIED_LABEL = re.compile(r"\((?:verified|unverified)\)|"
                             r"\b(?:quality|figures?|numbers?|data|facts?|claims?|sheet)[\s-]verified\b|"
                             r"(?<![-\w])verified,?\s+(?:[\w-]+\s+)?(?:evidence|points?|arithmetic|math|figures?|numbers?|"
                             r"data|facts?|claims?|filings?|fundamentals|growth|earnings|quality|cash|FCF|revenue|"
                             r"losses|record|results?|metrics?)\b|\b(?:real|correct|confirmed) and verified\b|"
                             r"(?<![-\w])verified\s*(?:\(|:)", re.I)
_VERIFIED_LABEL_CASED = re.compile(r"(?<![-\w])Verified\b")  # the capitalised label
# An erratum id ("per errata E5", "see E2", "(E5)", "fixed in E12"), not
# "E2 jets" or "Series E3"; case-sensitive.
_ERRATUM_ID = re.compile(r"(?:\b[Ee]rrat(?:a|um)\s+|\b(?:per|see)\s+|\b(?:fixed|corrected|addressed|resolved|closed)\s+"
                         r"(?:in|by|per)\s+|\()E\d{1,2}\b")


def check_process_language(text: str, stage: str, field_name: str) -> list[LintFlag]:
    """Review and process vocabulary in a digest field. Always load-bearing:
    the digest is what the reader sees. Other stages legitimately label their
    claims ("Verified (derived)") and are not checked."""
    if not field_name.startswith("digest") or not text:
        return []
    flags: list[LintFlag] = []
    seen: set[str] = set()
    for rx in (_PROCESS, _VERIFIED_LABEL, _VERIFIED_LABEL_CASED, _ERRATUM_ID):
        for m in rx.finditer(text):
            sentence = _sentence_at(text, m.start(), m.end())
            if sentence in seen:
                continue
            seen.add(sentence)
            flags.append(LintFlag("load_bearing", "process_language", stage, field_name, sentence[:300],
                                  f"review vocabulary ('{m.group(0)}') in reader text: state the substance, "
                                  "not the review's labels"))
    return flags


# Check 8. AMD, 2026-10-10 review: risk lenses quoted agents in the first
# person ("I'd reconsider the SELL/trim stance"), talked about themselves
# ("This lens adds..."), and the plan spoke to "the investor" about "after
# the weekend". The patterns are narrow on purpose: an open load-bearing flag
# costs a revision, so "Phase I", "the I/O die", "investor day", "on Tuesday,
# November 4" and quoted guidance ("we expect") must not fire.
#
# Load-bearing only in the editorial fields the editor can patch; in the
# excerpts (which carry news and quotes) and the risk lenses, a lead.
EDITORIAL_FIELDS = {"digest.headline", "digest.bull_thesis", "digest.bear_thesis", "digest.ruling",
                    "digest.bull_points", "digest.bear_points", "digest.call_thesis", "digest.deciding_variable",
                    "digest.catalyst",
                    "digest.conviction_note", "digest.sizing", "digest.entry_style", "digest.review_cycle",
                    "digest.exit_triggers"}
_QUOTED = re.compile(r"\"[^\"\n]{0,400}\"|“[^”\n]{0,400}”")
_FIRST_PERSON = re.compile(r"(?<![\w'’/-])(?:I(?=\s+[a-z])|I(?:'|’)(?:d|m|ve|ll)\b|[Mm]y(?=\s+[a-z])|"
                           r"[Ww]e(?:'|’)(?:d|re|ve|ll)\b|[Ww]e(?=\s+(?:would|will|think|believe|expect|see|"
                           r"prefer|recommend|favor|favour|view|rate|remain|maintain|reconsider)\b)|"
                           r"[Oo]ur(?=\s+(?:view|call|rating|target|thesis|stance|recommendation|position)\b))")
_SECOND_PERSON = re.compile(r"(?<![\w'’])(?:you(?=\s+(?:should|could|would|can|may|might|will|want|need|"
                            r"hold|own|buy|sell|trim|add)\b)|your(?=\s+(?:position|portfolio|shares|stake|"
                            r"exposure|target|risk|holdings?|loss|account)\b))|"
                            r"\bthe investor(?:'s|’s)?\b(?!\s+(?:day|relations|presentation|conference|call|deck)\b)"
                            r"(?!,\s+[A-Z])", re.I)
_WEEKDAY = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday)"
# "on Monday" is relative; "on Tuesday, November 4" and "on Monday 6 October" are dates.
_RELATIVE_TIME = re.compile(r"\b(?:after|over|through|this|next)\s+(?:the\s+)?weekend\b|\btomorrow\b|\btonight\b|"
                            rf"\b(?:this|next)\s+{_WEEKDAY}\b|\bon\s+{_WEEKDAY}\b(?!,?\s+(?:[A-Z][a-z]{{2}}|\d))", re.I)
# A first-person verb after "I" makes it the pronoun even after a capitalised
# word ("Overall I think", "Here I see").
_I_VERB = re.compile(r"\s+(?:think|believe|would|will|see|expect|prefer|recommend|am|have|remain|maintain|"
                     r"reconsider|favor|favour|view|rate|suggest|doubt|agree)\b")
_LENS_META = re.compile(r"\bthis lens\b|\blens (?:adds|makes|requires|favors|favours)\b", re.I)


def _unquoted(text: str) -> str:
    """The text with quoted spans blanked (same length, so offsets hold)."""
    return _QUOTED.sub(lambda m: " " * len(m.group(0)), text or "")


def check_voice(text: str, stage: str, field_name: str) -> list[LintFlag]:
    """Check 8: reader text is a published note in the third person."""
    if not text or not (field_name.startswith("digest") or field_name.startswith("pm")):
        return []
    severity = "load_bearing" if field_name in EDITORIAL_FIELDS else "minor"
    bare = _unquoted(text)
    flags: list[LintFlag] = []
    seen: set[str] = set()
    for rx, what in ((_FIRST_PERSON, "first person"), (_SECOND_PERSON, "addressed to a reader or 'the investor'"),
                     (_RELATIVE_TIME, "time relative to today"), (_LENS_META, "talk about the lens itself")):
        for m in rx.finditer(bare):
            sentence = _sentence_at(text, m.start(), m.end())
            # "Phase I trial", "Fund I raised", "Charles I": a numeral after a name.
            if sentence in seen or (m.group(0) == "I" and re.search(r"[A-Z][\w-]*\s*$", bare[:m.start()])
                                    and not _I_VERB.match(bare, m.end())):
                continue
            seen.add(sentence)
            flags.append(LintFlag(severity, "voice", stage, field_name, sentence[:300],
                                  f"{what} ('{m.group(0)}'): write in the third person about the stock, "
                                  "with time in market terms"))
    return flags


_TRADER_ACTION = re.compile(r"(?<![\w$-])(?:BUY|SELL|HOLD)\b(?!-(?:side|rated)\b)")
_TIER_WORD = re.compile(r"\b(?:Overweight|Underweight|OVERWEIGHT|UNDERWEIGHT)\b(?!-rated\b)")
# A sentence reporting someone else's rating action ("Morgan Stanley cut it to
# Underweight") is news, not the report's rating.
_RATING_NEWS = re.compile(r"\b(?:upgrad\w*|downgrad\w*)\b|\banalysts?\s+at\b|\bconsensus\s+rating\b|"
                          r"\b(?:cut|raised|moved|lowered|initiat\w*|reiterat\w*)\b[^.;]{0,40}?\b(?:to|at|with)\s+(?:an?\s+)?"
                          r"(?:Buy|Overweight|Hold|Underweight|Sell|Neutral|Outperform|Underperform|Equal[- ]weight)\b",
                          re.I)


def check_rating_words(text: str, rating: str | None, stage: str, field_name: str) -> list[LintFlag]:
    """Check 8: the only rating word in the editorial digest is the final
    rating; the trader's BUY/HOLD/SELL never reaches the reader. The
    excerpts (news, analysts) are not checked; the lenses only as a lead."""
    if not text or field_name in {"digest.market_excerpt", "digest.sentiment_excerpt", "digest.news_excerpt",
                                  "digest.fundamentals_excerpt"}:
        return []
    severity = "load_bearing" if field_name in EDITORIAL_FIELDS or field_name == "digest.trader_excerpt" else "minor"
    bare = _unquoted(text)
    flags: list[LintFlag] = []
    for rx in (_TRADER_ACTION, _TIER_WORD):
        for m in rx.finditer(bare):
            word = m.group(0)
            if rating and word.lower() == rating.lower():
                continue
            sentence = _sentence_at(text, m.start(), m.end())
            if _RATING_NEWS.search(sentence):
                continue
            expected = (f"'{word}' is the trader's action, not the report's rating" if rx is _TRADER_ACTION
                        else f"'{word}' names a tier the report did not assign")
            flags.append(LintFlag(severity, "rating_word", stage, field_name, sentence[:300],
                                  expected + (f" ({rating})" if rating else "")))
    return flags


# The margin word must describe the win, not a moat, a margin or a factor.
# Debate verbs only ("beat estimates slightly" is an earnings beat), within
# one clause.
_WIN = r"(?:won|wins|prevail\w*|carried|carries|edged|edges)"
_GAP = r"[^\w.;:]+"
_NARROW = re.compile(rf"\b{_WIN}{_GAP}(?:\w+{_GAP}){{0,3}}?(?:narrow(?:ly)?|slight(?:ly)?|marginal(?:ly)?|barely|by a hair)\b|"
                     rf"\b(?:narrow(?:ly)?|slight(?:ly)?|marginal(?:ly)?|barely){_GAP}(?:\w+{_GAP}){{0,2}}?{_WIN}\b", re.I)
_DECISIVE = re.compile(rf"\b{_WIN}{_GAP}(?:\w+{_GAP}){{0,3}}?(?:decisive(?:ly)?|overwhelming(?:ly)?|resounding(?:ly)?|"
                       rf"comfortabl[ye])\b|\b(?:decisive(?:ly)?|overwhelming(?:ly)?|resounding(?:ly)?|one-sided){_GAP}"
                       rf"(?:\w+{_GAP}){{0,2}}?(?:{_WIN}|victory|debate)\b", re.I)


def check_margin_wording(digest: dict) -> list[LintFlag]:
    """Check 8: the ruling's words about the win agree with the debate
    margin's band (ReportDigest.conviction: narrow 20-45, decisive 75+).
    AMD, 2026-10-10: "won narrowly" beside a margin of 62."""
    margin = digest.get("conviction")
    if not isinstance(margin, (int, float)) or isinstance(margin, bool):
        return []
    flags: list[LintFlag] = []
    for key in ("headline", "ruling", "conviction_note"):
        text = digest.get(key)
        if not isinstance(text, str):
            continue
        for rx, ok, band in ((_NARROW, margin < 45, "under 45"), (_DECISIVE, margin >= 75, "75 or more")):
            m = rx.search(text)
            if m and not ok:
                flags.append(LintFlag("load_bearing", "margin_wording", "digest", f"digest.{key}",
                                      _sentence_at(text, m.start(), m.end())[:300],
                                      f"'{m.group(0)}' needs a debate margin {band}; the margin is {margin:g}"))
    return flags


_FUNDAMENTAL = re.compile(r"revenue|sales|growth|margin|cash flow|\bFCF\b|guidance|\bEPS\b|earnings per|backlog|"
                          r"bookings|orders|net income|operating income|operating loss|net loss|debt|dilution|"
                          r"share count|burn|runway", re.I)


# A measured value: a money amount, a percentage or a multiple, not a year
# or a quarter label ("FY2026 guide $12B", "Q2: 34.0%").
_MEASURE = re.compile(r"([-−]?)\$\s?(\d[\d,]*(?:\.\d+)?)|([-−]?\d[\d,]*(?:\.\d+)?)\s?(%|x\b|×)")


def _first_number(text) -> float | None:
    m = _MEASURE.search(str(text or ""))
    if not m:
        return None
    sign, raw = (m.group(1), m.group(2)) if m.group(2) else ("", m.group(3))
    return float((sign + raw).replace("−", "-").replace(",", ""))


def check_trigger_headroom(digest: dict) -> list[LintFlag]:
    """Check 8: a trigger's threshold isn't today's value (GOOGL review: "trim if
    operating margin falls below 34.0%", the quarter's margin to the decimal).
    A lead for the editor."""
    flags = []
    for t in digest.get("exit_triggers") or []:
        if not isinstance(t, dict):
            continue
        now, at = _first_number(t.get("current")), _first_number(t.get("threshold"))
        if now is not None and at is not None and abs(now - at) <= max(abs(now) * 0.01, 1e-9):
            flags.append(LintFlag("minor", "trigger_headroom", "digest", "digest.exit_triggers",
                                  f"{t.get('metric') or t.get('title')}: current {t.get('current')}, threshold "
                                  f"{t.get('threshold')}"[:300],
                                  "a threshold at today's value fires on any move; give it headroom or call it a tripwire"))
    return flags


def check_target_debt(decision: dict | None, facts: Facts) -> list[LintFlag]:
    """Check 6 (part): the target math subtracts the debt EV uses (long-term plus
    current). GOOGL review: EV used $100.17B, the target $98.17B."""
    if not decision or not facts:
        return []
    math_text = decision.get("target_math") or ""
    sheet = facts.sheet or {}
    quarters = sheet.get("quarters") or []
    if not math_text or not quarters:
        return []
    q = quarters[-1]["calendar"]
    lt, cur = facts.value(f"debt.{q}"), facts.value(f"debt_current.{q}")
    if not isinstance(lt, (int, float)) or not isinstance(cur, (int, float)) or cur < 0.01 * (lt + cur):
        return []
    total = lt + cur
    near_debt = [f.value for f in figures(math_text)
                 if f.kind == "usd" and re.search(r"debt", math_text[f.end:f.end + 40], re.I)]
    if near_debt and not any(abs(x - total) <= 0.01 * total for x in near_debt) \
            and any(abs(x - lt) <= 0.005 * lt for x in near_debt):
        return [LintFlag("minor", "target_debt", "portfolio_manager", "pm.price_target", math_text[:300],
                         f"the target subtracts long-term debt only ({lt / 1e9:,.2f}B); EV uses long-term plus "
                         f"current debt ({total / 1e9:,.2f}B)", [f"debt.{q}", f"debt_current.{q}"])]
    return []


_ONE_OFF = re.compile(r"(\d{4}Q[1-4]) gross margin .*?likely a one-off")
_COMPARES = re.compile(r"\b(?:from|versus|vs\.?|compared|a year (?:earlier|ago)|year[- ]on[- ]year|turn(?:ed|around))\b",
                       re.I)
_CAVEAT = re.compile(r"one[- ]off|one[- ]time|charge|write[- ]?down|impairment|flatter|weak base", re.I)


def check_one_off_bases(state: dict) -> list[LintFlag]:
    """Check 8: a quarter the fact sheet flags as a likely one-off is not used as
    a comparison base without saying so (AMD review: the thesis cited Data
    Center's 2025Q2 loss as turnaround evidence beside the flag). A lead."""
    sheet = state.get("fact_sheet") or {}
    quarters = {m.group(1) for f in sheet.get("flags") or [] if (m := _ONE_OFF.search(f))}
    if not quarters:
        return []
    flags = []
    for stage, field_name, text in stage_texts(state):
        if field_name not in LOAD_BEARING_FIELDS and not field_name.startswith("digest"):
            continue
        for q in quarters:
            for m in re.finditer(re.escape(q), text):
                sentence = _sentence_at(text, m.start(), m.end())
                if _COMPARES.search(sentence) and not _CAVEAT.search(sentence):
                    flags.append(LintFlag("minor", "one_off_base", stage, field_name, sentence[:300],
                                          f"{q} is flagged as a likely one-off quarter: say so when using it as a base"))
                    break
    return flags


def check_trigger_mix(digest: dict) -> list[LintFlag]:
    """Check 8: at least one exit trigger is a fundamental threshold, not only
    price levels and indicators. A lead for the editor, never a hold."""
    triggers = digest.get("exit_triggers")
    if not isinstance(triggers, list) or not triggers:
        return []
    if any(_FUNDAMENTAL.search(" ".join(_digest_strings(t))) for t in triggers):
        return []
    return [LintFlag("minor", "trigger_mix", "digest", "digest.exit_triggers", str(triggers)[:300],
                     "no exit trigger is a fundamental threshold (growth, margin, cash flow, guidance) "
                     "with its current value")]


def _digest_strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _digest_strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _digest_strings(v)]
    return []


_BARE_RESULT = re.compile(r"\s*\$?\s?(\d[\d,]*(?:\.\d+)?)(?![\d.,])\s?(" + _SCALES + r")?\b(?!\s?(?:%|x\b|×))", re.I)
_PER_SHARE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(?:/\s*share|per\s+share|a\s+share)\b", re.I)


def check_target(decision: dict | None, facts: Facts) -> list[LintFlag]:
    """Check 6 (part): the target sits on the rating's side of the price."""
    if not decision:
        return []
    target, rating = decision.get("price_target"), (decision.get("rating") or "").title()
    close = facts.value("price.close")
    if not isinstance(target, (int, float)) or not isinstance(close, (int, float)) or not rating:
        return []
    flags: list[LintFlag] = []
    up = target > close
    if (rating in ("Buy", "Overweight") and not up) or (rating in ("Sell", "Underweight") and up):
        flags.append(LintFlag("load_bearing", "target_direction", "portfolio_manager", "pm.price_target",
                              f"{rating} with a target of {target} against a close of {close}",
                              "a buy-side target above the price, a sell-side one below", ["price.close"]))
    # The target's move sits in the rating's band (rating.RATING_BANDS), with a
    # point of slack at the edges.
    move = (target / close - 1) * 100 if close else 0.0
    if rating in RATING_BANDS and not ((rating in ("Buy", "Overweight") and not up)
                                       or (rating in ("Sell", "Underweight") and up)):
        low, high = RATING_BANDS[rating]
        if (low is not None and move < low - 1) or (high is not None and move > high + 1):
            flags.append(LintFlag("load_bearing", "target_tier", "portfolio_manager", "pm.price_target",
                                  f"{rating} with a target of {target} against a close of {close} ({move:+.1f}%)",
                                  f"{rating} needs a target move {band_text(rating)}; {move:+.1f}% is "
                                  f"{tier_for_move(move)}: change the rating or the target", ["price.close"]))
    # R7: the target is derived — its math ends at it, and it sits between
    # the bear and bull cases.
    math = decision.get("target_math") or ""
    math = math if isinstance(math, str) else " ".join(_digest_strings(math))
    # Every "= $x" result in the math; the target must be one of them. The
    # field often carries the bear and bull cases after the base case, so
    # the last result is not the target's (eval 2026-10-09: 4 of 6 GPT
    # reports flagged against their bull case).
    results, per_share = [], []
    for part in re.split(r"=|≈|→|~", math)[1:]:
        # A per-share result in the segment also counts ("= $2.09B ($6.08/share)");
        # a per-share input before the operator does not.
        per_share += [float(n.replace(",", "")) for n in _PER_SHARE.findall(part[:80])]
        figs = figures(part[:80])
        bare = _BARE_RESULT.match(part)
        if figs and figs[0].kind == "usd" and (not bare or figs[0].start <= bare.end()):
            results.append(figs[0].value)
        elif bare:                              # "= 6.08 per share"
            results.append(float(bare.group(1).replace(",", "")) * _SCALE.get((bare.group(2) or "").lower(), 1.0))
    tolerance = max(0.02 * target, 0.01)
    if results and not any(abs(r - target) <= tolerance for r in results + per_share):
        shown = ", ".join(f"{r:,.2f}" for r in results[:4])
        flags.append(LintFlag("load_bearing", "target_math", "portfolio_manager", "pm.price_target",
                              math[:300], f"no result in the math is the target {target:,.2f} (results: {shown})"))
    bear, bull = decision.get("bear_case_value"), decision.get("bull_case_value")
    if isinstance(bear, (int, float)) and isinstance(bull, (int, float)) and not min(bear, bull) <= target <= max(bear, bull):
        flags.append(LintFlag("load_bearing", "target_range", "portfolio_manager", "pm.price_target",
                              f"target {target} with bear {bear} and bull {bull}",
                              "the target sits between the bear and bull case values"))
    return flags


# ---- running it --------------------------------------------------------------------------------


def lint_text(text: str, facts: Facts, stage: str, field_name: str,
              sources: dict[str, str] | None = None) -> list[LintFlag]:
    """Checks 1–5 and 7 on one stage output (what the gates run)."""
    if not text:
        return []
    flags = [
        *check_numbers(text, facts, stage, field_name),
        *check_comparisons(text, facts, stage, field_name),
        *check_formulas(text, stage, field_name),
        *check_directions(text, facts, stage, field_name),
        *check_provenance(text, sources or {}, stage, field_name),
        *check_policy(text, stage, field_name),
        *check_style(text, stage, field_name),
        *check_process_language(text, stage, field_name),
        *check_voice(text, stage, field_name),
    ]
    if field_name in LOAD_BEARING_FIELDS:
        flags.extend(unsupported_figures(text, facts, stage, field_name))
        # R7 rule 3: a single author's unverified claim stays out of
        # load-bearing places (the headline, theses, ruling, decision).
        for m in re.finditer(r"unverified social claim", text, re.I):
            flags.append(LintFlag("load_bearing", "social_claim", stage, field_name,
                                  _sentence_at(text, m.start(), m.end())[:300],
                                  "a single author's unverified claim may not carry a load-bearing point"))
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
    # The digest fields stage_texts does not cover (points, excerpts, risk
    # lenses, sizing...): each point or entry as one text, so a figure in a
    # point's title and its citation in the detail are read together. Their
    # figure findings are minor: the points restate derived figures without
    # citations (eval 2026-10-09: 35 such findings in 79 reports, nearly all
    # correct derivations), so they are leads for the editor, not holds.
    linted = {f for _, f, _ in stage_texts(state)}
    for key, value in (state.get("report_digest") or {}).items():
        if f"digest.{key}" in linted:
            continue
        for item in (value if isinstance(value, list) else [value]):
            text = "\n".join(s.strip() for s in _digest_strings(item) if s.strip())
            flags.extend(f for f in lint_text(text, facts, "digest", f"digest.{key}", sources)
                         if f.severity != "style")
            flags.extend(unsupported_figures(text, facts, "digest", f"digest.{key}"))
    flags.extend(check_target(state.get("portfolio_decision"), facts))
    digest = state.get("report_digest") or {}
    rating = (state.get("portfolio_decision") or {}).get("rating") or state.get("final_rating")
    rating = rating if isinstance(rating, str) and rating.title() in RATING_BANDS else None
    for key, value in digest.items():
        for item in (value if isinstance(value, list) else [value]):
            flags.extend(check_rating_words("\n".join(_digest_strings(item)), rating, "digest", f"digest.{key}"))
    flags.extend(check_margin_wording(digest))
    flags.extend(check_trigger_mix(digest))
    flags.extend(check_trigger_headroom(digest))
    flags.extend(check_one_off_bases(state))
    flags.extend(check_target_debt(state.get("portfolio_decision"), facts))
    # Two figures in one sentence are one finding, not two.
    unique: dict[tuple, LintFlag] = {}
    for f in flags:
        unique.setdefault((f.kind, f.stage, f.field, f.quote), f)
    flags = list(unique.values())
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
