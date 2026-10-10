"""The editor's reader fields: a tag, a one-line statement and a plain gloss
for each load-bearing finding, and decision flags as reader issues (the
report page's "What failed")."""

import pytest
from langchain_core.messages import AIMessage

from tradingagents.quality import editor

SHEET = {"facts": [{"key": "revenue.2026Q2", "value": 83.8e6, "unit": "usd"}]}


def _schema():
    submit = next(t for t in editor.TOOLS if t["name"] == "submit_review")
    return submit["input_schema"]["properties"]["findings"]["items"]


@pytest.mark.unit
def test_reader_fields_are_optional_in_the_schema():
    item = _schema()
    assert {"tag", "short", "plain"} <= set(item["properties"])
    assert not {"tag", "short", "plain"} & set(item["required"])
    assert item["properties"]["tag"]["enum"] == list(editor.READER_TAGS)


@pytest.mark.unit
def test_the_brief_asks_for_reader_language():
    assert "short (one line of at most 90 characters" in editor.BRIEF
    assert "no fact keys" in editor.BRIEF


@pytest.mark.unit
@pytest.mark.parametrize("text, tag", [
    ("The stated math gives $6.30, not the $6.35 target.", "arithmetic"),
    ("price target not derived", "price_target"),
    ("stop sits inside the entry zone", "price_target"),
    ("The 18.7x multiple is asserted; the 9% compression has no stated basis.", "valuation"),
    ("rating not supported by the evidence", "call"),
    ("Q3 is already reported, not upcoming", "dates"),
    ("Cites short interest, which the platform may not use", "sources"),
    ("Gross margin was 39.7%, not 47.2%", "figures"),
    ("something odd", "other"),
])
def test_tags_are_inferred_from_the_words(text, tag):
    assert editor.infer_tag(text) == tag


@pytest.mark.unit
def test_reader_fields_are_cleaned():
    f = editor.reader_fields({"severity": "load_bearing", "problem": "The stated math gives $6.30 [F:pt]",
                              "tag": "Price Target", "short": "x " * 60 + "[F:revenue.2026Q2]",
                              "plain": "Lands short [F:a, F:b] of the target."})
    assert f["tag"] == "price_target"
    assert len(f["short"]) <= editor.SHORT_MAX and f["short"].endswith("…") and "[F:" not in f["short"]
    assert f["plain"] == "Lands short of the target."
    old = editor.reader_fields({"severity": "minor", "problem": "The stated math gives $6.30", "tag": "bogus"})
    assert old["tag"] == "arithmetic" and "short" not in old and "plain" not in old


@pytest.mark.unit
def test_decision_flags_become_reader_issues():
    i = editor.flag_issue("price target not derived")
    assert i == {"flag": "price target not derived", "tag": "price_target", "short": "Price target not derived.",
                 "plain": "The report states a target but never shows how it gets there."}
    assert editor.flag_issue("rating not supported by the evidence")["tag"] == "call"


@pytest.mark.unit
def test_a_review_carries_reader_fields_and_flag_issues():
    finding = {"severity": "load_bearing", "location": {"stage": "pm", "quote": "target $6.35"},
               "problem": "The stated math gives $6.30, not $6.35.", "short": "Its own math gives $6.30, not $6.35.",
               "plain": "Working the report's figures through lands short of the target.", "tag": "arithmetic"}
    replies = [AIMessage(content="", tool_calls=[{"name": "submit_review", "id": "1", "args": {
        "findings": [finding, {"severity": "minor", "location": {"stage": "bull", "quote": "q"}, "problem": "p"}],
        "digest_patch": [], "decision_flags": ["price target not derived"], "editor_note": "", "hold_reason": ""}}])]

    class Bound:
        def invoke(self, messages, config=None):
            return replies.pop(0)

    class Fake:
        model_name = "x"

        def bind_tools(self, _tools, **_kw):
            return Bound()

    out = editor.review(Fake(), {"fact_sheet": SHEET}, {"flags": []}, sleep=lambda _s: None)
    assert out["findings"][0]["short"] == finding["short"] and out["findings"][0]["tag"] == "arithmetic"
    assert out["findings"][1]["tag"] == "other" and "short" not in out["findings"][1]
    assert out["flag_issues"][0]["tag"] == "price_target"
