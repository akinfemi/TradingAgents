"""Sentiment sample rules (R7)."""

import pytest

from tradingagents.quality import social

ST = "\n".join(
    f"[2026-10-0{d}T10:00:00Z · @user{u} · Bullish] ONDS orders look strong {u}" for d, u in
    [(3, 1), (3, 2), (4, 3), (4, 4), (4, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 9)])
REPOST = "\n".join(f"[2026-10-05T10:00:00Z · @fan{u} · Bullish] Citizens initiates ONDS at Outperform" for u in range(6))
RD = ("r/stocks — 3 recent posts mentioning ONDS:\n  [2026-10-04 · u/alpha] ONDS drone orders\n"
      "  [2026-10-05 · u/beta] Is ONDS a buy?\n  [2026-10-05] ONDS dilution again")


@pytest.mark.unit
def test_counts_authors_events_and_span():
    s = social.sample(ST + "\n" + REPOST, RD)
    assert (s["stocktwits"], s["reddit"]) == (16, 3)
    assert s["authors"] == 9 + 6 + 2
    assert s["events"] == 9 + 1 + 3          # six reposts of one initiation are one event; one dup body
    assert (s["first"], s["last"]) == ("2026-10-03", "2026-10-05")
    assert "16 StockTwits messages and 3 Reddit posts from 17 authors" in social.coverage(s)


@pytest.mark.unit
def test_a_small_sample_has_no_score():
    s = social.sample(ST, "")
    out = social.apply({"overall_score": 6.4, "confidence": "high"}, s)
    assert out["overall_score"] is None and out["confidence"] == "low"
    assert "only one live source" in out["insufficient_reason"]
    assert "10 on-topic posts" in out["insufficient_reason"]


@pytest.mark.unit
def test_a_sufficient_sample_gets_a_whole_number_score():
    s = social.sample(ST + "\n" + REPOST, RD)
    out = social.apply({"overall_score": 6.4, "confidence": "medium"}, s)
    assert out["overall_score"] == 6 and out.get("insufficient_reason") is None


@pytest.mark.unit
def test_word_tickers_are_ambiguous():
    assert social.is_ambiguous("ON") and social.is_ambiguous("f") and not social.is_ambiguous("ONDS")


@pytest.mark.unit
def test_company_news_stands_in_for_a_rate_limited_reddit():
    many = "\n".join(f"[2026-10-05T10:00:00Z · @u{i} · Bullish] ONDS view {i}" for i in range(20))
    news = "## ONDS News\n\n### Ondas wins order (source: Reuters)\nLink: x"
    assert social.insufficient(social.sample(many, "<Reddit 429>", news)) is None
    assert "only one live source" in social.insufficient(social.sample(many, "<Reddit 429>", ""))
