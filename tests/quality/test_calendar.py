"""Earnings dates confirmed from the company's own announcement (R4b)."""

import pytest

from tradingagents.quality.calendar import confirmed_date

NEWS = """## ONDS News, from 2026-09-20 to 2026-10-08:

### Ondas Falls 5% Despite $165 Million in New Orders (source: 24/7 Wall St.)
Link: https://example.com/a

### Ondas Holdings to Report Third Quarter 2026 Financial Results on November 12, 2026 (source: GlobeNewswire)
Link: https://example.com/b
"""


@pytest.mark.unit
def test_the_companys_announcement_confirms_the_date():
    found = confirmed_date(NEWS, ["ONDS", "Ondas"], "2026-10-08", estimate="2026-11-12")
    assert found["date"] == "2026-11-12"
    assert found["link"] == "https://example.com/b"
    assert found["source"] == "GlobeNewswire"


@pytest.mark.unit
@pytest.mark.parametrize("title, expected", [
    ("NVIDIA Announces Date of Third-Quarter Fiscal 2027 Financial Results: Nov. 18", "2026-11-18"),
    ("Apple to host Q4 earnings call on Thursday, October 30", "2026-10-30"),
    ("Ondas Sets Conference Call for Q3 Results on Nov 12th", "2026-11-12"),
])
def test_common_headline_shapes(title, expected):
    news = f"### {title} (source: Business Wire)\nLink: https://x\n"
    names = ["NVDA", "NVIDIA", "Apple", "AAPL", "Ondas"]
    assert confirmed_date(news, names, "2026-10-08")["date"] == expected


@pytest.mark.unit
@pytest.mark.parametrize("title", [
    "Ondas reported third quarter results on November 13, 2025",            # past results, wrong year
    "Analysts expect Ondas to beat when it reports on November 12",          # no announcement verb on the company
    "Red Cat to Report Third Quarter Results on November 12",                # another company
    "Ondas to Report Third Quarter Results on March 3",                      # far from the estimate
])
def test_other_headlines_are_not_confirmations(title):
    news = f"### {title} (source: X)\nLink: https://x\n"
    assert confirmed_date(news, ["ONDS", "Ondas"], "2026-10-08", estimate="2026-11-12") is None
