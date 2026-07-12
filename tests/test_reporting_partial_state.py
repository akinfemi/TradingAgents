"""write_report_tree accepts partial states — the salvage contract.

An interrupted run's merged state has only some sections; callers salvage
what completed. This pins the .get() tolerance so a refactor that starts
requiring completed-run keys fails here instead of silently breaking salvage.
"""

from tradingagents.reporting import write_report_tree


def test_partial_state_writes_only_completed_sections(tmp_path):
    partial = {
        "market_report": "market text",
        "sentiment_report": "",  # empty — analyst never finished
        # news/fundamentals/debate/risk/decision keys entirely absent
    }
    complete_report = write_report_tree(partial, "NVDA", tmp_path / "reports")

    assert complete_report.exists()
    content = complete_report.read_text(encoding="utf-8")
    assert "market text" in content
    assert (tmp_path / "reports" / "1_analysts" / "market.md").exists()
    assert not (tmp_path / "reports" / "1_analysts" / "sentiment.md").exists()


def test_empty_state_still_writes_report_shell(tmp_path):
    complete_report = write_report_tree({}, "NVDA", tmp_path / "reports")
    assert complete_report.exists()
