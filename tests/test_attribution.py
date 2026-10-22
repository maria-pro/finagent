from finagent import attribution

FACTS = {
    "M1": ["NVDA 5y total return: 881.02%", "NVDA 5y max drawdown: -20.21% (2025-10-29 to 2026-03-30)"],
    "M2": ["AMD 1y total return: 132.21%"],
    "F1": [
        "NVDA FY ending 2026-01-25: revenue $215.94bn, up 65.47% year on year",
        "NVDA FY ending 2026-01-25: gross margin 71.07%, operating margin 60.38%, net margin 55.6%",
        "NVDA FY ending 2024-01-28: revenue $60.92bn (growth 125.85%), operating margin 54.12%",
    ],
    "F2": ["NVDA P/E (trailing 12 months): 27.73", "NVDA trailing-12-month revenue $302.97bn and net income $192.88bn (to 2026-07-26)"],
    "R1": [
        "NVDA 95% 10-day Value at Risk, historical (based on 1255 trading days of returns): 14.27% of value",
        "NVDA 95% 10-day Value at Risk, filtered historical simulation scaled to current volatility: 13.75%",
    ],
}
PASSAGE = "Export controls on our data center GPUs could reduce demand in China and other markets and lead to excess inventory."
EVIDENCE = [{"id": k, "agent": k[0], "result": {"facts": v}} for k, v in FACTS.items()]
EVIDENCE.append({"id": "D1", "agent": "D", "result": {"passages": [{"ref": "risk_factors#3", "text": PASSAGE}]}})


def issues(text, tickers=("NVDA",), allowed=None):
    report = attribution.check(text, EVIDENCE, list(tickers), allowed)
    return [i for c in report.claims for i in c.issues]


def test_verbatim_copy_is_supported():
    assert issues("- NVDA 5y total return: 881.02% [M1]") == []
    assert issues("- NVDA FY ending 2026-01-25: gross margin 71.07%, operating margin 60.38%, net margin 55.6% [F1]") == []


def test_dropped_period_is_caught():
    assert issues("- NVDA total return: 881.02% [M1]") == ["881.02%: period '5y' missing"]


def test_wrong_period_is_caught():
    assert "period '1y' is not the source's" in " ".join(issues("- NVDA 1-year total return: 881.02% [M1]"))


def test_older_fiscal_year_must_be_named_but_latest_need_not_be():
    assert issues("- Revenue grew 65.47% year on year [F1]") == []
    assert issues("- Revenue grew 125.85% year on year [F1]") == ["125.85%: fiscal year missing (source: FY ending 2024, not the latest)"]
    assert issues("- FY2024 revenue grew 125.85% [F1]") == []


def test_company_must_match_and_be_named_in_comparisons():
    assert "company AMD but the source is about NVDA" in " ".join(issues("- AMD 5y total return: 881.02% [M1]", ("NVDA", "AMD")))
    assert issues("- 5y total return: 881.02% [M1]", ("NVDA", "AMD")) == ["881.02%: company missing (source: NVDA)"]


def test_direction_words_must_agree_with_the_sign():
    assert "direction contradicts" in " ".join(issues("- NVDA fell 881.02% over 5y [M1]"))
    assert issues("- NVDA 5y max drawdown: -20.21% [M1]") == []
    assert issues("- The stock could lose 14.27% of value in 10 days at 95%, historical [R1]") == []  # loss size, no sign


def test_metric_and_method_labels_must_match():
    assert "says operating but the source says gross" in " ".join(issues("- NVDA FY2026 operating margin 71.07% [F1]"))
    assert "says filtered but the source says historical" in " ".join(
        issues("- NVDA 95% 10-day VaR from filtered historical simulation: 14.27% [R1]")
    )


def test_trailing_period_matters_only_where_fiscal_values_are_also_cited():
    assert issues("- NVDA P/E: 27.73 [F2]") == []
    assert issues("- NVDA revenue $302.97bn [F1, F2]") == ["$302.97bn: period 'trailing 12 months' missing"]


def test_qualitative_claims_are_checked_against_the_passage_wording():
    assert issues("- Export controls on data center GPUs could reduce demand in China [D1]") == []
    assert issues("- The company faces patent lawsuits from rival chipmakers [D1]")[0].startswith("not supported by the cited passage")


def test_interpretation_is_not_scored_but_uncited_numbers_are():
    report = attribution.check("NVIDIA looks expensive relative to its history.", EVIDENCE, ["NVDA"])
    assert report.checked == 0
    assert issues("NVIDIA trades at 27.73 times earnings.") == ["numbers without a citation"]


def test_citing_evidence_the_agent_never_received():
    assert "cites evidence it was not given (M1)" in issues("- NVDA 5y total return: 881.02% [M1]", allowed={"R1"})


def test_number_absent_from_cited_source():
    assert issues("- NVDA 5y total return: 900.5% [M1]") == ["900.5% is not in the cited source"]


def test_annotation_and_feedback_name_the_source():
    text = "- NVDA total return: 881.02% [M1]\n- NVDA 5y total return: 881.02% [M1]"
    report = attribution.check(text, EVIDENCE, ["NVDA"])
    marked = attribution.annotate(text, report)
    assert marked.splitlines()[0].endswith("⚠ 881.02%: period '5y' missing")
    assert not marked.splitlines()[1].count("⚠")
    assert 'Source says: "NVDA 5y total return: 881.02%"' in report.feedback()[0]
    assert report.summary()["supported"] == 1 and report.checked == 2


def test_company_names_count_as_the_ticker():
    evidence = EVIDENCE + [{"id": "F9", "agent": "F", "result": {"ticker": "NVDA", "company": "NVIDIA CORP", "facts": ["NVDA P/E (trailing 12 months): 27.73"]}}]
    report = attribution.check("- NVIDIA P/E (trailing 12 months): 27.73 [F9]", evidence, ["NVDA", "AMD"])
    assert report.score == 1.0
    assert attribution.company_aliases(evidence, ["NVDA", "AMD"]) == {"NVDA": {"NVIDIA"}, "AMD": set()}


def test_growth_rate_spans_need_their_start_year():
    evidence = [{"id": "F1", "agent": "F", "result": {"facts": ["NVDA revenue CAGR from FY 2024 to FY 2026: 88.68%"]}}]
    assert attribution.check("- NVDA revenue CAGR FY2024-FY2026: 88.68% [F1]", evidence, ["NVDA"]).score == 1.0
    flagged = attribution.check("- NVDA revenue growth of 88.68% [F1]", evidence, ["NVDA"]).flagged[0]
    assert flagged.issues == ["88.68%: period FY2024-FY2026 missing (source: NVDA revenue CAGR from FY 2024 to FY 2026)"]


def test_citing_a_failed_tool_call_is_named_as_such():
    evidence = EVIDENCE + [{"id": "D2", "agent": "D", "error": "no 10-K", "result": {"error": "no 10-K"}}]
    found = attribution.check("- Free cash flow was $12.5 billion [D2]", evidence, ["NVDA"]).flagged[0].issues
    assert found == ["cites a tool call that failed (D2)"]
