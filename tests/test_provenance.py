from finagent import provenance

EVIDENCE = [
    {"id": "M1", "result": {"total_return_pct": 22.82, "volatility_pct": 37.41, "as_of": "2026-10-21"}},
    {"id": "F1", "result": {"market_cap_bn": 5349.48, "pe_ratio_ttm": 27.73, "revenue": 215938000000}},
    {"id": "R1", "result": {"confidence": 0.95, "monte_carlo_var_pct": 13.49}},
    {"id": "D1", "result": {"passages": [{"ref": "risk_factors#70", "text": "two customers accounted for 35% of revenue"}]}},
]


def values(text):
    return [c.value for c in provenance.extract_numbers(text)]


def test_extracts_money_percent_and_multiples():
    assert values("Return was 22.82% with a P/E of 27.7x and cap of $5,349.48 billion.") == [22.82, 27.7, 5349.48]


def test_skips_years_dates_ids_and_parameters():
    text = "In FY2025 (10-K, Item 1A) the 52-week high on 2026-10-21 over 5 years and 10 trading days [M1] on Jan 25."
    assert values(text) == []


def test_index_names_are_not_figures():
    assert values("It beat the S&P 500 and the Russell 2000, per its Form 10-K.") == []


def test_unit_conversions_are_traced():
    report = provenance.check("Revenue of $215.94B, market cap of $5.35 trillion, 95% VaR of 13.49% [R1].", EVIDENCE)
    assert report.verified == report.checked == 4


def test_units_must_agree():
    # a P/E of 27.73 is not a 28% margin, and a 22.82% return is not a multiple of 22.82
    assert provenance.check("A margin of 28% [F1].", EVIDENCE).unverified
    assert provenance.check("A P/E of 22.82x [M1].", EVIDENCE).unverified


def test_rounded_numbers_count_only_from_the_evidence_their_sentence_cites():
    assert provenance.check("The stock returned about 23% [M1].", EVIDENCE).score == 1.0
    assert provenance.check("The stock returned about 23%.", EVIDENCE).score == 0.0
    assert provenance.check("The stock returned about 23% [F1].", EVIDENCE).score == 0.0
    # two decimals still trace anywhere in the ledger, reported as mis-cited
    claim = provenance.check("The stock returned 22.82% [F1].", EVIDENCE).claims[0]
    assert claim.verified and claim.miscited


def test_untraceable_number_is_flagged():
    report = provenance.check("Total return 22.82% [M1], analyst target 987.65% [M1].", EVIDENCE)
    assert [c.text for c in report.unverified] == ["987.65%"]
    assert "987.65% ⚠unverified" in provenance.annotate("Total return 22.82% [M1], analyst target 987.65% [M1].", report)


def test_close_but_wrong_numbers_are_not_accepted():
    # a 0.6% error on a large number used to slip through a relative tolerance
    assert provenance.check("market cap $5,380 billion", EVIDENCE).unverified


def test_citations_are_checked_against_the_cited_evidence():
    report = provenance.check("P/E of 27.73x [M1].", EVIDENCE)
    claim = report.claims[0]
    assert claim.verified and claim.miscited and claim.source == "F1"


def test_numbers_inside_filing_passages_count_as_evidence():
    assert provenance.check("Two customers were 35% of revenue [D1].", EVIDENCE).score == 1.0


def test_sec_identifiers_are_not_figures():
    assert values("No 10-K among the filings of CIK 2115436 (accession 0001045810-26-000021); it files 10-Qs.") == []
