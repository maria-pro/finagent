from datetime import date

import pytest
from conftest import SYNTHETIC_FACTS

from finagent.data import sec


def test_annual_values_use_one_year_durations_only():
    revenue = sec.annual_values(SYNTHETIC_FACTS, "revenue")
    assert revenue == {date(2024, 1, 28): 60e9, date(2025, 1, 26): 130e9, date(2026, 1, 25): 216e9}


def test_ttm_is_fiscal_year_plus_ytd_minus_prior_ytd():
    ttm = sec.ttm_value(SYNTHETIC_FACTS, "revenue")
    assert ttm["value"] == pytest.approx(216e9 + 178e9 - 90e9)
    assert ttm["period_end"] == "2026-07-26"


def test_restated_values_win():
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"start": "2025-01-01", "end": "2025-12-31", "val": 100.0, "form": "10-K", "filed": "2026-02-01"},
        {"start": "2025-01-01", "end": "2025-12-31", "val": 104.0, "form": "10-K/A", "filed": "2026-05-01"},
    ]}}}}}
    assert sec.annual_values(facts, "revenue") == {date(2025, 12, 31): 104.0}


def test_concept_switch_is_merged():
    facts = {"facts": {"us-gaap": {
        "SalesRevenueNet": {"units": {"USD": [{"start": "2017-01-01", "end": "2017-12-31", "val": 50.0, "form": "10-K", "filed": "2018-02-01"}]}},
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            {"start": "2018-01-01", "end": "2018-12-31", "val": 60.0, "form": "10-K", "filed": "2019-02-01"}]}},
    }}}
    assert sec.annual_values(facts, "revenue") == {date(2017, 12, 31): 50.0, date(2018, 12, 31): 60.0}


def test_shares_sum_share_classes_on_cover_page():
    facts = {"facts": {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        {"end": "2026-07-01", "val": 5e9, "form": "10-Q", "filed": "2026-07-30", "accn": "a"},
        {"end": "2026-07-01", "val": 1e9, "form": "10-Q", "filed": "2026-07-30", "accn": "b"},
    ]}}}}}
    assert sec.shares_outstanding(facts)[0] == 6e9


def test_section_split_prefers_body_over_table_of_contents():
    toc = "Item 1A. Risk Factors\nItem 1B. Unresolved Staff Comments\nItem 7. Management's Discussion\n"
    body = (
        "Item 1A. Risk Factors\n" + "Our business faces many risks. " * 40 + "\n"
        "Item 1B. Unresolved Staff Comments\nNone.\n"
        "Item 7. Management's Discussion and Analysis\n" + "Revenue grew. " * 60 + "\n"
        "Item 8. Financial Statements\nTables."
    )
    sections = sec.split_sections(toc + body)
    assert sections["risk_factors"].startswith("Item 1A. Risk Factors\nOur business")
    assert "Revenue grew." in sections["mda"] and "Tables" not in sections["mda"]


def test_html_to_text_drops_hidden_xbrl_header():
    html = '<html><body><div style="display:none"><ix:header>junk 123</ix:header></div><p>Item 1A.</p><p>Risk Factors</p></body></html>'
    text = sec.html_to_text(html)
    assert "junk" not in text and "Risk Factors" in text


def _bs(**concepts):
    facts = {"facts": {"us-gaap": {}}}
    for concept, value in concepts.items():
        facts["facts"]["us-gaap"][concept] = {"units": {"USD": [{"end": "2026-06-30", "val": value, "form": "10-Q", "filed": "2026-07-30", "accn": "x"}]}}
    return facts


@pytest.mark.parametrize(
    "concepts, debt, method",
    [
        ({"LongTermDebtNoncurrent": 90.0, "DebtCurrent": 15.0}, 105.0, "LongTermDebtNoncurrent + DebtCurrent"),
        ({"LongTermDebtNoncurrent": 90.0, "LongTermDebtCurrent": 10.0, "ShortTermBorrowings": 5.0}, 105.0, "LongTermDebtNoncurrent + LongTermDebtCurrent + ShortTermBorrowings"),
        # LongTermDebt already includes the current portion by definition; only add short-term borrowings
        ({"LongTermDebt": 100.0, "CommercialPaper": 5.0}, 105.0, "LongTermDebt + CommercialPaper"),
        ({}, None, "no debt tags"),  # unknown, not zero: no enterprise value is computed from it
    ],
)
def test_balance_sheet_debt_is_composed_explicitly(concepts, debt, method):
    sheet = sec.balance_sheet(_bs(CashAndCashEquivalentsAtCarryingValue=50.0, **concepts))
    assert sheet["debt"] == debt and sheet["as_of"] == "2026-06-30"
    assert sheet["debt_method"].startswith(method)


def test_net_income_prefers_the_parent_over_profit_including_minorities():
    def annual(value):
        return {"units": {"USD": [{"start": "2025-01-01", "end": "2025-12-31", "val": value, "form": "10-K", "filed": "2026-02-01", "accn": "a"}]}}

    both = {"facts": {"us-gaap": {"ProfitLoss": annual(110.0), "NetIncomeLoss": annual(100.0)}}}
    assert sec.annual_values(both, "net_income") == {date(2025, 12, 31): 100.0}
    only_total = {"facts": {"us-gaap": {"ProfitLoss": annual(110.0)}}}
    fact = sec.annual_facts(only_total, "net_income")[date(2025, 12, 31)]
    assert fact.value == 110.0 and fact.concept == "ProfitLoss"


def test_total_revenues_win_over_contract_revenue_in_the_same_year():
    # Berkshire Hathaway reports both; contract revenue alone leaves out insurance premiums and leasing
    def annual(value):
        return {"units": {"USD": [{"start": "2025-01-01", "end": "2025-12-31", "val": value, "form": "10-K", "filed": "2026-02-01", "accn": "a"}]}}

    facts = {"facts": {"us-gaap": {"RevenueFromContractWithCustomerExcludingAssessedTax": annual(247.0), "Revenues": annual(371.0)}}}
    fact = sec.annual_facts(facts, "revenue")[date(2025, 12, 31)]
    assert fact.value == 371.0 and fact.concept == "Revenues"


@pytest.mark.parametrize("sic, financial", [(6021, True), (6211, True), (6331, True), (3674, False), (6798, False), (None, False)])
def test_financials_are_recognised_by_sic_code(sic, financial):
    assert sec.is_financial(sic) is financial


def test_company_profile_reads_the_sic_code(monkeypatch, settings):
    monkeypatch.setattr(sec, "submissions", lambda _s, _cik: {"sic": "6211", "sicDescription": "Security Brokers, Dealers & Flotation Companies"})
    assert sec.company_profile(settings, 886982) == {"sic": 6211, "sic_description": "Security Brokers, Dealers & Flotation Companies"}


def test_year_end_balance_is_found_when_a_later_10q_repeats_it():
    # the 10-Q filed after the 10-K shows the same year-end balance as its comparison column
    facts = {"facts": {"us-gaap": {"StockholdersEquity": {"units": {"USD": [
        {"end": "2025-12-31", "val": 120.0, "form": "10-K", "filed": "2026-02-25", "accn": "k"},
        {"end": "2025-12-31", "val": 120.0, "form": "10-Q", "filed": "2026-05-01", "accn": "q"},
        {"end": "2026-03-31", "val": 123.0, "form": "10-Q", "filed": "2026-05-01", "accn": "q"},
    ]}}}}}
    assert sec.annual_values(facts, "equity", [date(2025, 12, 31)]) == {date(2025, 12, 31): 120.0}
