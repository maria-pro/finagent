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
