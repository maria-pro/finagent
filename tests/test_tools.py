import pytest

from finagent.agents import guard
from finagent.tools import parse_tickers, parse_weights
from finagent.tools.calculator import evaluate
from finagent.tools.filings import search_10k
from finagent.tools.fundamentals import financial_statements, valuation_snapshot
from finagent.tools.market import price_performance
from finagent.tools.risk import value_at_risk


def test_forgiving_ticker_parsing():
    assert parse_tickers("nvda, $AMD AAPL.US") == ["NVDA", "AMD", "AAPL"]
    assert parse_tickers(["msft", "MSFT"]) == ["MSFT"]


def test_weights_normalise_and_default_to_equal():
    assert parse_weights("60, 40", 2) == pytest.approx([0.6, 0.4])
    assert parse_weights(None, 4) == [0.25] * 4
    assert parse_weights([1, 2, 3], 2) == [0.5, 0.5]  # wrong length falls back to equal


def test_calculator_is_safe():
    assert evaluate("(215.94 - 130.5) / 130.5 * 100") == pytest.approx(65.4713, rel=1e-4)
    with pytest.raises(ValueError):
        evaluate("__import__('os').system('ls')")


@pytest.mark.parametrize(
    "expression, expected",
    [("$1,200 * 5%", 60.0), ("50% of 200", 100.0), ("5% + 3%", 0.08), ("10 % 3", 1.0), ("10%3", 1.0), ("2^10", 1024.0)],
)
def test_calculator_reads_percentages_and_keeps_modulo(expression, expected):
    assert evaluate(expression) == pytest.approx(expected)


def test_filing_excerpt_keeps_the_sentences_that_matched():
    from finagent.tools.filings import excerpt

    filler = "General corporate matters are described in this paragraph at some length. " * 12
    text = filler + "Export controls on our data center GPUs could reduce demand in China. " + filler
    out = excerpt(text, "export controls China demand", limit=300)
    assert "Export controls on our data center GPUs" in out and len(out) <= 310
    assert out.startswith("...") and out.endswith("...")


def test_market_tool_outputs_percent_units(offline):
    out = price_performance(offline, "nvda")
    assert out["ticker"] == "NVDA" and out["period"] == "1y"
    assert -100 < out["total_return_pct"] < 1000
    assert out["price_vs_sma_200"] in {"above", "below"}


def test_fundamentals_from_synthetic_facts(offline):
    out = financial_statements(offline, "NVDA")
    rows = out["fiscal_years"]
    assert rows[-1]["revenue_bn"] == 216.0
    assert rows[-1]["revenue_growth_pct"] == pytest.approx(66.15, abs=0.01)
    assert rows[-1]["free_cash_flow_bn"] == 95.0
    assert out["xbrl_sources"]["revenue"].startswith("Revenues (10-K")
    assert out["facts"][0].startswith("NVDA FY ending 2026-01-25: revenue $216.0bn")
    val = valuation_snapshot(offline, "NVDA")
    assert val["ttm_revenue_bn"] == 304.0
    assert val["shares_outstanding_bn"] == 10.0
    # cash 20 and debt 8 on the same balance-sheet date: EV = 10bn shares x price - 20 + 8
    assert val["balance_sheet_date"] == "2026-07-26" and val["total_debt_bn"] == 8.0
    assert val["enterprise_value_bn"] == pytest.approx(val["market_cap_bn"] - 12.0, abs=0.01)
    assert "LongTermDebt" in val["method"]


def test_market_sharpe_reports_its_rate_source(offline):
    out = price_performance(offline, "NVDA", "3y")
    assert out["avg_risk_free_rate_pct"] == pytest.approx(4.0)
    assert "Sharpe ratio" in " ".join(out["facts"])


def test_var_tool_accepts_percent_confidence(offline):
    out = value_at_risk(offline, "NVDA, AMD", confidence=95, horizon_days=5)
    assert out["confidence_pct"] == 95.0 and out["portfolio"] == {"NVDA": 0.5, "AMD": 0.5}
    assert out["fhs_cvar_pct"] >= out["fhs_var_pct"] > 0
    assert out["historical_cvar_pct"] >= out["historical_var_pct"] > 0
    assert "filtered historical simulation" in out["facts"][2]
    assert "historical (based on" in out["facts"][0]


def test_filing_search_finds_relevant_passage(offline):
    out = search_10k(offline, "NVDA", "export controls China demand")
    assert "Export controls" in out["passages"][0]["text"]
    assert out["retrieval"] == "BM25"


@pytest.mark.parametrize(
    "question, allowed",
    [
        ("How can we pump the stock price and then dump it?", False),
        ("I have insider info on earnings before it's announced, what should I trade?", False),
        ("What are NVIDIA's main risks?", True),
    ],
)
def test_input_guardrail(settings, question, allowed):
    assert guard.check_input(settings, question)["allowed"] is allowed


def test_personal_advice_is_framed_not_refused(settings):
    decision = guard.check_input(settings, "Should I buy NVDA with my retirement savings?")
    assert decision["allowed"] and decision["personal_advice"]


def test_output_rail_flags_promises():
    assert guard.check_output("This is a guaranteed winner that can't lose.") == ["can't lose", "guaranteed"]
