from __future__ import annotations

from ..data import sec
from ..data.prices import adjusted_close
from . import TICKER, Tool, ToolContext, bn, parse_tickers, pct, rnd


def _ratio(num: float | None, den: float | None) -> float | None:
    return num / den if num is not None and den not in (None, 0) else None


def financial_statements(ctx: ToolContext, ticker: str, years: int = 3) -> dict:
    ticker = parse_tickers(ticker)[0]
    years = max(2, min(int(years), 5))
    cik, title = sec.ticker_to_cik(ctx.settings, ticker)
    facts = sec.company_facts(ctx.settings, cik)

    revenue = sec.annual_values(facts, "revenue")
    if not revenue:
        return {"ticker": ticker, "error": "no annual revenue reported in XBRL (common for banks and insurers)"}
    ends = sorted(revenue)[-(years + 1) :]  # one extra year to compute growth for the first shown year
    series = {m: sec.annual_values(facts, m, ends) for m in ("gross_profit", "operating_income", "net_income", "eps_diluted", "rnd", "operating_cash_flow", "capex")}

    rows = []
    for i, end in enumerate(ends):
        if i == 0 and len(ends) > years:
            continue
        rev = revenue[end]
        prev = revenue.get(ends[i - 1]) if i > 0 else None
        ocf, capex = series["operating_cash_flow"].get(end), series["capex"].get(end)
        rows.append(
            {
                "fiscal_year_end": end.isoformat(),
                "revenue_bn": bn(rev),
                "revenue_growth_pct": pct(_ratio(rev - prev, prev)) if prev else None,
                "gross_margin_pct": pct(_ratio(series["gross_profit"].get(end), rev)),
                "operating_margin_pct": pct(_ratio(series["operating_income"].get(end), rev)),
                "net_income_bn": bn(series["net_income"].get(end)),
                "net_margin_pct": pct(_ratio(series["net_income"].get(end), rev)),
                "eps_diluted": rnd(series["eps_diluted"].get(end)),
                "free_cash_flow_bn": bn(ocf - capex) if ocf is not None and capex is not None else None,
                "rnd_pct_of_revenue": pct(_ratio(series["rnd"].get(end), rev)),
            }
        )
    first, last = revenue[ends[-len(rows)]], revenue[ends[-1]]
    span = (ends[-1] - ends[-len(rows)]).days / 365.25
    cagr = pct((last / first) ** (1 / span) - 1) if span > 0 and first > 0 else None

    # Latest year first: small models tend to quote whatever they read first.
    latest = rows[-1]
    fy = f"{ticker} FY ending {latest['fiscal_year_end']}"
    facts = [f"{fy}: revenue ${latest['revenue_bn']}bn" + (f", up {latest['revenue_growth_pct']}% year on year" if latest["revenue_growth_pct"] is not None else "")]
    margins = [(name, latest[key]) for name, key in (("gross", "gross_margin_pct"), ("operating", "operating_margin_pct"), ("net", "net_margin_pct")) if latest[key] is not None]
    if margins:
        facts.append(f"{fy}: " + ", ".join(f"{name} margin {value}%" for name, value in margins))
    extras = [(label, latest[key]) for label, key in (("net income $", "net_income_bn"), ("diluted EPS $", "eps_diluted"), ("free cash flow $", "free_cash_flow_bn")) if latest[key] is not None]
    if extras:
        facts.append(f"{fy}: " + ", ".join(f"{label}{value}{'bn' if 'EPS' not in label else ''}" for label, value in extras))
    for row in reversed(rows[:-1]):
        facts.append(
            f"{ticker} FY ending {row['fiscal_year_end']}: revenue ${row['revenue_bn']}bn"
            + (f" (growth {row['revenue_growth_pct']}%)" if row["revenue_growth_pct"] is not None else "")
            + (f", operating margin {row['operating_margin_pct']}%" if row["operating_margin_pct"] is not None else "")
        )
    if cagr is not None:
        facts.append(f"{ticker} revenue CAGR from FY {rows[0]['fiscal_year_end'][:4]} to FY {latest['fiscal_year_end'][:4]}: {cagr}%")
    return {
        "ticker": ticker,
        "company": title,
        "source": "SEC XBRL company facts (10-K)",
        "fiscal_years": rows,
        "revenue_cagr_pct": cagr,
        "facts": facts,
    }


def valuation_snapshot(ctx: ToolContext, ticker: str) -> dict:
    ticker = parse_tickers(ticker)[0]
    cik, title = sec.ticker_to_cik(ctx.settings, ticker)
    facts = sec.company_facts(ctx.settings, cik)
    prices = adjusted_close(ctx.settings, ticker, "1y")
    price = float(prices.iloc[-1])

    shares = sec.shares_outstanding(facts)
    if not shares:
        return {"ticker": ticker, "error": "shares outstanding not reported in XBRL"}
    if (prices.index[-1].date() - shares[1]).days > 550:
        # e.g. Berkshire Hathaway reports in class-A equivalents under custom tags
        return {"ticker": ticker, "error": f"latest share count in XBRL is from {shares[1]}; valuation skipped rather than guessed"}
    market_cap = price * shares[0]

    rev_ttm = sec.ttm_value(facts, "revenue")
    ni_ttm = sec.ttm_value(facts, "net_income")
    cash = sec.latest_instant(facts, "cash")
    securities = sec.latest_instant(facts, "marketable_securities")
    debt = sec.latest_instant(facts, "long_term_debt")
    cash_total = (cash[0] if cash else 0.0) + (securities[0] if securities and cash and securities[1] == cash[1] else 0.0)
    ev = market_cap - cash_total + (debt[0] if debt else 0.0)

    ocf = sec.annual_values(facts, "operating_cash_flow")
    capex = sec.annual_values(facts, "capex")
    fcf = None
    if ocf:
        fy_end = max(ocf)
        if fy_end in capex:
            fcf = ocf[fy_end] - capex[fy_end]

    rev = rev_ttm["value"] if rev_ttm else None
    ni = ni_ttm["value"] if ni_ttm else None
    out = {
        "ticker": ticker,
        "company": title,
        "price": rnd(price),
        "price_date": prices.index[-1].date().isoformat(),
        "shares_outstanding_bn": rnd(shares[0] / 1e9, 3),
        "market_cap_bn": bn(market_cap),
        "ttm_revenue_bn": bn(rev),
        "ttm_net_income_bn": bn(ni),
        "ttm_period_end": (rev_ttm or {}).get("period_end"),
        "pe_ratio_ttm": rnd(_ratio(market_cap, ni)) if ni and ni > 0 else None,
        "price_to_sales_ttm": rnd(_ratio(market_cap, rev)),
        "cash_and_securities_bn": bn(cash_total),
        "long_term_debt_bn": bn(debt[0]) if debt else 0.0,
        "enterprise_value_bn": bn(ev),
        "ev_to_sales_ttm": rnd(_ratio(ev, rev)),
        "free_cash_flow_last_fy_bn": bn(fcf),
        "fcf_yield_pct": pct(_ratio(fcf, market_cap)),
        "method": "market cap = price x cover-page shares; TTM = " + ((rev_ttm or {}).get("method") or "n/a"),
    }
    facts = [f"{ticker} market cap: ${out['market_cap_bn']}bn at ${out['price']} per share ({out['price_date']})"]
    if out["pe_ratio_ttm"] is not None:
        facts.append(f"{ticker} P/E (trailing 12 months): {out['pe_ratio_ttm']}")
    if out["price_to_sales_ttm"] is not None:
        facts.append(f"{ticker} price-to-sales (trailing 12 months): {out['price_to_sales_ttm']}; EV-to-sales: {out['ev_to_sales_ttm']}")
    if out["ttm_revenue_bn"] is not None:
        facts.append(f"{ticker} trailing-12-month revenue ${out['ttm_revenue_bn']}bn and net income ${out['ttm_net_income_bn']}bn (to {out['ttm_period_end']})")
    if out["fcf_yield_pct"] is not None:
        facts.append(f"{ticker} free-cash-flow yield: {out['fcf_yield_pct']}% (last fiscal year FCF ${out['free_cash_flow_last_fy_bn']}bn)")
    facts.append(f"{ticker} cash and securities ${out['cash_and_securities_bn']}bn vs long-term debt ${out['long_term_debt_bn']}bn")
    out["facts"] = facts
    return out


TOOLS = [
    Tool(
        "financial_statements",
        "Annual results from SEC filings: revenue, growth, margins, net income, EPS, free cash flow.",
        {
            "type": "object",
            "properties": {"ticker": TICKER, "years": {"type": "integer", "description": "Fiscal years to return (2-5), default 3"}},
            "required": ["ticker"],
        },
        financial_statements,
    ),
    Tool(
        "valuation_snapshot",
        "Current valuation: market cap, P/E, price-to-sales, EV/sales, free-cash-flow yield, cash and debt.",
        {"type": "object", "properties": {"ticker": TICKER}, "required": ["ticker"]},
        valuation_snapshot,
    ),
]
