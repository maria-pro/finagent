from __future__ import annotations

from ..data import sec
from ..data.prices import closing_price
from . import TICKER, Tool, ToolContext, bn, parse_tickers, pct, rnd

_NCI_NOTE = "includes non-controlling interests (ProfitLoss)"
_FINANCIALS_NOTE = (
    "banks, lenders, brokers and insurers (SIC 6000-6499) get price-to-book and return on equity instead of "
    "enterprise value, EV-to-sales, price-to-sales and free cash flow: their debt and cash flows are part of "
    "operations (deposits, customer funds, trading assets)"
)


def _ratio(num: float | None, den: float | None) -> float | None:
    return num / den if num is not None and den not in (None, 0) else None


def _industry(profile: dict) -> str:
    return f"SIC {profile['sic']} {profile['sic_description']}".strip() if profile.get("sic") else "SIC code not reported"


def financial_statements(ctx: ToolContext, ticker: str, years: int = 3) -> dict:
    ticker = parse_tickers(ticker)[0]
    years = max(2, min(int(years), 5))
    cik, title = sec.ticker_to_cik(ctx.settings, ticker)
    facts = sec.company_facts(ctx.settings, cik)
    profile = sec.company_profile(ctx.settings, cik)
    financial = sec.is_financial(profile["sic"])

    revenue = sec.annual_facts(facts, "revenue")
    if not revenue:
        return {
            "ticker": ticker,
            "error": f"no annual revenue in XBRL for {title} (CIK {cik}): some insurers tag revenue differently, "
            "and a newly registered successor company has no annual filings yet",
        }
    ends = sorted(revenue)[-(years + 1) :]  # one extra year to compute growth for the first shown year
    metrics = ["gross_profit", "operating_income", "net_income", "eps_diluted", "rnd"]
    metrics += ["equity"] if financial else ["operating_cash_flow", "capex"]
    series = {m: sec.annual_facts(facts, m, ends) for m in metrics}

    def value(metric: str, end) -> float | None:
        f = series.get(metric, {}).get(end)
        return f.value if f else None

    rows = []
    for i, end in enumerate(ends):
        if i == 0 and len(ends) > years:
            continue
        rev = revenue[end].value
        prev = revenue[ends[i - 1]].value if i > 0 else None
        row = {
            "fiscal_year_end": end.isoformat(),
            "revenue_bn": bn(rev),
            "revenue_growth_pct": pct(_ratio(rev - prev, prev)) if prev else None,
            "gross_margin_pct": pct(_ratio(value("gross_profit", end), rev)),
            "operating_margin_pct": pct(_ratio(value("operating_income", end), rev)),
            "net_income_bn": bn(value("net_income", end)),
            "net_margin_pct": pct(_ratio(value("net_income", end), rev)),
            "eps_diluted": rnd(value("eps_diluted", end)),
            "rnd_pct_of_revenue": pct(_ratio(value("rnd", end), rev)),
        }
        if financial:
            # return on average equity over the fiscal year
            opening = value("equity", ends[i - 1]) if i > 0 else None
            closing = value("equity", end)
            average = (opening + closing) / 2 if opening is not None and closing is not None else None
            row["roe_pct"] = pct(_ratio(value("net_income", end), average))
        else:
            ocf, capex = value("operating_cash_flow", end), value("capex", end)
            row["free_cash_flow_bn"] = bn(ocf - capex) if ocf is not None and capex is not None else None
        rows.append(row)
    first, last = revenue[ends[-len(rows)]].value, revenue[ends[-1]].value
    span = (ends[-1] - ends[-len(rows)]).days / 365.25
    cagr = pct((last / first) ** (1 / span) - 1) if span > 0 and first > 0 else None

    latest_end = ends[-1]
    latest_facts = {"revenue": revenue[latest_end], **{m: s[latest_end] for m, s in series.items() if latest_end in s}}
    ni_fact = latest_facts.get("net_income")
    nci = bool(ni_fact and ni_fact.concept == "ProfitLoss")

    # Latest year first: small models tend to quote whatever they read first.
    latest = rows[-1]
    fy = f"{ticker} FY ending {latest['fiscal_year_end']}"
    growth = latest["revenue_growth_pct"]
    change = "" if growth is None else f", {'up' if growth >= 0 else 'down'} {abs(growth)}% year on year"
    facts_out = [f"{fy}: revenue ${latest['revenue_bn']}bn{change}"]
    margins = [(name, latest[key]) for name, key in (("gross", "gross_margin_pct"), ("operating", "operating_margin_pct"), ("net", "net_margin_pct")) if latest[key] is not None]
    if margins:
        facts_out.append(f"{fy}: " + ", ".join(f"{name} margin {v}%" for name, v in margins))
    extras = [(label, latest.get(key)) for label, key in (("net income $", "net_income_bn"), ("diluted EPS $", "eps_diluted"), ("free cash flow $", "free_cash_flow_bn"))]
    extras = [(label, v) for label, v in extras if v is not None]
    if extras:
        facts_out.append(f"{fy}: " + ", ".join(f"{label}{v}{'bn' if 'EPS' not in label else ''}" for label, v in extras) + (f" (net income {_NCI_NOTE})" if nci else ""))
    if latest.get("roe_pct") is not None:
        facts_out.append(f"{fy}: return on equity {latest['roe_pct']}% (net income over average stockholders' equity)")
    for row in reversed(rows[:-1]):
        facts_out.append(
            f"{ticker} FY ending {row['fiscal_year_end']}: revenue ${row['revenue_bn']}bn"
            + (f" (growth {row['revenue_growth_pct']}%)" if row["revenue_growth_pct"] is not None else "")
            + (f", operating margin {row['operating_margin_pct']}%" if row["operating_margin_pct"] is not None else "")
        )
    if cagr is not None:
        facts_out.append(f"{ticker} revenue CAGR from FY {rows[0]['fiscal_year_end'][:4]} to FY {latest['fiscal_year_end'][:4]}: {cagr}%")
    out = {
        "ticker": ticker,
        "company": title,
        "industry": _industry(profile),
        "source": "SEC XBRL company facts (10-K)",
        "fiscal_years": rows,
        "revenue_cagr_pct": cagr,
        "net_income_includes_nci": nci,
        "xbrl_sources": {metric: f.source() for metric, f in latest_facts.items()},
        "filing_urls": sorted({sec.filing_url(cik, f.accession) for f in latest_facts.values()}),
        "facts": facts_out,
    }
    if financial:
        out["note"] = _FINANCIALS_NOTE
    return out


def valuation_snapshot(ctx: ToolContext, ticker: str) -> dict:
    ticker = parse_tickers(ticker)[0]
    cik, title = sec.ticker_to_cik(ctx.settings, ticker)
    facts = sec.company_facts(ctx.settings, cik)
    profile = sec.company_profile(ctx.settings, cik)
    financial = sec.is_financial(profile["sic"])
    prices = closing_price(ctx.settings, ticker, "1y")
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
    nci = bool(ni_ttm and "ProfitLoss" in ni_ttm["concept"])
    rev = rev_ttm["value"] if rev_ttm else None
    ni = ni_ttm["value"] if ni_ttm else None
    out = {
        "ticker": ticker,
        "company": title,
        "industry": _industry(profile),
        "price": rnd(price),
        "price_date": prices.index[-1].date().isoformat(),
        "shares_outstanding_bn": rnd(shares[0] / 1e9, 3),
        "market_cap_bn": bn(market_cap),
        "ttm_revenue_bn": bn(rev),
        "ttm_net_income_bn": bn(ni),
        "ttm_period_end": (rev_ttm or {}).get("period_end"),
        "pe_ratio_ttm": rnd(_ratio(market_cap, ni)) if ni and ni > 0 else None,
        "net_income_includes_nci": nci,
    }
    sources = {
        "revenue_ttm": f"{(rev_ttm or {}).get('concept')} {(rev_ttm or {}).get('accessions')}",
        "net_income_ttm": f"{(ni_ttm or {}).get('concept')} {(ni_ttm or {}).get('accessions')}",
    }
    method = f"market cap = closing price x {shares[2]} shares; TTM = {(rev_ttm or {}).get('method') or 'n/a'}"
    facts_out = [f"{ticker} market cap: ${out['market_cap_bn']}bn at ${out['price']} per share ({out['price_date']} close)"]
    if out["pe_ratio_ttm"] is not None:
        facts_out.append(f"{ticker} P/E (trailing 12 months): {out['pe_ratio_ttm']}" + (f" (net income {_NCI_NOTE})" if nci else ""))

    if financial:
        equity = sec.latest_instant(facts, "equity")
        out.update(
            {
                "book_value_bn": bn(equity.value) if equity else None,
                "book_value_date": equity.end.isoformat() if equity else None,
                "price_to_book": rnd(_ratio(market_cap, equity.value)) if equity and equity.value > 0 else None,
                "note": _FINANCIALS_NOTE,
            }
        )
        sources["book_value"] = equity.source() if equity else None
        method += "; book value = total stockholders' equity"
        if out["price_to_book"] is not None:
            facts_out.append(
                f"{ticker} price-to-book: {out['price_to_book']} (book value ${out['book_value_bn']}bn, stockholders' equity "
                f"at {out['book_value_date']})"
            )
        if out["ttm_net_income_bn"] is not None:
            facts_out.append(f"{ticker} trailing-12-month net income ${out['ttm_net_income_bn']}bn (to {out['ttm_period_end']})")
    else:
        sheet = sec.balance_sheet(facts)
        cash_total = (sheet["cash"] or 0.0) + (sheet["securities"] or 0.0)
        # No debt tags means unknown debt, not zero debt: then there is no enterprise value either.
        ev = market_cap - cash_total + sheet["debt"] if sheet["as_of"] and sheet["debt"] is not None else None
        ocf = sec.annual_facts(facts, "operating_cash_flow")
        capex = sec.annual_facts(facts, "capex")
        fcf = None
        if ocf:
            fy_end = max(ocf)
            if fy_end in capex:
                fcf = ocf[fy_end].value - capex[fy_end].value
        out.update(
            {
                "price_to_sales_ttm": rnd(_ratio(market_cap, rev)),
                "balance_sheet_date": sheet["as_of"],
                "cash_and_securities_bn": bn(cash_total),
                "total_debt_bn": bn(sheet["debt"]),
                "enterprise_value_bn": bn(ev),
                "ev_to_sales_ttm": rnd(_ratio(ev, rev)),
                "free_cash_flow_last_fy_bn": bn(fcf),
                "fcf_yield_pct": pct(_ratio(fcf, market_cap)),
            }
        )
        sources["balance_sheet"] = sheet["sources"]
        method += f"; debt = {sheet['debt_method']}; leases excluded"
        if out["price_to_sales_ttm"] is not None:
            facts_out.append(
                f"{ticker} price-to-sales (trailing 12 months): {out['price_to_sales_ttm']}"
                + (f"; EV-to-sales: {out['ev_to_sales_ttm']}" if out["ev_to_sales_ttm"] is not None else "")
            )
        if out["ttm_revenue_bn"] is not None:
            facts_out.append(f"{ticker} trailing-12-month revenue ${out['ttm_revenue_bn']}bn and net income ${out['ttm_net_income_bn']}bn (to {out['ttm_period_end']})")
        if out["fcf_yield_pct"] is not None:
            facts_out.append(f"{ticker} free-cash-flow yield: {out['fcf_yield_pct']}% (last fiscal year FCF ${out['free_cash_flow_last_fy_bn']}bn)")
        if sheet["as_of"] and ev is not None:
            facts_out.append(
                f"{ticker} cash and current securities ${out['cash_and_securities_bn']}bn vs total debt ${out['total_debt_bn']}bn "
                f"(balance sheet {sheet['as_of']}); enterprise value ${out['enterprise_value_bn']}bn"
            )
        elif sheet["as_of"]:
            facts_out.append(
                f"{ticker} cash and current securities ${out['cash_and_securities_bn']}bn (balance sheet {sheet['as_of']}); "
                "no debt reported in XBRL, so enterprise value is not computed"
            )
    out["method"] = method
    out["xbrl_sources"] = sources
    out["facts"] = facts_out
    return out


TOOLS = [
    Tool(
        "financial_statements",
        "Annual results from SEC filings: revenue, growth, margins, net income, EPS, and free cash flow "
        "(return on equity instead for banks, brokers and insurers).",
        {
            "type": "object",
            "properties": {"ticker": TICKER, "years": {"type": "integer", "description": "Fiscal years to return (2-5), default 3"}},
            "required": ["ticker"],
        },
        financial_statements,
    ),
    Tool(
        "valuation_snapshot",
        "Current valuation: market cap, P/E, and price-to-sales, EV/sales, free-cash-flow yield, cash and debt "
        "(price-to-book instead for banks, brokers and insurers).",
        {"type": "object", "properties": {"ticker": TICKER}, "required": ["ticker"]},
        valuation_snapshot,
    ),
]
