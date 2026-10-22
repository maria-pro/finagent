"""Assemble the final research note.

Only the opening summary and the specialists' bullet points are written by
language models. The key-figures table, the evidence ledger and the run
statistics are generated directly from tool outputs, so they cannot be
hallucinated.
"""

from __future__ import annotations

from datetime import datetime

from . import provenance
from .agents.guard import DISCLAIMER

_SOURCES = {
    "price_performance": "Yahoo Finance daily prices",
    "benchmark_comparison": "Yahoo Finance daily prices",
    "financial_statements": "SEC EDGAR XBRL company facts",
    "valuation_snapshot": "SEC EDGAR XBRL + Yahoo Finance price",
    "value_at_risk": "computed from 5y daily prices",
    "stress_test": "computed from 10y daily prices",
    "calculator": "calculation",
}

# (label, tool, field, suffix); {period} is filled from the tool call that produced the value
_KEY_FIGURES = [
    ("Price", "price_performance", "last_price", ""),
    ("Total return ({period})", "price_performance", "total_return_pct", "%"),
    ("Volatility, annualised ({period})", "price_performance", "volatility_pct", "%"),
    ("Max drawdown ({period})", "price_performance", "max_drawdown_pct", "%"),
    ("Beta vs SPY ({period})", "benchmark_comparison", "beta", ""),
    ("Market cap ($bn)", "valuation_snapshot", "market_cap_bn", ""),
    ("P/E (TTM)", "valuation_snapshot", "pe_ratio_ttm", "x"),
    ("Price / sales (TTM)", "valuation_snapshot", "price_to_sales_ttm", "x"),
    ("FCF yield", "valuation_snapshot", "fcf_yield_pct", "%"),
]


def _fmt(value, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.2f}{suffix}"
    return f"{value}{suffix}"


def key_figures(evidence: list[dict], tickers: list[str]) -> str:
    table: dict[str, dict[str, str]] = {}
    for label, tool, fieldname, suffix in _KEY_FIGURES:
        for ev in evidence:
            if ev["tool"] != tool or ev.get("error"):
                continue
            ticker = str(ev["result"].get("ticker", "")).upper()
            if ticker in tickers and fieldname in ev["result"]:
                row = label.format(period=ev["result"].get("period", ""))
                table.setdefault(row, {}).setdefault(ticker, f"{_fmt(ev['result'][fieldname], suffix)} [{ev['id']}]")
    for ev in evidence:
        if ev["tool"] == "financial_statements" and not ev.get("error") and ev["result"].get("fiscal_years"):
            ticker = ev["result"]["ticker"]
            latest = ev["result"]["fiscal_years"][-1]
            for label, key, suffix in (
                ("Revenue, latest FY ($bn)", "revenue_bn", ""),
                ("Revenue growth, latest FY", "revenue_growth_pct", "%"),
                ("Operating margin, latest FY", "operating_margin_pct", "%"),
            ):
                table.setdefault(label, {}).setdefault(ticker, f"{_fmt(latest.get(key), suffix)} [{ev['id']}]")
        if ev["tool"] == "value_at_risk" and not ev.get("error"):
            r = ev["result"]
            label = f"{r['horizon_days']}-day VaR {r['confidence_pct']:.0f}% (Monte Carlo)"
            single = len(r.get("portfolio", {})) == 1
            for ticker in r.get("portfolio", {}):
                cell = f"{_fmt(r['monte_carlo_var_pct'], '%')} [{ev['id']}]" + ("" if single else " (portfolio)")
                row = table.setdefault(label, {})
                if single or ticker not in row:  # a stock's own VaR beats a portfolio figure
                    row[ticker] = cell
    if not table:
        return ""
    lines = ["| Metric | " + " | ".join(tickers) + " |", "|---|" + "---|" * len(tickers)]
    for label, row in table.items():
        lines.append(f"| {label} | " + " | ".join(row.get(t, "–") for t in tickers) + " |")
    return "\n".join(lines)


def evidence_table(evidence: list[dict]) -> str:
    lines = ["| ID | Agent | Tool call | Chosen by | Source |", "|---|---|---|---|---|"]
    for ev in evidence:
        args = ", ".join(f"{k}={v}" for k, v in ev["args"].items())
        source = ev["result"].get("url") or _SOURCES.get(ev["tool"], "")
        if ev.get("error"):
            source = f"error: {ev['error']}"
        lines.append(f"| {ev['id']} | {ev['agent']} | `{ev['tool']}({args})` | {ev['chosen_by']} | {source} |")
    return "\n".join(lines)


def run_stats(state: dict, settings, wall_s: float) -> dict:
    calls = state.get("llm_calls", [])
    evidence = state.get("evidence", [])
    by_model = sum(1 for e in evidence if str(e["chosen_by"]).startswith("model"))
    mc = next((e["result"] for e in evidence if e["tool"] == "value_at_risk" and not e.get("error")), {})
    retrieval = next((e["result"].get("retrieval") for e in evidence if e["tool"] == "search_10k" and not e.get("error")), None)
    return {
        "llm_calls": len(calls),
        "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
        "completion_tokens": sum(c["completion_tokens"] for c in calls),
        "model_seconds": round(sum(c["latency_s"] for c in calls), 1),
        "wall_seconds": round(wall_s, 1),
        "tool_calls": len(evidence),
        "tool_calls_chosen_by_model": by_model,
        "revisions": len(state.get("revisions", [])),
        "monte_carlo_backend": mc.get("compute_backend"),
        "retrieval": retrieval,
        "endpoint": settings.base_url,
        "orchestrator_model": settings.orchestrator_model,
        "specialist_model": settings.specialist_model,
    }


def build_report(state: dict, settings, wall_s: float) -> tuple[str, dict]:
    tickers = state["tickers"]
    evidence = state.get("evidence", [])
    findings = [state["findings"][t["agent"]] for t in state.get("plan", []) if t["agent"] in state.get("findings", {})]

    narrative = state.get("narrative", "")
    body_parts = [narrative] + [f["text"] for f in findings]
    check = provenance.check("\n\n".join(body_parts), evidence)

    sections = [f"# Research note: {', '.join(tickers)}", f"**Question:** {state['question']}", ""]
    meta = (
        f"*Generated {datetime.now():%Y-%m-%d %H:%M} by FinAgent · {len(findings)} specialist agents · "
        f"models `{settings.orchestrator_model}` / `{settings.specialist_model}`*"
    )
    sections += [meta, "", f"> {DISCLAIMER}", ""]
    sections.append(provenance.annotate(narrative, provenance.check(narrative, evidence)) if narrative else "")
    sections += ["", "## Specialist findings"]
    for f in findings:
        note = " *(revised after fact-check)*" if f.get("revised") else ""
        sections += [f"### {f['title']}{note}", f"*Focus: {f['focus']}*", "", provenance.annotate(f["text"], provenance.check(f["text"], evidence)), ""]

    figures = key_figures(evidence, tickers)
    if figures:
        sections += ["## Key figures (generated from tool outputs, not by a model)", "", figures, ""]
    sections += ["## Evidence ledger", "", evidence_table(evidence), ""]

    stats = run_stats(state, settings, wall_s)
    stats["provenance"] = check.summary()
    stats["promissory_phrases"] = state.get("promissory_phrases", [])
    flagged = f"; {len(check.unverified)} flagged ⚠" if check.unverified else ""
    sections += [
        "## How this note was produced",
        "",
        f"- **Number provenance:** {check.verified} of {check.checked} numbers in the text trace back to a tool result "
        f"({check.score:.0%}){flagged}.",
        f"- **Agents:** planner → {', '.join(f['agent'] for f in findings)} → fact-check"
        f"{' → ' + str(stats['revisions']) + ' revision(s)' if stats['revisions'] else ''} → lead analyst.",
        f"- **Tool calls:** {stats['tool_calls']} ("
        + ("all chosen by the models" if stats["tool_calls_chosen_by_model"] == stats["tool_calls"]
           else f"{stats['tool_calls_chosen_by_model']} chosen by the models, {stats['tool_calls'] - stats['tool_calls_chosen_by_model']} added by coverage checks")
        + ").",
        f"- **Language model:** {stats['llm_calls']} calls, {stats['prompt_tokens']:,} prompt + {stats['completion_tokens']:,} "
        f"completion tokens, {stats['model_seconds']} s of model time; {stats['wall_seconds']} s end to end.",
        f"- **Compute:** Monte Carlo on {stats['monte_carlo_backend'] or 'n/a'}; 10-K retrieval: {stats['retrieval'] or 'n/a'}; endpoint `{settings.base_url}`.",
        "",
    ]
    return "\n".join(sections), stats
