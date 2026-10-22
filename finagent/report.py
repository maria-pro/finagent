"""Assemble the final research note.

Only the opening summary and the specialists' bullet points are written by
language models. The key-figures table, the evidence ledger and the run
statistics are generated directly from tool outputs, so they cannot be
hallucinated. Model-written text is annotated inline wherever the fact-check
found a problem: an untraced number, a claim that does not match its source, or
promissory language.
"""

from __future__ import annotations

from datetime import datetime

from . import attribution, provenance
from .agents.guard import DISCLAIMER, flag_promissory
from .checks import note_quality

_SOURCES = {
    "price_performance": "Yahoo Finance daily prices; US Treasury bill rates",
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
    ("Price / book", "valuation_snapshot", "price_to_book", "x"),
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
            single = len(r.get("portfolio", {})) == 1
            for label, key in (("historical", "historical_var_pct"), ("filtered historical simulation", "fhs_var_pct")):
                if key not in r:
                    continue
                row = table.setdefault(f"{r['horizon_days']}-day VaR {r['confidence_pct']:.0f}% ({label})", {})
                for ticker in r.get("portfolio", {}):
                    cell = f"{_fmt(r[key], '%')} [{ev['id']}]" + ("" if single else " (portfolio)")
                    if single or ticker not in row:  # a stock's own VaR beats a portfolio figure
                        row[ticker] = cell
    if not table:
        return ""
    lines = ["| Metric | " + " | ".join(tickers) + " |", "|---|" + "---|" * len(tickers)]
    for label, row in table.items():
        lines.append(f"| {label} | " + " | ".join(row.get(t, "–") for t in tickers) + " |")
    return "\n".join(lines)


def _data_status(ev: dict) -> str:
    sources = ev.get("sources") or []
    if not sources:
        return "–"
    oldest = min(s["fetched_at"] for s in sources)
    stale = [s for s in sources if s["status"] == "stale-cache"]
    when = oldest.replace("T", " ").replace("Z", " UTC")
    return f"⚠ stale cache from {when}" if stale else when


def evidence_table(evidence: list[dict]) -> str:
    lines = ["| ID | Agent | Tool call | Chosen by | Source | Data retrieved |", "|---|---|---|---|---|---|"]
    for ev in evidence:
        args = ", ".join(f"{k}={v}" for k, v in ev["args"].items())
        source = ev["result"].get("url") or _SOURCES.get(ev["tool"], "")
        if ev.get("error"):
            source = f"error: {ev['error']}"
        lines.append(f"| {ev['id']} | {ev['agent']} | `{ev['tool']}({args})` | {ev['chosen_by']} | {source} | {_data_status(ev)} |")
    return "\n".join(lines)


def run_stats(state: dict, settings, wall_s: float) -> dict:
    calls = state.get("llm_calls", [])
    evidence = state.get("evidence", [])
    by_model = sum(1 for e in evidence if str(e["chosen_by"]).startswith("model"))
    risk = next((e["result"] for e in evidence if e["tool"] == "value_at_risk" and not e.get("error")), {})
    retrieval = next((e["result"].get("retrieval") for e in evidence if e["tool"] == "search_10k" and not e.get("error")), None)
    sources = [s for e in evidence for s in e.get("sources") or []]
    return {
        "llm_calls": len(calls),
        "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
        "completion_tokens": sum(c["completion_tokens"] for c in calls),
        "model_seconds": round(sum(c["latency_s"] for c in calls), 1),
        "wall_seconds": round(wall_s, 1),
        "tool_calls": len(evidence),
        "tool_calls_chosen_by_model": by_model,
        "revisions": len(state.get("revisions", [])),
        "risk_backend": risk.get("compute_backend"),
        "retrieval": retrieval,
        "data_sources": len({s["url"] for s in sources}),
        "stale_sources": sorted({s["url"] for s in sources if s["status"] == "stale-cache"}),
        "dropped_params": sorted({p for c in calls for p in c.get("dropped_params", [])}),
        "ablations": sorted(settings.ablate),
        "endpoint": settings.base_url,
        "orchestrator_model": settings.orchestrator_model,
        "specialist_model": settings.specialist_model,
    }


def _annotate(text: str, evidence: list[dict], tickers: list[str], allowed: set[str] | None) -> str:
    """Mark untraced numbers, mismatched claims and promissory language in place.

    Both checks run on the original text and all markers are inserted in one pass,
    so a marker's own wording is never re-checked."""
    pool = evidence if allowed is None else [e for e in evidence if e["id"] in allowed]
    marks = [(n.end, " ⚠unverified") for n in provenance.check(text, pool).unverified]
    marks += [(c.end, f" ⚠ {c.issues[0]}") for c in attribution.check(text, evidence, tickers, allowed).flagged]
    for position, marker in sorted(marks, reverse=True):
        text = text[:position] + marker + text[position:]
    return flag_promissory(text)


def build_report(state: dict, settings, wall_s: float) -> tuple[str, dict]:
    tickers = state["tickers"]
    evidence = state.get("evidence", [])
    findings = [state["findings"][t["agent"]] for t in state.get("plan", []) if t["agent"] in state.get("findings", {})]
    narrative = state.get("narrative", "")
    quality = note_quality(findings, narrative, evidence, tickers)

    sections = [f"# Research note: {', '.join(tickers)}", f"**Question:** {state['question']}", ""]
    meta = (
        f"*Generated {datetime.now():%Y-%m-%d %H:%M} by FinAgent · {len(findings)} specialist agents · "
        f"models `{settings.orchestrator_model}` / `{settings.specialist_model}`*"
    )
    sections += [meta, "", f"> {DISCLAIMER}", ""]
    if narrative:
        note = "*(revised after fact-check)*\n\n" if "writer" in state.get("revisions", []) else ""
        sections.append(note + _annotate(narrative, evidence, tickers, None))
    sections += ["", "## Specialist findings"]
    for f in findings:
        note = " *(revised after fact-check)*" if f.get("revised") else ""
        own = {e["id"] for e in evidence if e["agent"] == f["agent"]}
        sections += [f"### {f['title']}{note}", f"*Focus: {f['focus']}*", "", _annotate(f["text"], evidence, tickers, own), ""]

    figures = key_figures(evidence, tickers)
    if figures:
        sections += ["## Key figures (generated from tool outputs, not by a model)", "", figures, ""]
    sections += ["## Evidence ledger", "", evidence_table(evidence), ""]

    stats = run_stats(state, settings, wall_s)
    stats["quality"] = quality
    stats["provenance"] = {
        "checked": quality["numbers_checked"],
        "verified": quality["numbers_traced"],
        "score": quality["number_provenance"],
        "unverified": [],
    }
    stats["promissory_phrases"] = state.get("promissory_phrases", [])
    untraced = quality["numbers_checked"] - quality["numbers_traced"]
    mismatched = quality["claims_checked"] - quality["claims_supported"]
    stale = stats["stale_sources"]
    sections += [
        "## How this note was produced",
        "",
        f"- **Number provenance:** {quality['numbers_traced']} of {quality['numbers_checked']} numbers trace back to a tool "
        f"result the agent received ({quality['number_provenance']:.0%})" + (f"; {untraced} flagged ⚠." if untraced else "."),
        f"- **Claim attribution:** {quality['claims_supported']} of {quality['claims_checked']} cited claims match their source "
        f"(company, period, metric, direction; 10-K points by wording) ({quality['claim_attribution']:.0%})"
        + (f"; {mismatched} flagged ⚠." if mismatched else "."),
        f"- **Agents:** planner → {', '.join(f['agent'] for f in findings)} → fact-check"
        f"{' → ' + str(stats['revisions']) + ' revision(s)' if stats['revisions'] else ''} → lead analyst.",
        f"- **Tool calls:** {stats['tool_calls']} ("
        + ("all chosen by the models" if stats["tool_calls_chosen_by_model"] == stats["tool_calls"]
           else f"{stats['tool_calls_chosen_by_model']} chosen by the models, {stats['tool_calls'] - stats['tool_calls_chosen_by_model']} added by coverage checks")
        + ").",
        f"- **Data:** {stats['data_sources']} sources, "
        + (f"⚠ {len(stale)} served from a stale cache because the provider was unreachable." if stale else "all fetched or cached within their freshness window.")
        + " Retrieval times are in the evidence ledger.",
        f"- **Language model:** {stats['llm_calls']} calls, {stats['prompt_tokens']:,} prompt + {stats['completion_tokens']:,} "
        f"completion tokens, {stats['model_seconds']} s of model time; {stats['wall_seconds']} s end to end.",
        f"- **Compute:** risk simulation on {stats['risk_backend'] or 'n/a'}; 10-K retrieval: {stats['retrieval'] or 'n/a'}; endpoint `{settings.base_url}`.",
    ]
    if stats["dropped_params"]:
        sections.append(f"- **Endpoint limits:** the server rejected {', '.join(stats['dropped_params'])}; requests were retried without them.")
    if stats["ablations"]:
        sections.append(f"- **Ablation run:** switched off {', '.join(stats['ablations'])}.")
    sections.append("")
    return "\n".join(sections), stats
