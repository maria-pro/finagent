"""Evaluation harness: many tickers, ablations, one yardstick.

Each run is scored the same way whatever was switched on while it was produced:

* **number provenance** - share of numbers in the note that trace to a tool result
  the writing agent actually received;
* **claim attribution** - share of cited claims that keep their source's company,
  period, metric and direction (10-K points: by wording);
* **facet coverage** - whether the note's opening addresses each part of the
  question (valuation, fundamentals, risk, performance, filings, comparison) with
  cited evidence of the right kind - a transparent proxy for "does it answer the
  question", not a judgement of the reasoning;
* autonomy (tool calls chosen by the model vs added by coverage checks), revisions,
  cost (model calls, tokens, seconds) and data freshness.

Pre-revision drafts are scored too, so the effect of the revision pass is measured
from the same runs. Results stream to a JSON-lines file so a long study survives
interruption, and ``summarize`` turns them into a table per variant.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path
from typing import Callable

from .checks import note_quality
from .config import ABLATIONS, Settings

TICKER_SETS: dict[str, list[str]] = {
    # Mix of sectors, two banks and a broker-dealer (no gross margin), two multi-class
    # issuers (GOOGL; BRK-B reports in class-A equivalents).
    "pilot": ["NVDA", "AMD", "AAPL", "MSFT", "GOOGL", "META", "JPM", "BAC", "GS", "BRK-B", "XOM", "KO"],
}
TICKER_SETS["broad"] = TICKER_SETS["pilot"] + [
    "AVGO", "INTC", "ORCL", "CRM", "ADBE", "AMZN", "TSLA", "NFLX", "WFC", "C", "MS", "V", "MA", "AXP",
    "WMT", "COST", "PG", "JNJ", "PFE", "UNH", "CVX", "CAT", "BA", "GE", "DIS", "MCD", "NKE", "FOXA",
]

QUESTIONS = {
    "valuation": "Is {ticker}'s valuation justified by its fundamentals, and what are the main risks?",
    "risk": "How risky is {ticker} for a long-term investor, and what does its 10-K flag?",
}

# question pattern -> tools whose evidence addresses that facet (None: checked by text)
FACETS: dict[str, tuple[str, set[str] | None]] = {
    "valuation": (r"valu|expensive|cheap|priced|multiple|p/e", {"valuation_snapshot"}),
    "fundamentals": (r"fundamental|growth|revenue|margin|profit|earning", {"financial_statements"}),
    "risk": (r"risk|downside|drawdown|volatil|crash", {"value_at_risk", "stress_test", "search_10k", "price_performance"}),
    "performance": (r"perform|return|stock price", {"price_performance", "benchmark_comparison"}),
    "filings": (r"10-k|filing|annual report|\bflag", {"search_10k"}),
    "comparison": (r"compar|versus|\bvs\b|relative to", None),
}

_ID_RE = re.compile(r"\b([A-Z]\d+)\b")


def variants(names: list[str] | None = None) -> dict[str, frozenset[str]]:
    """``full`` plus one variant per ablation (``-facts`` ...); ``all`` expands to every one."""
    every = {"full": frozenset(), **{f"-{a}": frozenset({a}) for a in ABLATIONS}}
    if not names or names == ["all"]:
        return every
    unknown = [n for n in names if n not in every]
    if unknown:
        raise ValueError(f"unknown variant(s) {unknown}; choose from {sorted(every)}")
    return {n: every[n] for n in names}


def facet_coverage(question: str, tickers: list[str], narrative: str, evidence: list[dict]) -> dict:
    by_id = {e["id"]: e for e in evidence}
    cited = {by_id[i]["tool"] for block in re.findall(r"\[([^\]]+)\]", narrative) for i in _ID_RE.findall(block) if i in by_id}
    answer = re.search(r"##\s*Answer\s*\n(.*?)(?:\n##|\Z)", narrative, re.S)
    answer_text = answer.group(1).strip() if answer else ""
    facets = [name for name, (pattern, _) in FACETS.items() if re.search(pattern, question, re.I)]
    addressed = []
    for name in facets:
        tools = FACETS[name][1]
        if tools is None:
            ok = all(re.search(rf"(?<![A-Za-z]){re.escape(t)}(?![A-Za-z])", answer_text) for t in tickers)
        else:
            ok = bool(cited & tools)
        if ok:
            addressed.append(name)
    return {
        "facets": facets,
        "addressed": addressed,
        "score": round(len(addressed) / len(facets), 3) if facets else 1.0,
        "answer_present": bool(answer_text),
    }


def summarize_run(final: dict, events: list[dict], *, ticker: str, question_key: str, variant: str) -> dict:
    stats = final.get("stats") or {}
    evidence = final.get("evidence", [])
    plan = final.get("plan", [])
    findings = [final["findings"][t["agent"]] for t in plan if t["agent"] in final.get("findings", {})]
    tickers = final.get("tickers", [ticker])
    quality = note_quality(findings, final.get("narrative", ""), evidence, tickers)
    drafts = [dict(f, text=f.get("draft", f["text"])) for f in findings]
    draft_quality = note_quality(drafts, final.get("narrative_draft", final.get("narrative", "")), evidence, tickers)
    facets = facet_coverage(final.get("question", ""), tickers, final.get("narrative", ""), evidence)
    by_model = sum(1 for e in evidence if str(e["chosen_by"]).startswith("model"))
    return {
        "ticker": ticker,
        "question": question_key,
        "variant": variant,
        "ok": bool(final.get("report")) and not stats.get("refused"),
        "wall_seconds": stats.get("wall_seconds"),
        "llm_calls": stats.get("llm_calls"),
        "tokens": (stats.get("prompt_tokens") or 0) + (stats.get("completion_tokens") or 0),
        "tool_calls": len(evidence),
        "tool_calls_by_model": by_model,
        "tool_errors": sum(1 for e in evidence if e.get("error")),
        "nudges": sum(1 for e in events if e.get("event") == "nudge"),
        "revisions": len(final.get("revisions", [])),
        "plan": [t["agent"] for t in plan],
        "plan_source": final.get("plan_source"),
        "number_provenance": quality["number_provenance"],
        "numbers_checked": quality["numbers_checked"],
        "claim_attribution": quality["claim_attribution"],
        "claims_checked": quality["claims_checked"],
        "claim_issues": quality["claim_issues"],
        "draft_number_provenance": draft_quality["number_provenance"],
        "draft_claim_attribution": draft_quality["claim_attribution"],
        "facet_coverage": facets["score"],
        "facets_missed": sorted(set(facets["facets"]) - set(facets["addressed"])),
        "answer_present": facets["answer_present"],
        "stale_sources": len(stats.get("stale_sources", [])),
    }


def run_eval(
    settings: Settings,
    tickers: list[str],
    question_keys: list[str],
    variant_names: list[str],
    out_path: str | Path,
    llm_factory: Callable | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Run every (ticker, question, variant) combination, appending one JSON line per run."""
    from .graph import run_research
    from .llm import OpenAICompatLLM
    from .tools import ToolContext

    factory = llm_factory or OpenAICompatLLM
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    chosen = variants(variant_names)
    results = []
    for ticker in tickers:
        for qkey in question_keys:
            question = QUESTIONS[qkey].format(ticker=ticker)
            for name, ablate in chosen.items():
                run_settings = settings.with_overrides(ablate=ablate)
                events: list[dict] = []
                started = time.time()
                try:
                    final = run_research(factory(run_settings), ToolContext(run_settings), [ticker], question, on_event=events.append)
                    row = summarize_run(final, events, ticker=ticker, question_key=qkey, variant=name)
                except Exception as err:  # one failed run must not end a long study
                    row = {"ticker": ticker, "question": qkey, "variant": name, "ok": False, "error": f"{type(err).__name__}: {err}"}
                row["elapsed"] = round(time.time() - started, 1)
                row["model"] = run_settings.specialist_model
                row["endpoint"] = run_settings.base_url
                with out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                results.append(row)
                if on_progress:
                    on_progress(row)
    return results


def load_results(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.fmean(values), 3) if values else None


def summarize(results: list[dict]) -> dict[str, dict]:
    """Aggregate per variant. Means are over successful runs; n counts all attempts."""
    table: dict[str, dict] = {}
    for name in dict.fromkeys(r["variant"] for r in results):
        rows = [r for r in results if r["variant"] == name]
        ok = [r for r in rows if r.get("ok")]
        tool_calls = sum(r["tool_calls"] for r in ok)
        table[name] = {
            "runs": len(rows),
            "succeeded": len(ok),
            "median_seconds": round(statistics.median(r["wall_seconds"] for r in ok), 1) if ok else None,
            "number_provenance": _mean([r["number_provenance"] for r in ok]),
            "claim_attribution": _mean([r["claim_attribution"] for r in ok]),
            "draft_claim_attribution": _mean([r["draft_claim_attribution"] for r in ok]),
            "facet_coverage": _mean([r["facet_coverage"] for r in ok]),
            "tool_calls_by_model": round(sum(r["tool_calls_by_model"] for r in ok) / tool_calls, 3) if tool_calls else None,
            "nudges_per_run": _mean([r["nudges"] for r in ok]),
            "revisions_per_run": _mean([r["revisions"] for r in ok]),
            "tokens_per_run": _mean([r["tokens"] for r in ok]),
        }
    return table


def summary_markdown(results: list[dict], title: str = "FinAgent evaluation") -> str:
    table = summarize(results)
    tickers = sorted({r["ticker"] for r in results})
    model = next((r.get("model") for r in results if r.get("model")), "?")
    lines = [
        f"# {title}",
        "",
        f"{len(results)} runs · {len(tickers)} tickers ({', '.join(tickers)}) · model `{model}`",
        "",
        "| Variant | Runs (ok) | Median s | Number provenance | Claim attribution (draft → final) | Facet coverage | Tool calls by model | Nudges/run | Revisions/run | Tokens/run |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def pct(v):
        return "–" if v is None else f"{v:.0%}"

    for name, s in table.items():
        lines.append(
            f"| {name} | {s['runs']} ({s['succeeded']}) | {s['median_seconds']} | {pct(s['number_provenance'])} | "
            f"{pct(s['draft_claim_attribution'])} → {pct(s['claim_attribution'])} | {pct(s['facet_coverage'])} | "
            f"{pct(s['tool_calls_by_model'])} | {s['nudges_per_run']} | {s['revisions_per_run']} | {s['tokens_per_run']:,.0f} |"
            if s["tokens_per_run"] is not None
            else f"| {name} | {s['runs']} ({s['succeeded']}) | – | – | – | – | – | – | – | – |"
        )
    failures = [r for r in results if not r.get("ok")]
    if failures:
        lines += ["", "Failed runs: " + "; ".join(f"{r['ticker']}/{r['variant']}: {r.get('error', 'no report')}" for r in failures)]
    issues = [(r["ticker"], r["variant"], i) for r in results if r.get("ok") for i in r.get("claim_issues", [])]
    if issues:
        lines += ["", "<details><summary>Claim issues left in final notes</summary>", ""]
        lines += [f"- {t} ({v}): {i}" for t, v, i in issues[:60]]
        lines += ["", "</details>"]
    return "\n".join(lines) + "\n"
