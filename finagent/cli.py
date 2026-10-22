"""Command line: ``finagent analyze``, ``finagent doctor``, ``finagent bench-risk``, ``finagent tools``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.table import Table

from .config import PROFILES, load_settings
from .runs import save_run

console = Console()

_AGENT_STYLE = {
    "market": "cyan",
    "fundamentals": "green",
    "risk": "magenta",
    "filings": "yellow",
    "writer": "bold white",
    "planner": "blue",
}


def _args_text(args: dict) -> str:
    return escape(", ".join(f"{k}={v}" for k, v in args.items()))


def print_event(event: dict) -> None:
    """Render one workflow event as a single trace line."""
    stamp = datetime.fromtimestamp(event.get("t", time.time())).strftime("%H:%M:%S")
    kind = event["event"]
    agent = event.get("agent", "")
    tag = f"[{_AGENT_STYLE.get(agent, 'white')}]{agent or kind:<12}[/]"
    if kind == "guard":
        status = "[green]allowed[/]" if event["allowed"] else f"[red]refused: {event['reason']}[/]"
        extra = " · personal-advice framing on" if event.get("personal_advice") else ""
        console.print(f"{stamp}  {'guardrail':<12} {status} ({event['rail']}){extra}")
    elif kind == "plan":
        team = " · ".join(t["agent"] for t in event["tasks"])
        console.print(f"{stamp}  [blue]{'planner':<12}[/] → {team}  [dim]({event['source']})[/]")
        for t in event["tasks"]:
            console.print(f"{'':10}{'':13}[dim]{t['agent']}: {escape(t['focus'])}[/]")
    elif kind == "agent_start":
        console.print(f"{stamp}  {tag} ▶ {event['title']}")
    elif kind == "llm":
        calls = ", ".join(event["tool_calls"]) if event["tool_calls"] else "text"
        label = "revision" if event.get("revision") else "model"
        console.print(
            f"{stamp}  {tag}   {label} {event['latency_s']:>5.1f}s  "
            f"[dim]{event['prompt_tokens']}→{event['completion_tokens']} tok[/]  → {calls}"
        )
    elif kind == "tool":
        flag = "" if event["chosen_by"].startswith("model") else f" [dim]({event['chosen_by']})[/]"
        err = f" [red]{escape(event['error'])}[/]" if event.get("error") else ""
        stale = " [yellow]⚠ stale cached data[/]" if event.get("stale") else ""
        console.print(f"{stamp}  {tag}   [bold]{event['id']:<3}[/] {event['tool']}({_args_text(event['args'])}) {event['seconds']}s{flag}{err}{stale}")
    elif kind == "agent_done":
        console.print(f"{stamp}  {tag} ✓ done in {event['seconds']}s")
    elif kind == "verify":
        when = " after revision" if event.get("after_revision") else ""
        bad = f" · [red]untraced: {escape(', '.join(event['unverified']))}[/]" if event["unverified"] else ""
        claims = ""
        if event.get("claims_checked"):
            colour = "green" if event["claims_supported"] == event["claims_checked"] else "red"
            claims = f" · [{colour}]{event['claims_supported']}/{event['claims_checked']} claims match their sources[/]"
        console.print(f"{stamp}  {'fact-check':<12} {agent}{when}: {event['verified']}/{event['checked']} numbers traced{claims}{bad}")
        for issue in event.get("claim_issues", [])[:4]:
            console.print(f"{'':10}{'':13}[dim red]{escape(issue)}[/]")
    elif kind == "revise":
        console.print(f"{stamp}  {tag} ↺ revising ({len(event.get('problems', []))} problem(s))")
    elif kind == "nudge":
        console.print(f"{stamp}  {tag}   [yellow]answered without data → discarded, asked to call tools[/]")
    elif kind == "done":
        console.print(f"{stamp}  {'complete':<12} [green]report assembled[/]")


def cmd_analyze(args) -> int:
    from .graph import run_research
    from .llm import OpenAICompatLLM
    from .tools import ToolContext, parse_tickers

    settings = load_settings(args.profile)
    tickers = parse_tickers(args.tickers)
    console.rule(f"FinAgent · {', '.join(tickers)}")
    console.print(f"[dim]profile={settings.profile} endpoint={settings.base_url}\n"
                  f"orchestrator={settings.orchestrator_model} specialists={settings.specialist_model}[/]")
    events: list[dict] = []

    def on_event(event):
        events.append(event)
        print_event(event)

    started = time.time()
    final = run_research(OpenAICompatLLM(settings), ToolContext(settings), tickers, args.question, on_event=on_event)
    console.rule(f"finished in {time.time() - started:.0f}s")

    path = save_run(final, events, settings, args.out_dir)

    if not args.quiet:
        console.print(Markdown(final.get("report", "")))
    console.print(f"\n[bold]Saved[/] {path.with_suffix('.md')} (+ .json run record for the web UI)")
    return 0


def cmd_doctor(args) -> int:
    from .accel import array_backend
    from .agents.planner import make_plan
    from .data.prices import adjusted_close, risk_free_rate
    from .data.sec import ticker_to_cik
    from .llm import OpenAICompatLLM

    settings = load_settings(args.profile)
    table = Table(title=f"FinAgent doctor · profile '{settings.profile}'", show_lines=False)
    table.add_column("Check")
    table.add_column("Result")
    ok = True

    def row(name, passed, detail):
        nonlocal ok
        ok &= bool(passed)
        table.add_row(name, f"[{'green' if passed else 'red'}]{'✓' if passed else '✗'}[/] {escape(str(detail))}")

    table.add_row("Endpoint", settings.base_url)
    table.add_row("Models", f"orchestrator {settings.orchestrator_model} · specialists {settings.specialist_model}")
    llm = OpenAICompatLLM(settings)
    style = llm.reasoning_style
    table.add_row("Reasoning control", f"{style} ({'set in profile' if settings.reasoning_style != 'auto' else 'detected'})")
    try:
        listed = llm.list_models()
        # Ollama reports untagged models as "name:latest"
        missing = {m for m in {settings.orchestrator_model, settings.specialist_model} if m not in listed and f"{m}:latest" not in listed}
        row("Model list", not missing or not listed, f"{len(listed)} served" + (f"; missing {', '.join(missing)}" if missing else ""))
    except Exception as err:
        row("Model list", False, f"{type(err).__name__}: {err}"[:120])

    try:
        r = llm.chat([{"role": "user", "content": "Reply with the single word: ready"}], role="specialist", max_tokens=20)
        row("Chat", bool(r.content), f"{r.content[:30]!r} in {r.usage['latency_s']}s")
    except Exception as err:
        row("Chat", False, f"{type(err).__name__}: {err}"[:160])

    tool = {"type": "function", "function": {"name": "price_performance", "description": "Stock price performance.",
            "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}}}
    try:
        r = llm.chat([{"role": "user", "content": "How has NVDA stock performed? Use the tool."}], role="specialist", tools=[tool], max_tokens=200)
        call = r.tool_calls[0] if r.tool_calls else None
        good = bool(call and call.name == "price_performance" and str(call.arguments.get("ticker", "")).upper() == "NVDA")
        how = "native" if call and call.native else "parsed from text"
        row("Tool calling", good, f"{call.name}({call.arguments}) [{how}] in {r.usage['latency_s']}s" if call else "no tool call returned")
    except Exception as err:
        row("Tool calling", False, f"{type(err).__name__}: {err}"[:160])

    try:
        tasks, usage, source = make_plan(llm, ["AMD"], "Is AMD's valuation justified by its growth?")
        latency = usage[0]["latency_s"] if usage else 0
        row("Planner (JSON schema)", source == "model", f"{len(tasks)} tasks from the {source} in {latency}s: {', '.join(t['agent'] for t in tasks)}")
    except Exception as err:
        row("Planner (JSON schema)", False, f"{type(err).__name__}: {err}"[:160])

    if settings.embed_model:
        try:
            from .retrieval import Chunk, FilingIndex

            index = FilingIndex([Chunk("a#0", "a", "Export controls may reduce demand."), Chunk("b#0", "b", "Revenue grew.")], settings, "doctor")
            row("Embeddings", index.mode.startswith("BM25 +"), index.mode)
        except Exception as err:
            row("Embeddings", False, f"{type(err).__name__}: {err}"[:160])
    if settings.guard_model:
        from .agents.guard import check_input

        decision = check_input(settings, "What are NVIDIA's main risks?")
        row("Topic rail", decision["allowed"] and "unavailable" not in decision["rail"], decision["rail"])
    if llm.unsupported_params:
        table.add_row("Endpoint limits", f"rejected {', '.join(sorted(llm.unsupported_params))}; runs retry without them")

    for name, probe in (
        ("SEC EDGAR", lambda: f"NVDA → CIK {ticker_to_cik(settings, 'NVDA')[0]}"),
        ("Prices", lambda: f"SPY {len(adjusted_close(settings, 'SPY', '1y'))} daily bars"),
        ("Risk-free rate", lambda: f"{risk_free_rate(settings) * 100:.2f}% (13-week T-bill)"),
    ):
        try:
            row(name, True, probe())
        except Exception as err:
            row(name, False, f"{type(err).__name__}: {err}"[:120])
    table.add_row("Risk engine", array_backend()[1])
    console.print(table)
    return 0 if ok else 1


def cmd_bench(args) -> int:
    import numpy as np

    from .accel import array_backend
    from .analytics import filtered_historical_simulation

    rng = np.random.default_rng(0)
    returns = rng.standard_t(4, size=(1260, args.assets)).astype(np.float32) * 0.015
    weights = np.full(args.assets, 1 / args.assets)
    backends = [(np, "CPU (NumPy)")]
    xp, label = array_backend()
    if xp is not np:
        backends.append((xp, label))
    table = Table(title=f"Filtered historical simulation VaR · {args.paths:,} paths × {args.horizon} days × {args.assets} assets")
    for col in ("Backend", "Run time", "Paths / second", "VaR 95%"):
        table.add_column(col)
    for module, name in backends:
        filtered_historical_simulation(returns, weights, n_paths=1000, horizon_days=args.horizon, xp=module, backend=name)  # warm-up
        res = filtered_historical_simulation(returns, weights, n_paths=args.paths, horizon_days=args.horizon, xp=module, backend=name)
        table.add_row(name, f"{res.runtime_ms:,.0f} ms", f"{args.paths / res.runtime_ms * 1000:,.0f}", f"{res.var:.2%}")
    console.print(table)
    if len(backends) == 1:
        console.print("[dim]No NVIDIA GPU detected - install finagent\\[gpu] on a CUDA machine to add a CuPy row.[/]")
    return 0


def cmd_eval(args) -> int:
    from .evaluate import (
        QUESTIONS,
        TICKER_SETS,
        load_results,
        run_eval,
        summary_markdown,
        variants,
    )
    from .tools import parse_tickers

    if args.summarize:
        console.print(Markdown(summary_markdown(load_results(args.summarize))))
        return 0
    if args.rescore:
        from .evaluate import rescore

        rows = rescore(args.rescore)
        if args.out:
            Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        console.print(Markdown(summary_markdown(rows, title="FinAgent evaluation (re-scored)")))
        return 0
    settings = load_settings(args.profile)
    tickers = parse_tickers(args.tickers) if args.tickers else TICKER_SETS[args.set]
    questions = [q.strip() for q in args.questions.split(",")]
    chosen = variants([v.strip() for v in args.variants.split(",")])
    unknown = [q for q in questions if q not in QUESTIONS]
    if unknown:
        raise SystemExit(f"unknown question key(s) {unknown}; choose from {sorted(QUESTIONS)}")
    out = Path(args.out or f"eval/results/{datetime.now():%Y%m%d-%H%M}-{settings.profile}.jsonl")
    total = len(tickers) * len(questions) * len(chosen)
    console.rule(f"FinAgent eval · {total} runs")
    console.print(f"[dim]tickers={', '.join(tickers)}\nquestions={', '.join(questions)} variants={', '.join(chosen)}\n"
                  f"model={settings.specialist_model} endpoint={settings.base_url}\nresults → {out}[/]")

    def progress(row):
        if row.get("ok"):
            console.print(
                f"{row['ticker']:<6} {row['variant']:<14} {row['wall_seconds']:>6.0f}s  numbers {row['number_provenance']:.0%}  "
                f"claims {row['draft_claim_attribution']:.0%}→{row['claim_attribution']:.0%}  facets {row['facet_coverage']:.0%}  "
                f"tools by model {row['tool_calls_by_model']}/{row['tool_calls']}"
            )
        else:
            console.print(f"{row['ticker']:<6} {row['variant']:<14} [red]failed: {escape(str(row.get('error')))}[/]")

    results = run_eval(settings, tickers, questions, list(chosen), out, on_progress=progress, records_dir=args.records)
    md = summary_markdown(results, title=f"FinAgent evaluation · {settings.specialist_model}")
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    console.print(Markdown(md))
    console.print(f"[bold]Saved[/] {out} and {out.with_suffix('.md')}")
    return 0


def cmd_tools(args) -> int:
    from .agents.specialists import _ALL_TOOLS, SPECIALISTS

    table = Table(title="Agents and their tools")
    table.add_column("Agent")
    table.add_column("Tool")
    table.add_column("What it does")
    for spec in SPECIALISTS.values():
        for name in spec.tools:
            table.add_row(spec.title, name, _ALL_TOOLS[name].description)
    console.print(table)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finagent", description="Grounded multi-agent equity research on small NVIDIA Nemotron models.")
    sub = parser.add_subparsers(dest="command", required=True)

    profiles = sorted(PROFILES)
    p = sub.add_parser("analyze", help="research one or more tickers")
    p.add_argument("tickers", nargs="+")
    p.add_argument("-q", "--question", default="What are the investment case and the main risks?")
    p.add_argument("--profile", choices=profiles, default=None)
    p.add_argument("--out-dir", default="reports")
    p.add_argument("--quiet", action="store_true", help="do not print the report")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("doctor", help="check endpoint, model capabilities and data sources")
    p.add_argument("--profile", choices=profiles, default=None)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("eval", help="run the evaluation harness (tickers x questions x ablations)")
    p.add_argument("--set", choices=["pilot", "broad"], default="pilot", help="built-in ticker set")
    p.add_argument("--tickers", nargs="*", help="override the ticker set")
    p.add_argument("--questions", default="valuation", help="comma-separated: valuation,risk")
    p.add_argument("--variants", default="full", help="comma-separated: full,-facts,-phases,-coverage,-revision,-own_evidence,-attribution or all")
    p.add_argument("--profile", choices=profiles, default=None)
    p.add_argument("--out", help="results file (.jsonl); a .md summary is written next to it")
    p.add_argument("--records", metavar="DIR", help="where to save each run's full record (default: <out>-runs next to the results)")
    p.add_argument("--summarize", metavar="JSONL", help="only print the summary of an existing results file")
    p.add_argument("--rescore", metavar="RUNS_DIR", help="recompute metrics from saved run records with the current checks")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("bench-risk", help="time the risk simulation engine")
    p.add_argument("--paths", type=int, default=1_000_000)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--assets", type=int, default=5)
    p.set_defaults(func=cmd_bench)

    sub.add_parser("tools", help="list agents and tools").set_defaults(func=cmd_tools)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
