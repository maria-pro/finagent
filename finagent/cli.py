"""Command line: ``finagent analyze``, ``finagent doctor``, ``finagent bench-risk``, ``finagent tools``."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.table import Table

from .config import load_settings
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
        console.print(f"{stamp}  {tag}   [bold]{event['id']:<3}[/] {event['tool']}({_args_text(event['args'])}) {event['seconds']}s{flag}{err}")
    elif kind == "agent_done":
        console.print(f"{stamp}  {tag} ✓ done in {event['seconds']}s")
    elif kind == "verify":
        when = " after revision" if event.get("after_revision") else ""
        bad = f" · [red]untraced: {escape(', '.join(event['unverified']))}[/]" if event["unverified"] else ""
        console.print(f"{stamp}  {'fact-check':<12} {agent}{when}: {event['verified']}/{event['checked']} numbers traced{bad}")
    elif kind == "revise":
        console.print(f"{stamp}  {tag} ↺ revising")
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
    try:
        listed = [m.id for m in llm._client.models.list().data]
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
    from .analytics import monte_carlo_var

    rng = np.random.default_rng(0)
    returns = rng.standard_t(4, size=(1260, args.assets)).astype(np.float32) * 0.015
    weights = np.full(args.assets, 1 / args.assets)
    backends = [(np, "CPU (NumPy)")]
    xp, label = array_backend()
    if xp is not np:
        backends.append((xp, label))
    table = Table(title=f"Monte Carlo VaR · {args.paths:,} paths × {args.horizon} days × {args.assets} assets")
    for col in ("Backend", "Run time", "Paths / second", "VaR 95%"):
        table.add_column(col)
    for module, name in backends:
        monte_carlo_var(returns, weights, n_paths=1000, horizon_days=args.horizon, xp=module, backend=name)  # warm-up
        res = monte_carlo_var(returns, weights, n_paths=args.paths, horizon_days=args.horizon, xp=module, backend=name)
        table.add_row(name, f"{res.runtime_ms:,.0f} ms", f"{args.paths / res.runtime_ms * 1000:,.0f}", f"{res.var:.2%}")
    console.print(table)
    if len(backends) == 1:
        console.print("[dim]No NVIDIA GPU detected - install finagent\\[gpu] on a CUDA machine to add a CuPy row.[/]")
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

    p = sub.add_parser("analyze", help="research one or more tickers")
    p.add_argument("tickers", nargs="+")
    p.add_argument("-q", "--question", default="What are the investment case and the main risks?")
    p.add_argument("--profile", choices=["local", "nim"], default=None)
    p.add_argument("--out-dir", default="reports")
    p.add_argument("--quiet", action="store_true", help="do not print the report")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("doctor", help="check endpoint, model capabilities and data sources")
    p.add_argument("--profile", choices=["local", "nim"], default=None)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("bench-risk", help="time the Monte Carlo risk engine")
    p.add_argument("--paths", type=int, default=1_000_000)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--assets", type=int, default=5)
    p.set_defaults(func=cmd_bench)

    sub.add_parser("tools", help="list agents and tools").set_defaults(func=cmd_tools)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
