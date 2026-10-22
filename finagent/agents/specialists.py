"""Specialist sub-agents: small, focused tool-calling loops.

Each specialist owns two or three tools and a short prompt. Narrow scope is
what lets a 4B model call tools reliably; the coverage check below guarantees
that a report never rests on an agent that skipped its data.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable

from ..llm import LLM, ToolCall, tool_call_message, usage_record
from ..tools import (
    Tool,
    ToolContext,
    calculator,
    filings,
    fundamentals,
    market,
    parse_tickers,
    risk,
)

Emit = Callable[[dict], None]

_ALL_TOOLS: dict[str, Tool] = {
    t.name: t for module in (market, fundamentals, risk, filings, calculator) for t in module.TOOLS
}

_FINDINGS_RULES = """Now write your findings for the research team.
- At most 5 bullet points. Each bullet copies one fact from the tool results word for word and ends with its evidence id in square brackets:
  - <fact copied from a tool result> [<evidence id>]
- Pick the facts that matter most for your focus.
- End with one sentence of interpretation that contains no numbers."""


@dataclass(frozen=True)
class Specialist:
    key: str
    title: str
    prefix: str  # evidence ids: M1, M2... for market
    job: str
    tools: tuple[str, ...]
    findings_rules: str = _FINDINGS_RULES

    def system_prompt(self) -> str:
        # Output-format rules are deliberately absent here: shown before any data, they
        # tempt a small model to skip its tools and write a plausible-looking answer.
        return (
            f"You are the {self.title} on an equity research team. Your job: {self.job}\n"
            "You have no data until you call your tools, so start by calling the tools you need; you can call several at once. "
            "Never guess numbers."
        )

    def required_calls(self, tickers: list[str], question: str) -> list[tuple[str, dict]]:
        """Tool calls a complete answer needs; run by the system if the model skips them."""
        if self.key == "market":
            return [(n, {"ticker": t}) for t in tickers for n in ("price_performance", "benchmark_comparison")]
        if self.key == "fundamentals":
            return [(n, {"ticker": t}) for t in tickers for n in ("financial_statements", "valuation_snapshot")]
        if self.key == "risk":
            groups = [tickers] if len(tickers) == 1 or "portfolio" in question.lower() else [[t] for t in tickers]
            return [(n, {"tickers": g}) for g in groups for n in ("value_at_risk", "stress_test")]
        if self.key == "filings":
            return [("search_10k", {"ticker": t, "query": question}) for t in tickers]
        return []


SPECIALISTS: dict[str, Specialist] = {
    s.key: s
    for s in (
        Specialist(
            "market",
            "Market Analyst",
            "M",
            "assess price performance, volatility, drawdowns and behaviour versus the S&P 500 (SPY).",
            ("price_performance", "benchmark_comparison"),
        ),
        Specialist(
            "fundamentals",
            "Fundamentals Analyst",
            "F",
            "assess growth, profitability, cash generation and valuation using SEC filings.",
            ("financial_statements", "valuation_snapshot", "calculator"),
        ),
        Specialist(
            "risk",
            "Risk Analyst",
            "R",
            "quantify downside risk with Value at Risk, expected shortfall and historical stress tests. "
            "When comparing companies, run each tool once per ticker; combine tickers only for a portfolio question.",
            ("value_at_risk", "stress_test"),
        ),
        Specialist(
            "filings",
            "Filings Analyst",
            "D",
            "find what the company's latest 10-K says that matters for the question.",
            ("search_10k",),
            findings_rules="""Now write your findings for the research team.
- At most 4 bullet points summarising what the passages say, each ending with the evidence id in square brackets:
  - <one risk or business point from the passages> [<evidence id>]
- Only quote numbers that appear in the passages.""",
        ),
    )
}


def model_view(ev_id: str, result: dict) -> dict:
    """What the model reads from a tool result: the pre-labelled facts when the tool
    provides them (full results stay in the evidence ledger). Labelled facts stop a
    small model from attaching a number to the wrong period or metric, and cost
    fewer prompt tokens than raw JSON."""
    if "facts" in result:
        return {"evidence_id": ev_id, "facts": result["facts"]}
    return {"evidence_id": ev_id, **result}


def _covered(call: tuple[str, dict], evidence: list[dict]) -> bool:
    name, args = call
    for ev in evidence:
        if ev["tool"] != name or ev.get("error"):
            continue
        if "ticker" in args and parse_tickers(ev["args"].get("ticker", ""))[:1] != [args["ticker"]]:
            continue
        if "tickers" in args and set(parse_tickers(ev["args"].get("tickers", []))) != set(args["tickers"]):
            continue
        return True
    return False


def run_specialist(
    llm: LLM,
    ctx: ToolContext,
    spec: Specialist,
    *,
    tickers: list[str],
    question: str,
    focus: str,
    emit: Emit,
    max_rounds: int = 3,
) -> dict:
    """Run one specialist to completion. Returns its finding, evidence and LLM usage."""
    tools = [_ALL_TOOLS[name] for name in spec.tools]
    specs = [t.spec() for t in tools]
    messages = [
        {"role": "system", "content": spec.system_prompt()},
        {"role": "user", "content": f"Tickers: {', '.join(tickers)}\nResearch question: {question}\nYour focus: {focus}"},
    ]
    evidence: list[dict] = []
    usage: list[dict] = []
    started = time.time()
    emit({"event": "agent_start", "agent": spec.key, "title": spec.title, "focus": focus})

    def execute(call: ToolCall, chosen_by: str) -> None:
        ev_id = f"{spec.prefix}{len(evidence) + 1}"
        tool = _ALL_TOOLS.get(call.name)
        t0 = time.perf_counter()
        error = None
        if tool is None or call.name not in spec.tools:
            result, error = {"error": f"unknown tool {call.name}; available: {', '.join(spec.tools)}"}, "unknown tool"
        else:
            try:
                result = tool(ctx, call.arguments)
            except Exception as err:  # tool failures go back to the model so it can adapt
                error = f"{type(err).__name__}: {err}"
                result = {"error": error}
        seconds = round(time.perf_counter() - t0, 2)
        record = {
            "id": ev_id,
            "agent": spec.key,
            "tool": call.name,
            "args": call.arguments,
            "result": result,
            "seconds": seconds,
            "chosen_by": chosen_by,
            "error": error,
        }
        evidence.append(record)
        emit({"event": "tool", **{k: record[k] for k in ("agent", "id", "tool", "args", "seconds", "chosen_by", "error")}})
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(model_view(ev_id, result), separators=(",", ":"))})

    def ask(with_tools: bool):
        response = llm.chat(messages, role="specialist", tools=specs if with_tools else None, max_tokens=700)
        usage.append(usage_record(response, spec.key))
        emit(
            {
                "event": "llm",
                "agent": spec.key,
                "tool_calls": [c.name for c in response.tool_calls],
                **{k: response.usage[k] for k in ("latency_s", "prompt_tokens", "completion_tokens")},
            }
        )
        return response

    text = ""
    rules_shown = nudged = False
    for _ in range(max_rounds):
        response = ask(with_tools=True)
        if response.tool_calls:
            messages.append(tool_call_message(response.content, response.tool_calls))
            for call in response.tool_calls:
                execute(call, "model" if call.native else "model (text-parsed)")
            if not rules_shown:
                unused = [n for n in spec.tools if n != "calculator" and n not in {e["tool"] for e in evidence}]
                hint = f"Tools you have not used yet: {', '.join(unused)}. Call any that would help your focus now. " if unused else ""
                messages.append({"role": "user", "content": hint + "When you have the data you need:\n" + spec.findings_rules})
                rules_shown = True
            continue
        if not evidence and not nudged:
            # Answered without any data: whatever it wrote is invented. Discard it and ask once more.
            emit({"event": "nudge", "agent": spec.key})
            messages[1] = dict(messages[1], content=messages[1]["content"] + "\nYou have no data yet. Call your tools first.")
            nudged = True
            continue
        text = response.content if evidence else ""
        break

    missing = [c for c in spec.required_calls(tickers, question) if not _covered(c, evidence)]
    if missing:
        calls = [ToolCall(f"coverage_{i}", name, args) for i, (name, args) in enumerate(missing)]
        messages.append(tool_call_message("", calls))
        for call in calls:
            execute(call, "coverage check")
        text = ""
    if not text:
        messages.append({"role": "user", "content": spec.findings_rules})
        text = ask(with_tools=False).content

    emit({"event": "agent_done", "agent": spec.key, "seconds": round(time.time() - started, 1)})
    finding = {
        "agent": spec.key,
        "title": spec.title,
        "focus": focus,
        "text": text.strip(),
        "draft": text.strip(),
        "seconds": round(time.time() - started, 1),
    }
    return {"finding": finding, "evidence": evidence, "usage": usage}


def revise_finding(llm: LLM, spec: Specialist, finding: dict, evidence: list[dict], unverified: list[str], emit: Emit) -> dict:
    """One self-correction pass: show the agent its untraceable numbers and its own evidence."""
    own = [model_view(e["id"], e["result"]) for e in evidence if e["agent"] == spec.key and not e.get("error")]
    messages = [
        {"role": "system", "content": spec.system_prompt() + "\n" + spec.findings_rules},
        {"role": "assistant", "content": finding["text"]},
        {
            "role": "user",
            "content": (
                f"A fact-checker could not find these numbers in your tool results: {', '.join(unverified)}.\n"
                "Rewrite your findings in the same format. Copy every fact word for word from the tool results below and "
                "cite its evidence id; remove anything you cannot find there. The closing sentence must contain no numbers.\n"
                f"Tool results: {json.dumps(own, separators=(',', ':'))}"
            ),
        },
    ]
    response = llm.chat(messages, role="specialist", max_tokens=700)
    emit(
        {
            "event": "llm",
            "agent": spec.key,
            "tool_calls": [],
            "revision": True,
            **{k: response.usage[k] for k in ("latency_s", "prompt_tokens", "completion_tokens")},
        }
    )
    revised = dict(finding, text=response.content.strip() or finding["text"], revised=True)
    return {"finding": revised, "usage": [usage_record(response, spec.key)]}
