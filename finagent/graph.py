"""The multi-agent workflow, as a LangGraph state machine.

    guard -> plan -> [specialists in parallel] -> verify -> [revise failing ones] -> write

``verify`` is deterministic: it checks every number each specialist wrote
against the evidence ledger and sends only the agents with untraceable numbers
back for one revision. Progress events are streamed with LangGraph's custom
stream so the CLI and the web UI can show each agent and tool call live.
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from . import provenance
from .agents import guard as guardrails
from .agents.planner import make_plan
from .agents.specialists import SPECIALISTS, revise_finding, run_specialist
from .agents.writer import write_summary
from .llm import LLM
from .report import build_report
from .tools import ToolContext


def _merge(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}


class ResearchState(TypedDict, total=False):
    tickers: list[str]
    question: str
    started: float
    guard: dict
    plan: list[dict]
    plan_source: str
    findings: Annotated[dict[str, dict], _merge]
    evidence: Annotated[list[dict], operator.add]
    llm_calls: Annotated[list[dict], operator.add]
    revisions: Annotated[list[str], operator.add]
    narrative: str
    promissory_phrases: list[str]
    report: str
    stats: dict


def _emitter():
    try:
        writer = get_stream_writer()
    except Exception:  # invoked outside a streaming run
        return lambda event: None
    return lambda event: writer({"t": round(time.time(), 2), **event})


def build_graph(llm: LLM, ctx: ToolContext):
    settings = ctx.settings

    def guard(state: ResearchState) -> dict:
        decision = guardrails.check_input(settings, state["question"])
        _emitter()({"event": "guard", **decision})
        return {"guard": decision, "started": state.get("started") or time.time()}

    def refuse(state: ResearchState) -> dict:
        return {"report": guardrails.refusal_message(state["guard"]["reason"]), "stats": {"refused": True}}

    def plan(state: ResearchState) -> dict:
        tasks, usage, source = make_plan(llm, state["tickers"], state["question"])
        _emitter()({"event": "plan", "tasks": tasks, "source": source})
        return {"plan": tasks, "plan_source": source, "llm_calls": usage}

    def specialist(payload: dict) -> dict:
        spec = SPECIALISTS[payload["agent"]]
        out = run_specialist(
            llm, ctx, spec, tickers=payload["tickers"], question=payload["question"], focus=payload["focus"], emit=_emitter()
        )
        return {"findings": {spec.key: out["finding"]}, "evidence": out["evidence"], "llm_calls": out["usage"]}

    def verify(state: ResearchState) -> dict:
        emit = _emitter()
        updated = {}
        for key, finding in state["findings"].items():
            # An agent can only know what its own tools returned, so check against those alone.
            own = [e for e in state.get("evidence", []) if e["agent"] == key]
            report = provenance.check(finding["text"], own)
            updated[key] = dict(finding, provenance=report.summary())
            emit({"event": "verify", "agent": key, **report.summary()})
        return {"findings": updated}

    def revise(payload: dict) -> dict:
        spec = SPECIALISTS[payload["agent"]]
        emit = _emitter()
        emit({"event": "revise", "agent": spec.key, "unverified": payload["unverified"]})
        out = revise_finding(llm, spec, payload["finding"], payload["evidence"], payload["unverified"], emit)
        finding = out["finding"]
        own = [e for e in payload["evidence"] if e["agent"] == spec.key]
        finding["provenance"] = provenance.check(finding["text"], own).summary()
        emit({"event": "verify", "agent": spec.key, "after_revision": True, **finding["provenance"]})
        return {"findings": {spec.key: finding}, "llm_calls": out["usage"], "revisions": [spec.key]}

    def write(state: ResearchState) -> dict:
        emit = _emitter()
        emit({"event": "agent_start", "agent": "writer", "title": "Lead Analyst", "focus": "summary"})
        findings = [state["findings"][t["agent"]] for t in state["plan"] if t["agent"] in state["findings"]]
        narrative, usage = write_summary(
            llm, state["tickers"], state["question"], findings, state["guard"].get("personal_advice", False)
        )
        emit({"event": "llm", "agent": "writer", "tool_calls": [], **{k: usage[0][k] for k in ("latency_s", "prompt_tokens", "completion_tokens")}})
        phrases = guardrails.check_output(narrative + " ".join(f["text"] for f in findings))
        draft = {**state, "narrative": narrative, "llm_calls": state.get("llm_calls", []) + usage, "promissory_phrases": phrases}
        report, stats = build_report(draft, settings, time.time() - state["started"])
        emit({"event": "done", "stats": stats})
        return {"narrative": narrative, "llm_calls": usage, "promissory_phrases": phrases, "report": report, "stats": stats}

    def route_after_guard(state: ResearchState) -> str:
        return "plan" if state["guard"]["allowed"] else "refuse"

    def fan_out(state: ResearchState) -> list[Send]:
        return [
            Send("specialist", {"agent": t["agent"], "focus": t["focus"], "tickers": state["tickers"], "question": state["question"]})
            for t in state["plan"]
        ]

    def route_after_verify(state: ResearchState) -> list[Send] | str:
        sends = []
        for key, finding in state["findings"].items():
            unverified = finding.get("provenance", {}).get("unverified", [])
            if unverified and key not in state.get("revisions", []):
                sends.append(Send("revise", {"agent": key, "finding": finding, "evidence": state["evidence"], "unverified": unverified}))
        return sends or "write"

    graph = StateGraph(ResearchState)
    graph.add_node("guard", guard)
    graph.add_node("refuse", refuse)
    graph.add_node("plan", plan)
    graph.add_node("specialist", specialist)
    graph.add_node("verify", verify)
    graph.add_node("revise", revise)
    graph.add_node("write", write)

    graph.add_edge(START, "guard")
    graph.add_conditional_edges("guard", route_after_guard, ["plan", "refuse"])
    graph.add_edge("refuse", END)
    graph.add_conditional_edges("plan", fan_out, ["specialist"])
    graph.add_edge("specialist", "verify")
    graph.add_conditional_edges("verify", route_after_verify, ["revise", "write"])
    graph.add_edge("revise", "write")
    graph.add_edge("write", END)
    return graph.compile()


def run_research(llm: LLM, ctx: ToolContext, tickers: list[str], question: str, on_event=None) -> dict[str, Any]:
    """Run the workflow to completion, forwarding progress events to ``on_event``."""
    graph = build_graph(llm, ctx)
    inputs = {"tickers": tickers, "question": question, "started": time.time()}
    # Model concurrency is limited inside the LLM client, not here: LangGraph's
    # max_concurrency=1 starves its own stream consumer and deadlocks.
    config = {"recursion_limit": 50}
    final: dict[str, Any] = dict(inputs)
    for mode, chunk in graph.stream(inputs, config=config, stream_mode=["custom", "values"]):
        if mode == "custom" and on_event:
            on_event(chunk)
        elif mode == "values":
            final = chunk
    return final
