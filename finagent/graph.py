"""The multi-agent workflow, as a LangGraph state machine.

    guard -> plan -> [specialists in parallel] -> verify -> [revise failing ones] -> write (+ check, revise once)

``verify`` is deterministic. For each specialist it traces every number to the
agent's own tool results (``provenance``) and checks that every cited claim says
what its source says - company, period, metric, direction (``attribution``).
Only agents with problems go back for one revision, with the exact problems in
the prompt. The lead analyst's summary gets the same check and one revision.
Progress events are streamed with LangGraph's custom stream so the CLI and the
web UI can show each agent and tool call live.
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .agents import guard as guardrails
from .agents.planner import make_plan
from .agents.specialists import SPECIALISTS, revise_finding, run_specialist
from .agents.writer import revise_summary, write_summary
from .checks import fact_check
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
    narrative_draft: str
    writer_check: dict
    promissory_phrases: list[str]
    report: str
    stats: dict


def _emitter():
    try:
        writer = get_stream_writer()
    except Exception:  # invoked outside a streaming run
        return lambda event: None
    return lambda event: writer({"t": round(time.time(), 2), **event})


def _check_event(agent: str, check: dict, **extra) -> dict:
    claims = check["attribution"] or {}
    return {
        "event": "verify",
        "agent": agent,
        **check["provenance"],
        "claims_checked": claims.get("claims", 0),
        "claims_supported": claims.get("supported", 0),
        "claim_issues": claims.get("issues", []),
        **extra,
    }


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
            check = fact_check(settings, finding["text"], state.get("evidence", []), state["tickers"], agent=key)
            updated[key] = dict(finding, **check)
            emit(_check_event(key, check))
        return {"findings": updated}

    def revise(payload: dict) -> dict:
        spec = SPECIALISTS[payload["agent"]]
        emit = _emitter()
        emit({"event": "revise", "agent": spec.key, "problems": payload["problems"]})
        out = revise_finding(
            llm, spec, payload["finding"], payload["evidence"], payload["problems"], emit, use_facts=settings.enabled("facts")
        )
        finding = out["finding"]
        check = fact_check(settings, finding["text"], payload["evidence"], payload["tickers"], agent=spec.key)
        finding.update(check)
        emit(_check_event(spec.key, check, after_revision=True))
        return {"findings": {spec.key: finding}, "llm_calls": out["usage"], "revisions": [spec.key]}

    def write(state: ResearchState) -> dict:
        emit = _emitter()
        emit({"event": "agent_start", "agent": "writer", "title": "Lead Analyst", "focus": "summary"})
        findings = [state["findings"][t["agent"]] for t in state["plan"] if t["agent"] in state["findings"]]
        advice = state["guard"].get("personal_advice", False)
        evidence, tickers = state.get("evidence", []), state["tickers"]

        narrative, usage = write_summary(llm, tickers, state["question"], findings, advice)
        emit({"event": "llm", "agent": "writer", "tool_calls": [], **{k: usage[-1][k] for k in ("latency_s", "prompt_tokens", "completion_tokens")}})
        draft, check = narrative, fact_check(settings, narrative, evidence, tickers)
        emit(_check_event("writer", check))
        revisions = []
        if check["problems"] and settings.enabled("revision"):
            emit({"event": "revise", "agent": "writer", "problems": check["problems"]})
            revised, more = revise_summary(llm, tickers, state["question"], findings, narrative, check["problems"], advice)
            usage += more
            emit({"event": "llm", "agent": "writer", "tool_calls": [], "revision": True, **{k: more[-1][k] for k in ("latency_s", "prompt_tokens", "completion_tokens")}})
            revised_check = fact_check(settings, revised, evidence, tickers)
            emit(_check_event("writer", revised_check, after_revision=True))
            revisions = ["writer"]
            if len(revised_check["problems"]) <= len(check["problems"]):  # keep the better of the two
                narrative, check = revised, revised_check

        phrases = guardrails.check_output(narrative + " ".join(f["text"] for f in findings))
        final = {
            **state,
            "narrative": narrative,
            "narrative_draft": draft,
            "writer_check": check,
            "llm_calls": state.get("llm_calls", []) + usage,
            "revisions": state.get("revisions", []) + revisions,
            "promissory_phrases": phrases,
        }
        report, stats = build_report(final, settings, time.time() - state["started"])
        emit({"event": "done", "stats": stats})
        return {
            "narrative": narrative,
            "narrative_draft": draft,
            "writer_check": check,
            "llm_calls": usage,
            "revisions": revisions,
            "promissory_phrases": phrases,
            "report": report,
            "stats": stats,
        }

    def route_after_guard(state: ResearchState) -> str:
        return "plan" if state["guard"]["allowed"] else "refuse"

    def fan_out(state: ResearchState) -> list[Send]:
        return [
            Send("specialist", {"agent": t["agent"], "focus": t["focus"], "tickers": state["tickers"], "question": state["question"]})
            for t in state["plan"]
        ]

    def route_after_verify(state: ResearchState) -> list[Send] | str:
        if not settings.enabled("revision"):
            return "write"
        sends = []
        for key, finding in state["findings"].items():
            if finding.get("problems") and key not in state.get("revisions", []):
                sends.append(
                    Send(
                        "revise",
                        {"agent": key, "finding": finding, "evidence": state["evidence"], "problems": finding["problems"], "tickers": state["tickers"]},
                    )
                )
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
