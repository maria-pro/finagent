"""A scripted stand-in for the language model, so the workflow can be tested offline and deterministically."""

from __future__ import annotations

import json
import re

from finagent.llm import LLMResponse, ToolCall

_TOOLS_BY_TITLE = {
    "Market Analyst": [("price_performance", "ticker"), ("benchmark_comparison", "ticker")],
    "Fundamentals Analyst": [("financial_statements", "ticker"), ("valuation_snapshot", "ticker")],
    "Risk Analyst": [("value_at_risk", "tickers")],  # stress_test left for the coverage check
    "Filings Analyst": [("search_10k", "ticker")],
}


class ScriptedLLM:
    """Plays every role. ``hallucinate`` makes the market analyst's first draft include an untraceable number."""

    def __init__(self, hallucinate: bool = True, plan: list[str] | None = None, sloppy_writer: bool = False):
        self.hallucinate = hallucinate
        self.sloppy_writer = sloppy_writer
        self.plan = plan or ["market", "fundamentals", "risk", "filings"]
        self.calls: list[dict] = []

    def _usage(self, role: str) -> dict:
        return {"role": role, "model": "scripted", "prompt_tokens": 100, "completion_tokens": 20, "latency_s": 0.01}

    def chat(self, messages, *, role, tools=None, json_schema=None, max_tokens=1024):
        self.calls.append({"role": role, "messages": messages, "tools": [t["function"]["name"] for t in tools or []]})
        system = messages[0]["content"]
        if role == "planner":
            tasks = [{"agent": a, "focus": f"focus for {a}"} for a in self.plan]
            return LLMResponse(json.dumps({"tasks": tasks}), usage=self._usage(role))
        if role == "writer":
            # copy finding bullets word for word, as the writer prompt asks
            bullets = [line for line in messages[1]["content"].splitlines() if line.startswith("- ") and "[" in line]
            if self.sloppy_writer and messages[-1]["role"] == "user" and "fact-checker" not in messages[-1]["content"]:
                bullets = [re.sub(r"\b\d+y\b ", "", b) for b in bullets]  # drops the period from each fact
            text = (
                f"## Answer\n{bullets[0][2:] if bullets else 'No data.'}\n## Strengths\n" + "\n".join(bullets[1:3])
                + "\n## Risks\n" + "\n".join(bullets[3:4]) + "\n## What to watch\n- margins"
            )
            return LLMResponse(text, usage=self._usage(role))

        # specialist or revision
        title = re.search(r"You are the (.+?) on an", system).group(1)
        tool_results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        is_revision = messages[-1]["role"] == "user" and "fact-checker" in messages[-1]["content"]
        if is_revision:
            payload = json.loads(messages[-1]["content"].split("Tool results: ", 1)[1])
            return LLMResponse(self._bullets(payload), usage=self._usage(role))
        if tools and not tool_results:
            ticker = re.search(r"Tickers: ([A-Z, ]+)", messages[1]["content"]).group(1).split(",")[0].strip()
            calls = [
                ToolCall(f"call_{i}", name, {key: [ticker] if key == "tickers" else ticker, **({"query": "risks"} if name == "search_10k" else {})})
                for i, (name, key) in enumerate(_TOOLS_BY_TITLE[title])
            ]
            return LLMResponse("", tool_calls=calls, usage=self._usage(role))
        text = self._bullets(tool_results)
        if self.hallucinate and title == "Market Analyst":
            text += "\n- Analyst price target: 987.65% [M1]"
        return LLMResponse(text, usage=self._usage(role))

    @staticmethod
    def _bullets(results: list[dict]) -> str:
        """Copy the first facts of each tool result, as the prompt instructs."""
        lines = []
        for r in results:
            for fact in r.get("facts", [])[:2]:
                lines.append(f"- {fact} [{r['evidence_id']}]")
            for passage in r.get("passages", [])[:1]:
                lines.append(f"- {passage['text'][:80]} [{r['evidence_id']}]")
        return "\n".join(lines[:5]) + "\nOverall the data is mixed."
