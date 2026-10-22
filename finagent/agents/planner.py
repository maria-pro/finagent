"""The planner decides which specialists a question needs.

Its output is constrained to a JSON schema, so even a 4B model returns a valid
plan (llama.cpp and Ollama turn the schema into a decoding grammar). If parsing
still fails, the planner falls back to the full team rather than failing the run.
"""

from __future__ import annotations

from ..llm import LLM, extract_json, usage_record
from .specialists import SPECIALISTS

PLAN_SCHEMA = {
    "title": "research_plan",
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "enum": list(SPECIALISTS)},
                    "focus": {"type": "string"},
                },
                "required": ["agent", "focus"],
            },
        }
    },
    "required": ["tasks"],
}

_SYSTEM = """You plan equity research by assigning work to specialist agents.
Specialists:
- market: price performance, volatility, drawdowns, comparison with the S&P 500
- fundamentals: revenue growth, margins, cash flow and valuation from SEC filings
- risk: Value at Risk, expected shortfall, crash stress tests
- filings: what the latest 10-K annual report says about risks and the business
Choose the specialists the question needs (usually 3 or 4). For each, write a focus: one short sentence, specific to the question, saying what to look for.
Example for "Is AMD's growth priced in?":
{"tasks": [{"agent": "fundamentals", "focus": "revenue growth trend versus the P/E and price-to-sales multiples"}, {"agent": "market", "focus": "how much of the growth is already reflected in recent returns"}, {"agent": "filings", "focus": "competitive threats to future growth"}]}
Return JSON only."""

DEFAULT_FOCUS = {
    "market": "recent price performance, volatility and behaviour versus the S&P 500",
    "fundamentals": "growth, profitability and valuation",
    "risk": "downside risk and losses in past market crashes",
    "filings": "the main risks the company itself highlights",
}


def make_plan(llm: LLM, tickers: list[str], question: str) -> tuple[list[dict], list[dict], str]:
    """Return (tasks, usage records, how the plan was produced)."""
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Tickers: {', '.join(tickers)}\nQuestion: {question}"},
    ]
    usage: list[dict] = []
    try:
        response = llm.chat(messages, role="planner", json_schema=PLAN_SCHEMA, max_tokens=300)
        usage.append(usage_record(response, "planner"))
        data = extract_json(response.content)
        tasks, seen = [], set()
        for item in data.get("tasks", []) if isinstance(data, dict) else []:
            agent = str(item.get("agent", "")).strip().lower()
            if agent in SPECIALISTS and agent not in seen:
                seen.add(agent)
                focus = str(item.get("focus") or "").strip()[:160] or DEFAULT_FOCUS[agent]
                tasks.append({"agent": agent, "focus": focus})
        if tasks:
            return tasks, usage, "model"
    except Exception:
        pass
    return [{"agent": a, "focus": f} for a, f in DEFAULT_FOCUS.items()], usage, "fallback (full team)"
