"""The lead analyst turns specialist findings into the note's opening sections."""

from __future__ import annotations

from ..llm import LLM, usage_record

_SYSTEM = """You are the lead analyst on an equity research team. Write the opening of a research note from your team's findings.
Use exactly this structure:
## Answer
Two or three sentences that directly answer the question.
## Strengths
- two or three bullets
## Risks
- two or three bullets
## What to watch
- two bullets
Rules:
- Build each Strengths and Risks bullet by copying one finding bullet word for word, including its evidence id in square brackets.
- The Answer must address every part of the question, citing the finding that bears on each part: valuation questions need the valuation figures (P/E, and price-to-sales or, for banks and insurers, price-to-book), fundamentals the revenue and margin figures, risks the risk figures or the 10-K points.
- In the Answer, copy each number from a finding together with its label - the same company, period, fiscal year and metric - and put that finding's evidence id in square brackets right after it.
- Do not recommend buying or selling. Write for an educated reader, in plain English."""


def _personal_advice_note(personal_advice: bool) -> str:
    if not personal_advice:
        return ""
    return "\nThe user asked for personal advice; answer with general information and note that it is not personalised advice."


def _team_notes(findings: list[dict]) -> str:
    return "\n\n".join(f"### {f['title']}\n{f['text']}" for f in findings if f.get("text"))


def write_summary(llm: LLM, tickers: list[str], question: str, findings: list[dict], personal_advice: bool) -> tuple[str, list[dict]]:
    messages = [
        {"role": "system", "content": _SYSTEM + _personal_advice_note(personal_advice)},
        {"role": "user", "content": f"Tickers: {', '.join(tickers)}\nQuestion: {question}\n\nTeam findings:\n{_team_notes(findings)}"},
    ]
    response = llm.chat(messages, role="writer", max_tokens=900)
    return response.content.strip(), [usage_record(response, "writer")]


def revise_summary(
    llm: LLM, tickers: list[str], question: str, findings: list[dict], draft: str, problems: list[str], personal_advice: bool
) -> tuple[str, list[dict]]:
    """One correction pass for the lead analyst, driven by the same fact-check as the specialists."""
    issues = "\n".join(f"- {p}" for p in problems)
    messages = [
        {"role": "system", "content": _SYSTEM + _personal_advice_note(personal_advice)},
        {"role": "user", "content": f"Tickers: {', '.join(tickers)}\nQuestion: {question}\n\nTeam findings:\n{_team_notes(findings)}"},
        {"role": "assistant", "content": draft},
        {
            "role": "user",
            "content": (
                f"A fact-checker found these problems in your draft:\n{issues}\n"
                "Rewrite the note opening in the same structure. Copy each number from the findings with its company, "
                "period and fiscal year and its evidence id; remove anything the findings do not support."
            ),
        },
    ]
    response = llm.chat(messages, role="writer", max_tokens=900)
    return (response.content.strip() or draft), [usage_record(response, "writer")]
