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
- Use only numbers that appear in the findings, copied exactly, each followed by the evidence id the finding gives it: <number> [<evidence id>].
- Keep each number with the label the finding gives it (the same period, metric and company).
- Do not recommend buying or selling. Write for an educated reader, in plain English."""


def write_summary(llm: LLM, tickers: list[str], question: str, findings: list[dict], personal_advice: bool) -> tuple[str, list[dict]]:
    team_notes = "\n\n".join(f"### {f['title']}\n{f['text']}" for f in findings if f.get("text"))
    framing = (
        "\nThe user asked for personal advice; answer with general information and note that it is not personalised advice."
        if personal_advice
        else ""
    )
    messages = [
        {"role": "system", "content": _SYSTEM + framing},
        {"role": "user", "content": f"Tickers: {', '.join(tickers)}\nQuestion: {question}\n\nTeam findings:\n{team_notes}"},
    ]
    response = llm.chat(messages, role="writer", max_tokens=900)
    return response.content.strip(), [usage_record(response, "writer")]
