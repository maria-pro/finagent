"""The fact-check applied to everything a model writes, shared by the workflow,
the report and the evaluation harness."""

from __future__ import annotations

from . import attribution, provenance


def fact_check(settings, text: str, evidence: list[dict], tickers: list[str], agent: str | None = None) -> dict:
    """Number provenance plus claim attribution for one piece of model-written text.

    A specialist is checked against its own evidence (it cannot know what the other
    agents' tools returned); the lead analyst may cite anything in the ledger.
    Returns the two summaries and the list of problems phrased for a revision prompt.
    """
    if agent and settings.enabled("own_evidence"):
        pool = [e for e in evidence if e["agent"] == agent]
        allowed = {e["id"] for e in pool}
    else:
        pool, allowed = evidence, None
    numbers = provenance.check(text, pool)
    claims = attribution.check(text, evidence, tickers, allowed) if settings.enabled("attribution") else None
    problems = claims.feedback() if claims else []
    mentioned = " ".join(problems)
    problems += [f"{n} is not in your tool results" for n in numbers.summary()["unverified"] if n not in mentioned]
    return {
        "provenance": numbers.summary(),
        "attribution": claims.summary() if claims else None,
        "problems": problems,
    }


def note_quality(findings: list[dict], narrative: str, evidence: list[dict], tickers: list[str]) -> dict:
    """Measure a finished note the same way whatever was switched on while writing it.

    Used for the run summary and by the evaluation harness, so an ablated run is
    scored by exactly the same yardstick as a full run: each specialist's text is
    checked against its own evidence, the summary against the whole ledger.
    """
    numbers_checked = numbers_traced = claims_checked = claims_ok = 0
    issues: list[str] = []
    sections = [(f["agent"], f["text"]) for f in findings] + [("writer", narrative)]
    for agent, text in sections:
        if not text:
            continue
        pool = evidence if agent == "writer" else [e for e in evidence if e["agent"] == agent]
        allowed = None if agent == "writer" else {e["id"] for e in pool}
        n = provenance.check(text, pool)
        c = attribution.check(text, evidence, tickers, allowed)
        numbers_checked += n.checked
        numbers_traced += n.verified
        claims_checked += c.checked
        claims_ok += c.supported
        issues += [f"{agent}: {line}" for line in c.summary()["issues"]]
    return {
        "numbers_checked": numbers_checked,
        "numbers_traced": numbers_traced,
        "number_provenance": round(numbers_traced / numbers_checked, 3) if numbers_checked else 1.0,
        "claims_checked": claims_checked,
        "claims_supported": claims_ok,
        "claim_attribution": round(claims_ok / claims_checked, 3) if claims_checked else 1.0,
        "claim_issues": issues,
    }
