"""Number provenance: is every figure in the report backed by a tool result?

This is the deterministic "critic". Instead of asking a (large, expensive)
model to judge whether a draft hallucinated, we extract every number it wrote
and look for it in the evidence ledger, allowing for rounding and unit changes
(0.2282 -> 22.82%, 215938000000 -> $215.94B). Numbers that cannot be traced are
sent back to the agent that wrote them, and flagged in the final report if they
survive the revision.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

_NUMBER_RE = re.compile(
    r"(?<![\w.\-/#])(?P<cur>[$€£])?(?P<sign>[-−–+])?(?P<cur2>[$€£])?"
    r"(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?P<dec>\.\d+)?"
    r"(?:\s?(?P<suf>%|x(?![a-z])|×|bn\b|billion\b|b\b|mn\b|million\b|m\b|trillion\b|tn\b|t\b|k\b|thousand\b))?",
    re.IGNORECASE,
)
_MONTHS = re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*$", re.IGNORECASE)
_TIME_UNIT_AFTER = re.compile(r"^[\s-]*(trading\s+)?(day|days|week|weeks|month|months|year|years|quarter|quarters|yr|yrs)\b", re.I)
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# Names that contain numbers but are not figures: market indices, filing forms, SEC identifiers.
_NAMED_RE = re.compile(
    r"S\s?&\s?P\s?(500|400|600|100)|Russell\s?(1000|2000|3000)|Nasdaq[- ]?100|FTSE\s?(100|250)|Dow\s?(30|Jones)|"
    r"DAX\s?40|CAC\s?40|Nikkei\s?225|Stoxx\s?(50|600)|ASX\s?200|MSCI\s?\w+|Form\s?\d+-?\w*|"
    r"CIK\s?\d+|\b\d{10}-\d{2}-\d{6}\b",
    re.IGNORECASE,
)
_CITATION_AFTER = re.compile(r"^[\s)*,]{0,3}\[((?:[A-Z]\d+)(?:\s*[,;]\s*[A-Z]\d+)*)\]")

_SCALES = {
    "%": (1.0, 100.0),
    "b": (1.0, 1e-9),
    "m": (1.0, 1e-6, 1e3),
    "t": (1.0, 1e-12, 1e-3),
    "k": (1.0, 1e-3),
    "x": (1.0,),
    "": (1.0, 100.0, 1e-9, 1e-6),
}
_SUFFIX_KEY = {
    "%": "%", "x": "x", "×": "x",
    "bn": "b", "billion": "b", "b": "b",
    "mn": "m", "million": "m", "m": "m",
    "trillion": "t", "tn": "t", "t": "t",
    "k": "k", "thousand": "k",
}


@dataclass
class NumberClaim:
    text: str
    value: float
    unit: str
    decimals: int
    start: int
    end: int
    cited: list[str] = field(default_factory=list)  # evidence ids written right after the number
    source: str | None = None
    miscited: bool = False  # found in the ledger, but not in the evidence it cites

    @property
    def verified(self) -> bool:
        return self.source is not None


@dataclass
class ProvenanceReport:
    claims: list[NumberClaim] = field(default_factory=list)

    @property
    def checked(self) -> int:
        return len(self.claims)

    @property
    def verified(self) -> int:
        return sum(c.verified for c in self.claims)

    @property
    def score(self) -> float:
        return self.verified / self.checked if self.claims else 1.0

    @property
    def unverified(self) -> list[NumberClaim]:
        return [c for c in self.claims if not c.verified]

    def summary(self) -> dict:
        return {
            "checked": self.checked,
            "verified": self.verified,
            "score": round(self.score, 3),
            "unverified": [c.text for c in self.unverified],
            "miscited": [c.text for c in self.claims if c.miscited],
        }


def extract_numbers(text: str) -> list[NumberClaim]:
    """Numeric claims in prose, skipping years, dates, small counts, citations and identifiers."""
    masked = _DATE_RE.sub(lambda m: " " * len(m.group()), text)
    masked = _NAMED_RE.sub(lambda m: " " * len(m.group()), masked)
    claims = []
    for m in _NUMBER_RE.finditer(masked):
        raw_int, dec, suf = m.group("int"), m.group("dec") or "", (m.group("suf") or "").lower()
        has_currency = bool(m.group("cur") or m.group("cur2"))
        after, before = masked[m.end() : m.end() + 12], masked[max(0, m.start() - 12) : m.start()]
        if not suf and re.match(r"[A-Za-z]", after[:1]):
            continue  # identifiers such as 1A, 3rd, 5G
        if re.match(r"-[A-Za-z]", after):
            continue  # 10-K, 52-week, 200-day
        if not suf and not dec and _TIME_UNIT_AFTER.match(after):
            continue  # "5 years", "10 trading days" are parameters, not findings
        value = float(raw_int.replace(",", "") + dec)
        if not suf and not dec and not has_currency:
            if 1900 <= value <= 2100 or value < 10:
                continue  # years and small counts
            if value <= 31 and _MONTHS.search(before):
                continue  # day of month
        citation = _CITATION_AFTER.match(masked[m.end() : m.end() + 40])
        claims.append(
            NumberClaim(
                text=m.group().strip(),
                value=value,
                unit=_SUFFIX_KEY.get(suf, ""),
                decimals=len(dec) - 1 if dec else 0,
                start=m.start(),
                end=m.end(),
                cited=re.findall(r"[A-Z]\d+", citation.group(1)) if citation else [],
            )
        )
    return claims


def _key_unit(key: str) -> str:
    """The unit a tool result's field is in, from its name (``total_return_pct``, ``market_cap_bn``)."""
    key = key.lower()
    if key.endswith("_pct"):
        return "%"
    if key.endswith("_bn"):
        return "b"
    return ""


def _walk(value: Any, key: str = "") -> Iterable[tuple[float, str]]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            yield float(value), _key_unit(key)
    elif isinstance(value, str):
        for claim in extract_numbers(value):
            yield claim.value, claim.unit
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _walk(v, str(k))
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk(v, key)


def evidence_numbers(evidence: list[dict]) -> list[tuple[float, str, str]]:
    """(absolute value, unit, evidence id) for every number in every tool result."""
    out = []
    for ev in evidence:
        for v, unit in _walk(ev.get("result", {})):
            out.append((abs(v), unit, ev["id"]))
    return out


# How a value stored in one unit may be written in another: {claim unit: {evidence unit: scales}}.
# A percentage traces to a percentage or to a fraction (0.2282 -> 22.82%), never to a ratio such
# as a P/E of 23.1; amounts convert between bn, million and trillion and raw dollars.
_COMPATIBLE = {
    "%": {"%": (1.0,), "": (100.0,)},
    "b": {"b": (1.0,), "": (1e-9,)},
    "m": {"m": (1.0,), "b": (1e3,), "": (1e-6,)},
    "t": {"t": (1.0,), "b": (1e-3,), "": (1e-12,)},
    "k": {"k": (1.0,), "": (1e-3,)},
    "x": {"x": (1.0,), "": (1.0,)},
    "": {"": (1.0,), "x": (1.0,)},
}


def _matches(claim: NumberClaim, value: float, unit: str | None = None) -> bool:
    """Display rounding only (5349.48bn shown as $5.35 trillion), not "close enough". With ``unit``,
    only unit-consistent conversions count; without it, any scale does (the unit-agnostic match)."""
    tolerance = 0.5 * 10 ** (-claim.decimals) * 1.001 + 0.0005 * claim.value
    scales = _SCALES.get(claim.unit, _SCALES[""]) if unit is None else _COMPATIBLE.get(claim.unit, {}).get(unit, ())
    return any(abs(claim.value - value * scale) <= tolerance for scale in scales)


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'*\-•])|\n")
_CITATIONS = re.compile(r"\[((?:[A-Z]\d+)(?:\s*[,;]\s*[A-Z]\d+)*)\]")


def _sentence_citations(text: str, claim: NumberClaim) -> list[str]:
    """Evidence ids cited anywhere in the sentence or bullet that holds ``claim``."""
    starts = [m.end() for m in _SENTENCE_END.finditer(text, 0, claim.start)]
    end = _SENTENCE_END.search(text, claim.end)
    sentence = text[starts[-1] if starts else 0 : end.start() if end else len(text)]
    return re.findall(r"[A-Z]\d+", " ".join(_CITATIONS.findall(sentence)))


def check(text: str, evidence: list[dict]) -> ProvenanceReport:
    """Trace each number to the evidence it cites, falling back to the whole ledger.

    Units must agree. A number found only in evidence it does not cite still counts as
    grounded, but is reported as mis-cited - except a rounded number (fewer than two
    decimals): it matches so many values by chance that it counts only if the evidence its
    own sentence cites contains it.
    """
    pool = evidence_numbers(evidence)
    claims = extract_numbers(text)
    for claim in claims:
        if claim.decimals < 2:
            nearby = set(claim.cited or _sentence_citations(text, claim))
            claim.source = next((ev for v, u, ev in pool if ev in nearby and _matches(claim, v, u)), None)
            continue
        if claim.cited:
            claim.source = next((ev for v, u, ev in pool if ev in claim.cited and _matches(claim, v, u)), None)
        if claim.source is None:
            claim.source = next((ev for v, u, ev in pool if _matches(claim, v, u)), None)
            claim.miscited = claim.source is not None and bool(claim.cited)
    return ProvenanceReport(claims)


def annotate(text: str, report: ProvenanceReport, marker: str = " ⚠unverified") -> str:
    """Insert a visible marker after each number that could not be traced."""
    for claim in sorted(report.unverified, key=lambda c: c.start, reverse=True):
        text = text[: claim.end] + marker + text[claim.end :]
    return text
