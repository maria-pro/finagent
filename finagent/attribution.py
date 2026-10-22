"""Claim attribution: does each claim say what its cited source says?

``provenance.py`` asks whether a number exists somewhere in the evidence. That
is a safety net, not a quality measure: a model told to copy pre-labelled facts
can quote a real number under the wrong label - an older year's growth as the
latest, a 25-day VaR as "1-year", one company's figure under another's name, a
fall described as a rise. This module checks each cited claim against the
sources it cites:

* every number in the claim must appear in a cited source, and the claim must
  keep that source's labels - company (when the run covers several), period
  (1y/3y/5y, n-day horizon, fiscal year, trailing twelve months), method
  (historical vs filtered simulation) and metric (gross vs operating margin,
  VaR vs expected shortfall ...);
* direction words ("fell", "rose", "gained" ...) must agree with the sign of the
  source value;
* claims without numbers (typically 10-K points) must share most of their
  content words with a cited passage.

Everything here is deterministic and explainable: each issue names the claim,
the problem and the source text, which is what the revision prompt shows the
agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .provenance import NumberClaim, extract_numbers
from .provenance import _matches as numbers_match

_CITATION_RE = re.compile(r"\[((?:[A-Z]\d+)(?:\s*[,;]\s*[A-Z]\d+)*)\]")
_ID_RE = re.compile(r"[A-Z]\d+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")
_WORD_RE = re.compile(r"[a-z0-9/&']+")

_NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "ten": "10", "twelve": "12"}
_STOP = set(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with "
    "we our could can may might also than into more over such which their they been not per vs versus".split()
)
# Mutually exclusive labels: if the source says one member and the claim another, the label is wrong.
_CONFLICT_GROUPS = [
    {"gross", "operating", "net"},
    {"value at risk", "expected shortfall"},
    {"historical", "filtered"},
    {"total return", "annualized return", "excess"},
    {"revenue", "net income", "free cash flow", "eps"},
    {"p/e", "price-to-sales", "ev-to-sales", "p/b"},
]
# Regex aliases for each label (matched on lower-cased text).
_SYNONYMS = {
    "value at risk": (r"value at risk", r"\bvar\b"),
    "expected shortfall": (r"expected shortfall", r"\bcvar\b", r"\bes\b"),
    "filtered": (r"\bfiltered\b", r"\bsimulation\b", r"\bfhs\b", r"current volatility", r"\bconditional\b"),
    "historical": (r"\bhistorical\b",),
    "p/e": (r"\bp/e\b", r"price[- ]to[- ]earnings", r"\bpe\b"),
    "price-to-sales": (r"price[- ]to[- ]sales", r"\bp/s\b"),
    "ev-to-sales": (r"\bev[- ]to[- ]sales\b", r"\bev/sales\b"),
    "p/b": (r"\bp/b\b", r"price[- ]to[- ]book"),
    "free cash flow": (r"free[- ]cash[- ]flow", r"\bfcf\b"),
    "eps": (r"\beps\b", r"earnings per share"),
    "annualized return": (r"annuali[sz]ed return", r"\bcagr\b", r"per year"),
    "total return": (r"total return",),
    "excess": (r"\bexcess\b", r"outperform", r"underperform"),
    "gross": (r"\bgross\b",),
    "operating": (r"\boperating\b",),
    "net": (r"\bnet margin\b", r"\bnet income\b", r"\bnet profit\b", r"%\s+net\b"),  # "32.81% net"
    "revenue": (r"\brevenues?\b",),
    "net income": (r"\bnet income\b", r"\bnet profit\b"),
}
_NEGATIVE = {"fell", "fall", "falls", "declined", "decline", "declines", "dropped", "drop", "down", "lost", "lose", "loses",
             "decreased", "decrease", "negative", "shrank", "contracted", "slumped", "plunged", "underperformed", "minus"}
_POSITIVE = {"rose", "rise", "rises", "gained", "gain", "gains", "up", "increased", "increase", "grew", "grow", "grows",
             "positive", "jumped", "surged", "climbed", "outperformed", "plus"}


@dataclass
class ClaimCheck:
    text: str
    cited: list[str]
    start: int
    end: int
    issues: list[str] = field(default_factory=list)
    source: str | None = None
    support: float = 0.0
    # "numeric" (has numbers), "qualitative" (cites a passage, no numbers), "interpretation"
    # (no numbers, cites only numeric facts: an opinion, not a claim about data), "uncited"
    kind: str = "interpretation"

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass
class AttributionReport:
    claims: list[ClaimCheck] = field(default_factory=list)

    @property
    def checked(self) -> int:
        return len(self.claims)

    @property
    def supported(self) -> int:
        return sum(c.ok for c in self.claims)

    @property
    def score(self) -> float:
        return self.supported / self.checked if self.claims else 1.0

    @property
    def flagged(self) -> list[ClaimCheck]:
        return [c for c in self.claims if not c.ok]

    def feedback(self) -> list[str]:
        """One line per problem, phrased for a revision prompt."""
        lines = []
        for c in self.flagged:
            src = f' Source says: "{c.source}"' if c.source else ""
            lines.append(f'"{c.text[:110]}": {"; ".join(c.issues)}.{src}')
        return lines

    def summary(self) -> dict:
        return {
            "claims": self.checked,
            "supported": self.supported,
            "score": round(self.score, 3),
            "issues": [f"{c.text[:80]} -> {'; '.join(c.issues)}" for c in self.flagged],
        }


# --------------------------------------------------------------------------- text helpers


def _normalize(text: str) -> str:
    t = " " + text.lower().replace("−", "-").replace("–", "-").replace("—", "-") + " "
    for word, digit in _NUMBER_WORDS.items():
        t = re.sub(rf"\b{word}\b", digit, t)
    return t


def _content_words(text: str) -> set[str]:
    words = set()
    for w in _WORD_RE.findall(_CITATION_RE.sub(" ", text.lower())):
        if w in _STOP or len(w) < 3 or w.replace(".", "").isdigit():
            continue
        for suffix in ("ing", "ed", "es", "s", "ly"):
            if w.endswith(suffix) and len(w) - len(suffix) >= 4:
                w = w[: -len(suffix)]
                break
        words.add(w)
    return words


_TTM_RE = re.compile(r"trailing[- ](?:12|twelve)[- ]months?|\bttm\b|\bltm\b|last 12 months")


def _periods(text: str) -> set[str]:
    """Period/horizon qualifiers, normalised: 1y/3y/5y, Nd horizons, fyYYYY, ttm."""
    t = _normalize(text)
    found = set()
    if _TTM_RE.search(t):
        found.add("ttm")
        t = _TTM_RE.sub(" ", t)  # "trailing 12 months" is not a 1-year price window
    found |= {f"{n}y" for n in re.findall(r"\b(\d{1,2})[- ]?(?:y|yr|yrs|year|years)\b", t)}
    if re.search(r"\b(past|last|trailing|latest)\s+year\b|\b12[- ]month\b", t):
        found.add("1y")
    found |= {f"{n}d" for n in re.findall(r"\b(\d{1,3})[- ](?:trading[- ])?days?\b", t)}
    found |= {f"fy{y}" for y in re.findall(r"\bfy\s*ending\s*(\d{4})-\d{2}-\d{2}", t)}
    found |= {f"fy{y}" for y in re.findall(r"\b(?:fy|fiscal(?:\s+year)?)\s*'?(\d{4})\b", t)}
    return found


def _years(text: str) -> set[str]:
    return set(re.findall(r"(?<!\d)(20\d{2})(?!\d)", text))  # also inside "FY2024"


def _labels(text: str) -> set[str]:
    t = _normalize(text)
    found = set()
    for group in _CONFLICT_GROUPS:
        for label in group:
            if any(re.search(alias, t) for alias in _SYNONYMS.get(label, (re.escape(label),))):
                found.add(label)
    if "filtered" in found:
        found.discard("historical")  # "filtered historical simulation" is not historical VaR
    return found


_CORPORATE_SUFFIX = re.compile(r"\b(inc|corp|corporation|co|company|ltd|plc|holdings?|group|class [abc])\b\.?", re.I)


def company_aliases(evidence: list[dict], tickers: list[str]) -> dict[str, set[str]]:
    """Names a model may use instead of a ticker, taken from the tools' own company
    fields: "NVIDIA CORP" -> "NVIDIA", "ADVANCED MICRO DEVICES INC" -> "Advanced Micro Devices"."""
    aliases: dict[str, set[str]] = {t: set() for t in tickers}
    for ev in evidence:
        result = ev.get("result", {})
        ticker, company = str(result.get("ticker", "")).upper(), result.get("company")
        if ticker in aliases and company and company.upper() != ticker:
            name = _CORPORATE_SUFFIX.sub("", company).strip(" ,.")
            if len(name) >= 3:
                aliases[ticker].add(name)
                first = name.split()[0]
                if len(first) >= 4:
                    aliases[ticker].add(first)
    return aliases


def _tickers(text: str, universe: set[str], aliases: dict[str, set[str]] | None = None) -> set[str]:
    found = {t for t in universe if re.search(rf"(?<![A-Za-z]){re.escape(t)}(?![A-Za-z])", text)}
    for ticker, names in (aliases or {}).items():
        if any(re.search(rf"(?<![A-Za-z]){re.escape(n)}(?![A-Za-z])", text, re.I) for n in names):
            found.add(ticker)
    return found


def _claim_sign(text: str, number: NumberClaim) -> int:
    if number.text.lstrip("$€£").startswith(("-", "−")):
        return -1
    before = _normalize(text[max(0, number.start - 40) : number.start])
    words = _WORD_RE.findall(before)[-4:]
    if any(w in _NEGATIVE for w in words):
        return -1
    if any(w in _POSITIVE for w in words):
        return 1
    return 0


def _source_sign(source: str, number: NumberClaim) -> int:
    """Sign of a source value: explicit minus, "up"/"down", or positive for a change
    metric (return, growth, excess, alpha). Loss sizes such as VaR carry no sign."""
    if number.text.lstrip("$€£").startswith(("-", "−")):
        return -1
    words = _WORD_RE.findall(source[max(0, number.start - 12) : number.start].lower())[-1:]
    if words and words[0] == "down":
        return -1
    if words and words[0] == "up":
        return 1
    clause = _clause(source, number).lower()
    if re.search(r"value at risk|shortfall|probability|drawdown|volatility", clause):
        return 0
    return 1 if number.unit == "%" and re.search(r"\b(return|growth|excess|alpha)\b", clause) else 0


def _clause(source: str, number: NumberClaim) -> str:
    """The part of a fact that labels one of its numbers: the clause around it, plus
    the fact's lead label (text before the first colon) - unless a semicolon has
    started a new sub-fact ("3-year beta: 1.15; a 20% SPY fall implies -23%")."""
    left, left_sep = max((source.rfind(sep, 0, number.start), sep) for sep in (",", ";", "(", " vs "))
    right_candidates = [i for i in (source.find(sep, number.end) for sep in (",", ";", ")", " vs ")) if i >= 0]
    right = min(right_candidates) if right_candidates else len(source)
    lead = source.split(":", 1)[0] if ":" in source[: number.start] and not (left >= 0 and left_sep == ";") else ""
    return f"{lead} {source[left + 1 : right]}"


_CLAIM_SEPARATORS = (",", ";", "(", ")", " and ", " while ", " but ", " with ", " whereas ", " vs ", " versus ", ". ")


def _claim_clause(text: str, number: NumberClaim) -> str:
    """The clause of a claim that holds one number, so that a sentence combining
    several facts ("a 14.27% 10-day VaR and revenue of $215.94bn") is checked fact by fact."""
    starts = [text.rfind(sep, 0, number.start) + len(sep) for sep in _CLAIM_SEPARATORS if text.rfind(sep, 0, number.start) >= 0]
    ends = [i for i in (text.find(sep, number.end) for sep in _CLAIM_SEPARATORS) if i >= 0]
    return text[max(starts, default=0) : min(ends, default=len(text))]


def _latest_fy(sources: list[str]) -> str | None:
    years = [y for s in sources for y in re.findall(r"FY ending (\d{4})-\d{2}-\d{2}", s)]
    return max(years) if years else None


# --------------------------------------------------------------------------- checking


def split_claims(text: str) -> list[ClaimCheck]:
    """Bullets are claims; paragraphs are split into sentences. Headings are skipped."""
    claims: list[ClaimCheck] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        line_start = offset + (len(line) - len(line.lstrip()))
        offset += len(line)
        if not stripped or stripped.startswith("#") or stripped.startswith("*Focus"):
            continue
        is_bullet = stripped[:1] in "-*•" and not stripped.startswith("**")
        body = stripped[1:].strip() if is_bullet else stripped
        body_start = line_start + (stripped.find(body) if body else 0)
        parts = [body] if is_bullet else _SENTENCE_RE.split(body)
        cursor = body_start
        for part in parts:
            pos = text.find(part, cursor)
            cursor = pos + len(part)
            cited = _ID_RE.findall(" ".join(_CITATION_RE.findall(part)))
            claims.append(ClaimCheck(text=_CITATION_RE.sub("", part).strip(), cited=cited, start=pos, end=pos + len(part)))
    return claims


def source_strings(evidence: dict) -> list[str]:
    result = evidence.get("result", {})
    if result.get("facts"):
        return list(result["facts"])
    if result.get("passages"):
        meta = [f"{result['filing']} {result.get('url', '')}"] if result.get("filing") else []
        return [p["text"] for p in result["passages"]] + meta
    if "expression" in result and "result" in result:
        return [f"{result['expression']} = {result['result']}"]
    # A raw result without labelled facts: one string of all its fields, nested ones
    # included, so a number keeps its context (period, ticker) for the label checks.
    return ["; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in _flatten(result) if k != "error")]


def _flatten(value, prefix: str = ""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _flatten(v, f"{prefix} {k}".strip())
    elif isinstance(value, list):
        for item in value:
            yield from _flatten(item, prefix)
    elif isinstance(value, (int, float, str)) and not isinstance(value, bool):
        yield prefix, value


def check_claim(
    claim: ClaimCheck,
    sources: dict[str, list[str]],
    universe: set[str],
    multi_company: bool,
    passages: set[str] = frozenset(),
    aliases: dict[str, set[str]] | None = None,
    failed: set[str] = frozenset(),
    benchmarks: set[str] = frozenset({"SPY"}),
) -> None:
    numbers = extract_numbers(claim.text)
    if not claim.cited:
        if numbers:
            claim.kind = "uncited"
            claim.issues.append("numbers without a citation")
        return
    unknown = [c for c in claim.cited if c not in sources]
    failed_ids = [c for c in unknown if c in failed]
    if failed_ids:
        claim.issues.append(f"cites a tool call that failed ({', '.join(failed_ids)})")
    if set(unknown) - set(failed_ids):
        claim.issues.append(f"cites evidence it was not given ({', '.join(c for c in unknown if c not in failed)})")
    pool = [(eid, s) for eid in claim.cited if eid in sources for s in sources[eid]]
    if numbers:
        claim.kind = "numeric"
    elif any(eid in passages for eid in claim.cited):
        claim.kind = "qualitative"
    if not pool:
        return

    words = _content_words(claim.text)
    text_pool = [p for p in pool if p[0] in passages] if claim.kind == "qualitative" else pool
    best = max(text_pool, key=lambda p: len(words & _content_words(p[1])))
    claim.support = len(words & _content_words(best[1])) / len(words) if words else 1.0
    claim.source = best[1]

    if not numbers:
        # Only claims about a filing's text can be checked by wording; a sentence citing
        # numeric facts without quoting any number is interpretation.
        if claim.kind == "qualitative" and claim.support < 0.4:
            claim.issues.append(f"not supported by the cited passage (word overlap {claim.support:.0%})")
        return

    claim_periods, claim_labels = _periods(claim.text), _labels(claim.text)
    claim_tickers, claim_years = _tickers(claim.text, universe, aliases), _years(claim.text)
    clauses = [_claim_clause(claim.text, n) for n in numbers]
    for number, local in zip(numbers, clauses):
        # A benchmark named only in another number's clause is not what this number is about:
        # in "a 14.55% VaR and a 20% SPY fall implying -46.13%", SPY belongs to the stress test.
        elsewhere = _tickers(" ".join(c for c in clauses if c != local), universe, aliases)
        number_tickers = claim_tickers - ((elsewhere & benchmarks) - _tickers(local, universe, aliases))
        hits = []
        for eid, source in pool:
            for candidate in extract_numbers(source):
                if numbers_match(number, candidate.value):
                    hits.append((eid, source, candidate))
        if not hits:
            claim.issues.append(f"{number.text} is not in the cited source{'s' if len(claim.cited) > 1 else ''}")
            continue

        # Among sources that contain the number, prefer the one whose labels the claim keeps.
        def mismatches(hit, number=number, local=local, number_tickers=number_tickers) -> list[str]:
            eid, source, candidate = hit
            problems = []
            clause = _clause(source, candidate)
            src_tickers = _tickers(source, universe, aliases)
            if src_tickers and number_tickers and not (src_tickers & number_tickers):
                problems.append(f"company {', '.join(sorted(number_tickers))} but the source is about {', '.join(sorted(src_tickers))}")
            elif multi_company and src_tickers and not number_tickers:
                problems.append(f"company missing (source: {', '.join(sorted(src_tickers))})")
            covered = set(claim_periods)
            source_fys = sorted(p[2:] for p in _periods(clause) if p.startswith("fy"))
            if len(source_fys) > 1:
                # a span such as "CAGR from FY 2024 to FY 2026": the starting year is part of the label
                if source_fys[0] not in claim_years:
                    problems.append(f"period FY{source_fys[0]}-FY{source_fys[-1]} missing (source: {source.split(':')[0].strip()})")
                covered |= {f"fy{y}" for y in source_fys}
            for period in sorted(_periods(clause) - covered):
                if period.startswith("fy"):
                    # Readers take an undated figure to be the latest year; an older year must be named.
                    year = period[2:]
                    if year in claim_years or year == _latest_fy(sources.get(eid, [])):
                        continue
                    problems.append(f"fiscal year missing (source: FY ending {year}, not the latest)")
                elif period == "ttm":
                    # "Trailing 12 months" only disambiguates where fiscal-year values of the
                    # same metric are also cited (revenue, net income), not for P/E.
                    cited_fy = any("FY ending" in s for _, s in pool)
                    if cited_fy and _labels(clause) & {"revenue", "net income"}:
                        problems.append("period 'trailing 12 months' missing")
                else:
                    problems.append(f"period '{period}' missing")
            for wrong in sorted(_periods(local) - _periods(source)):
                if not wrong.startswith("fy"):
                    problems.append(f"period '{wrong}' is not the source's")
            src_labels, local_labels = _labels(clause), _labels(local)
            for group in _CONFLICT_GROUPS:
                # the number's own clause decides; the whole sentence only when that clause names none
                claim_side = (local_labels & group) or (claim_labels & group)
                src_side = src_labels & group
                if claim_side and src_side and not (claim_side & src_side):
                    problems.append(f"says {'/'.join(sorted(claim_side))} but the source says {'/'.join(sorted(src_side))}")
            want, have = _claim_sign(claim.text, number), _source_sign(source, candidate)
            if want and have and want != have:
                problems.append(f"direction contradicts the source ({candidate.text})")
            return problems

        scored = [(mismatches(h), h) for h in hits]
        problems, (eid, source, _) = min(scored, key=lambda item: len(item[0]))
        if problems:
            claim.source = source
            claim.issues.extend(f"{number.text}: {p}" for p in problems)


def check(text: str, evidence: list[dict], tickers: list[str], allowed_ids: set[str] | None = None) -> AttributionReport:
    """Check every cited claim in ``text``. ``allowed_ids`` restricts which evidence a
    claim may cite (an agent's own); by default any evidence in the ledger counts."""
    in_scope = [e for e in evidence if allowed_ids is None or e["id"] in allowed_ids]
    usable = [e for e in in_scope if not e.get("error")]
    failed = {e["id"] for e in in_scope if e.get("error")}
    sources = {e["id"]: source_strings(e) for e in usable}
    passages = {e["id"] for e in usable if e.get("result", {}).get("passages")}
    universe = set(tickers) | {"SPY"}
    aliases = company_aliases(evidence, tickers)
    claims = split_claims(text)
    for claim in claims:
        check_claim(claim, sources, universe, multi_company=len(tickers) > 1, passages=passages, aliases=aliases, failed=failed,
                    benchmarks=universe - set(tickers))
    # Interpretation (opinions without numbers) is not a claim about the data.
    return AttributionReport([c for c in claims if c.kind != "interpretation" or c.issues])


def annotate(text: str, report: AttributionReport, marker: str = " ⚠") -> str:
    """Append the first issue after each flagged claim."""
    for claim in sorted(report.flagged, key=lambda c: c.end, reverse=True):
        note = f"{marker} {claim.issues[0]}"
        text = text[: claim.end] + note + text[claim.end :]
    return text
