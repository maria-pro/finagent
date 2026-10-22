"""SEC EDGAR: company fundamentals from XBRL "company facts" and 10-K text.

Everything here is free, keyless and authoritative - the numbers come straight
from what companies filed with the SEC, which is what makes the evidence ledger
auditable.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from datetime import date, timedelta

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from ..config import Settings
from .http import DataSourceError, http_for
from .prices import normalize_ticker

_DAY = 86400

# metric -> (candidate XBRL concepts in priority order, unit, kind)
CONCEPTS: dict[str, tuple[list[str], str, str]] = {
    "revenue": (
        [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueGoodsNet",
        ],
        "USD",
        "duration",
    ),
    "gross_profit": (["GrossProfit"], "USD", "duration"),
    "operating_income": (["OperatingIncomeLoss"], "USD", "duration"),
    "net_income": (["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"], "USD", "duration"),
    "eps_diluted": (["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"], "USD/shares", "duration"),
    "rnd": (["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"], "USD", "duration"),
    "operating_cash_flow": (
        ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
        "USD",
        "duration",
    ),
    "capex": (
        [
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets",
            "PaymentsForProceedsFromProductiveAssets",
            "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets",
        ],
        "USD",
        "duration",
    ),
    "cash": (
        ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash"],
        "USD",
        "instant",
    ),
    "marketable_securities": (
        ["MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent", "ShortTermInvestments"],
        "USD",
        "instant",
    ),
    "long_term_debt": (["LongTermDebt", "LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"], "USD", "instant"),
    "equity": (
        ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
        "USD",
        "instant",
    ),
    "total_assets": (["Assets"], "USD", "instant"),
}

_ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}
_PERIODIC_FORMS = _ANNUAL_FORMS | {"10-Q", "10-Q/A"}


@dataclass(frozen=True)
class Fact:
    start: date | None
    end: date
    value: float
    form: str
    filed: str
    accession: str

    @property
    def days(self) -> int:
        return (self.end - self.start).days if self.start else 0


def _d(text: str | None) -> date | None:
    return date.fromisoformat(text) if text else None


# --------------------------------------------------------------------------- lookups


def ticker_to_cik(settings: Settings, ticker: str) -> tuple[int, str]:
    table = http_for(settings).get_json("https://www.sec.gov/files/company_tickers.json", 7 * _DAY)
    wanted = normalize_ticker(ticker)
    for row in table.values():
        if row["ticker"].upper() == wanted:
            return int(row["cik_str"]), row["title"]
    raise DataSourceError(f"{wanted} is not in SEC's company ticker list (ETFs and non-US listings are not covered)")


def company_facts(settings: Settings, cik: int) -> dict:
    return http_for(settings).get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", _DAY)


# --------------------------------------------------------------------------- XBRL facts


def _facts_for(facts: dict, concept: str, unit: str, taxonomy: str = "us-gaap") -> list[Fact]:
    rows = facts.get("facts", {}).get(taxonomy, {}).get(concept, {}).get("units", {}).get(unit, [])
    out = []
    for r in rows:
        if r.get("form") not in _PERIODIC_FORMS:
            continue
        out.append(Fact(_d(r.get("start")), _d(r["end"]), float(r["val"]), r["form"], r.get("filed", ""), r.get("accn", "")))
    return out


def _latest_filed(rows: list[Fact]) -> dict[tuple, Fact]:
    """One fact per period, keeping the most recently filed (restated) value."""
    best: dict[tuple, Fact] = {}
    for f in rows:
        key = (f.start, f.end)
        if key not in best or f.filed > best[key].filed:
            best[key] = f
    return best


def metric_facts(facts: dict, metric: str) -> list[Fact]:
    """Facts for a metric, merged across XBRL concepts.

    Filers switch tags over time (e.g. ``Revenues`` -> ``RevenueFromContract...``),
    so the concept with the most recent data leads and older concepts only fill
    periods it does not cover.
    """
    concepts, unit, _ = CONCEPTS[metric]
    per_concept = [_latest_filed(_facts_for(facts, c, unit)) for c in concepts]
    per_concept = [p for p in per_concept if p]
    if not per_concept:
        return []
    per_concept.sort(key=lambda p: max(f.end for f in p.values()), reverse=True)
    merged: dict[tuple, Fact] = {}
    for periods in per_concept:
        for key, fact in periods.items():
            merged.setdefault(key, fact)
    return sorted(merged.values(), key=lambda f: (f.end, f.days))


def annual_values(facts: dict, metric: str, fiscal_year_ends: list[date] | None = None) -> dict[date, float]:
    """Fiscal-year values keyed by period end date."""
    _, _, kind = CONCEPTS[metric]
    rows = metric_facts(facts, metric)
    if kind == "duration":
        annual = [f for f in rows if f.form in _ANNUAL_FORMS and 340 <= f.days <= 390]
        return {f.end: f.value for f in annual}
    values = {f.end: f.value for f in rows if f.form in _ANNUAL_FORMS}
    if fiscal_year_ends:
        aligned = {}
        for fy_end in fiscal_year_ends:
            near = [d for d in values if abs((d - fy_end).days) <= 7]
            if near:
                aligned[fy_end] = values[min(near, key=lambda d: abs((d - fy_end).days))]
        return aligned
    return values


def ttm_value(facts: dict, metric: str) -> dict | None:
    """Trailing-twelve-month value: last fiscal year + current YTD - prior-year YTD."""
    rows = metric_facts(facts, metric)
    annual = [f for f in rows if 340 <= f.days <= 390]
    if not annual:
        return None
    fy = max(annual, key=lambda f: f.end)
    interim = [f for f in rows if f.form.startswith("10-Q") and f.end > fy.end and 80 <= f.days <= 290]
    if not interim:
        return {"value": fy.value, "period_end": fy.end.isoformat(), "method": "latest fiscal year"}

    latest_end = max(f.end for f in interim)
    ytd = max((f for f in interim if f.end == latest_end), key=lambda f: f.days)
    if ytd.start and abs((ytd.start - (fy.end + timedelta(days=1))).days) > 7:
        return {"value": fy.value, "period_end": fy.end.isoformat(), "method": "latest fiscal year"}
    prior = [
        f
        for f in rows
        if abs((f.end - (ytd.end - timedelta(days=364))).days) <= 10 and abs(f.days - ytd.days) <= 10
    ]
    if not prior:
        return {"value": fy.value, "period_end": fy.end.isoformat(), "method": "latest fiscal year"}
    prior_ytd = max(prior, key=lambda f: f.filed)
    return {
        "value": fy.value + ytd.value - prior_ytd.value,
        "period_end": ytd.end.isoformat(),
        "method": f"FY to {fy.end} + YTD to {ytd.end} - prior-year YTD",
    }


def latest_instant(facts: dict, metric: str) -> tuple[float, date] | None:
    rows = metric_facts(facts, metric)
    if not rows:
        return None
    last = max(rows, key=lambda f: (f.end, f.filed))
    return last.value, last.end


def shares_outstanding(facts: dict) -> tuple[float, date, str] | None:
    """Most recent share count: the 10-K/10-Q cover page, or the balance-sheet or
    diluted weighted-average count when the cover-page tag is missing or stale."""
    candidates = []
    for rank, (taxonomy, concept) in enumerate(
        (
            ("dei", "EntityCommonStockSharesOutstanding"),
            ("us-gaap", "CommonStockSharesOutstanding"),
            ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
        )
    ):
        rows = _facts_for(facts, concept, "shares", taxonomy)
        if not rows:
            continue
        latest_end = max(f.end for f in rows)
        same_day = [f for f in rows if f.end == latest_end]
        latest_filing = max(f.filed for f in same_day)
        values = [f.value for f in same_day if f.filed == latest_filing]
        # Multi-class issuers report one cover-page value per share class on the same date.
        total = sum(values) if taxonomy == "dei" else max(values)
        candidates.append((latest_end, -rank, total, concept))
    if not candidates:
        return None
    newest = max(c[0] for c in candidates)
    # prefer the cover page unless another source is more than ~6 months fresher
    fresh = [c for c in candidates if (newest - c[0]).days <= 190]
    end_date, _, total, concept = max(fresh, key=lambda c: c[1])
    return total, end_date, concept


# --------------------------------------------------------------------------- 10-K text

_SECTION_PATTERNS = {
    "business": r"item\s*1\s*[.:\-–—]?\s*business\b",
    "risk_factors": r"item\s*1a\s*[.:\-–—]?\s*risk\s*factors",
    "unresolved_comments": r"item\s*1b\b",
    "cybersecurity": r"item\s*1c\b",
    "properties": r"item\s*2\s*[.:\-–—]?\s*properties",
    "mda": r"item\s*7\s*[.:\-–—]?\s*management[’'`s\s]*discussion",
    "market_risk": r"item\s*7a\s*[.:\-–—]?\s*quantitative",
    "financial_statements": r"item\s*8\s*[.:\-–—]?\s*financial\s*statements",
}
SECTIONS_OF_INTEREST = ("business", "risk_factors", "mda", "market_risk")


def latest_annual_filing(settings: Settings, cik: int) -> dict:
    subs = http_for(settings).get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", _DAY)
    recent = subs["filings"]["recent"]
    for form in ("10-K", "20-F", "40-F"):
        for i, f in enumerate(recent["form"]):
            if f == form:
                accession = recent["accessionNumber"][i]
                doc = recent["primaryDocument"][i]
                return {
                    "form": form,
                    "accession": accession,
                    "filed": recent["filingDate"][i],
                    "period": recent["reportDate"][i],
                    "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{doc}",
                }
    raise DataSourceError(f"no annual report found for CIK {cik}")


def html_to_text(html: str) -> str:
    with warnings.catch_warnings():
        # Inline XBRL filings are XHTML; the lenient HTML parser handles them fine.
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        tag.decompose()
    text = soup.get_text("\n").replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    lines = [ln for ln in lines if ln and not re.fullmatch(r"(table of contents|\d{1,3})", ln, re.I)]
    return "\n".join(lines)


def split_sections(text: str) -> dict[str, str]:
    """Locate 10-K items. The table of contents also lists every heading, so for
    each item we keep the occurrence that yields the longest section."""
    marks: list[tuple[int, str]] = []
    for key, pattern in _SECTION_PATTERNS.items():
        for m in re.finditer(r"(?im)^\s*" + pattern, text):
            marks.append((m.start(), key))
    marks.sort()
    sections: dict[str, str] = {}
    for key in SECTIONS_OF_INTEREST:
        best = ""
        for i, (pos, k) in enumerate(marks):
            if k != key:
                continue
            end = next((p for p, kk in marks[i + 1 :] if kk != key), len(text))
            if end - pos > len(best):
                best = text[pos:end]
        if len(best) > 500:
            sections[key] = best.strip()
    return sections


def annual_report_sections(settings: Settings, ticker: str) -> tuple[dict, dict[str, str]]:
    cik, _ = ticker_to_cik(settings, ticker)
    filing = latest_annual_filing(settings, cik)
    html = http_for(settings).get_text(filing["url"], -1)  # filings are immutable: cache forever
    text = html_to_text(html)
    sections = split_sections(text) or {"document": text}
    return filing, sections
