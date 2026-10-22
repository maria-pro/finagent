"""SEC EDGAR: company fundamentals from XBRL "company facts" and 10-K text.

Everything here is free, keyless and authoritative - the numbers come straight
from what companies filed with the SEC, which is what makes the evidence ledger
auditable. Each figure keeps the XBRL concept and the filing (accession number)
it came from.
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


@dataclass(frozen=True)
class Metric:
    concepts: tuple[str, ...]
    unit: str
    kind: str  # "duration" (income/cash-flow) or "instant" (balance sheet)
    # "recency": the concept with the most recent data leads (filers switch revenue tags over time).
    # "priority": per period, the first concept in the list that reports it wins.
    merge: str = "recency"


CONCEPTS: dict[str, Metric] = {
    "revenue": Metric(
        (
            # Total revenues first: when a filer reports both for the latest year, contract revenue
            # alone leaves out insurance premiums, leasing and investment income (Berkshire Hathaway).
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueGoodsNet",
            "RevenuesNetOfInterestExpense",  # broker-dealers' net revenue (e.g. Goldman Sachs)
        ),
        "USD",
        "duration",
    ),
    "gross_profit": Metric(("GrossProfit",), "USD", "duration"),
    "operating_income": Metric(("OperatingIncomeLoss",), "USD", "duration"),
    # Attributable to the parent first; ProfitLoss includes non-controlling interests and is a last resort.
    "net_income": Metric(
        ("NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss"), "USD", "duration", "priority"
    ),
    "eps_diluted": Metric(("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"), "USD/shares", "duration"),
    "rnd": Metric(("ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"), "USD", "duration"),
    "operating_cash_flow": Metric(
        ("NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
        "USD",
        "duration",
    ),
    "capex": Metric(
        (
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets",
            "PaymentsForProceedsFromProductiveAssets",
            "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets",
        ),
        "USD",
        "duration",
    ),
    "cash": Metric(
        ("CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash"),
        "USD",
        "instant",
        "priority",
    ),
    "marketable_securities": Metric(
        ("MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent", "ShortTermInvestments"),
        "USD",
        "instant",
        "priority",
    ),
    # Debt pieces, combined explicitly in balance_sheet(): which tags a filer uses varies.
    "debt_noncurrent": Metric(("LongTermDebtNoncurrent",), "USD", "instant"),
    "debt_current_portion": Metric(("LongTermDebtCurrent",), "USD", "instant"),
    "debt_current_total": Metric(("DebtCurrent",), "USD", "instant"),  # current LTD + short-term borrowings
    "debt_long_term_total": Metric(("LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"), "USD", "instant", "priority"),
    "short_term_borrowings": Metric(("ShortTermBorrowings", "CommercialPaper"), "USD", "instant", "priority"),
    "equity": Metric(
        ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"), "USD", "instant", "priority"
    ),
    "total_assets": Metric(("Assets",), "USD", "instant"),
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
    concept: str = ""

    @property
    def days(self) -> int:
        return (self.end - self.start).days if self.start else 0

    def source(self) -> str:
        return f"{self.concept} ({self.form} {self.accession}, filed {self.filed})"


def _d(text: str | None) -> date | None:
    return date.fromisoformat(text) if text else None


def filing_url(cik: int, accession: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/"


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


def submissions(settings: Settings, cik: int) -> dict:
    return http_for(settings).get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", _DAY)


def company_profile(settings: Settings, cik: int) -> dict:
    """The filer's SEC industry code (SIC), which decides which valuation measures apply."""
    subs = submissions(settings, cik)
    sic = int(subs["sic"]) if str(subs.get("sic") or "").isdigit() else None
    return {"sic": sic, "sic_description": subs.get("sicDescription") or ""}


def is_financial(sic: int | None) -> bool:
    """Banks, lenders, brokers and insurers (SIC 6000-6499). Their debt and cash flows are
    part of operations - deposits, customer funds, trading assets - so enterprise value,
    free cash flow and price-to-sales do not mean for them what they mean for other firms."""
    return sic is not None and 6000 <= sic < 6500


# --------------------------------------------------------------------------- XBRL facts


def _facts_for(facts: dict, concept: str, unit: str, taxonomy: str = "us-gaap") -> list[Fact]:
    rows = facts.get("facts", {}).get(taxonomy, {}).get(concept, {}).get("units", {}).get(unit, [])
    out = []
    for r in rows:
        if r.get("form") not in _PERIODIC_FORMS:
            continue
        out.append(
            Fact(_d(r.get("start")), _d(r["end"]), float(r["val"]), r["form"], r.get("filed", ""), r.get("accn", ""), concept)
        )
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
    """Facts for a metric, merged across its XBRL concepts (see ``Metric.merge``)."""
    spec = CONCEPTS[metric]
    per_concept = [_latest_filed(_facts_for(facts, c, spec.unit)) for c in spec.concepts]
    per_concept = [p for p in per_concept if p]
    if not per_concept:
        return []
    if spec.merge == "recency":
        per_concept.sort(key=lambda p: max(f.end for f in p.values()), reverse=True)
    merged: dict[tuple, Fact] = {}
    for periods in per_concept:
        for key, fact in periods.items():
            merged.setdefault(key, fact)
    return sorted(merged.values(), key=lambda f: (f.end, f.days))


def annual_facts(facts: dict, metric: str, fiscal_year_ends: list[date] | None = None) -> dict[date, Fact]:
    """Fiscal-year facts keyed by period end date."""
    rows = metric_facts(facts, metric)
    if CONCEPTS[metric].kind == "duration":
        return {f.end: f for f in rows if f.form in _ANNUAL_FORMS and 340 <= f.days <= 390}
    if fiscal_year_ends:
        # A year-end balance reappears in the next 10-Qs' comparison column, and the most recently
        # filed copy is often one of those: align by date, whichever periodic filing reports it.
        values = {f.end: f for f in rows}
        aligned = {}
        for fy_end in fiscal_year_ends:
            near = [d for d in values if abs((d - fy_end).days) <= 7]
            if near:
                aligned[fy_end] = values[min(near, key=lambda d: abs((d - fy_end).days))]
        return aligned
    return {f.end: f for f in rows if f.form in _ANNUAL_FORMS}


def annual_values(facts: dict, metric: str, fiscal_year_ends: list[date] | None = None) -> dict[date, float]:
    """Fiscal-year values keyed by period end date."""
    return {end: f.value for end, f in annual_facts(facts, metric, fiscal_year_ends).items()}


def ttm_value(facts: dict, metric: str) -> dict | None:
    """Trailing-twelve-month value: last fiscal year + current YTD - prior-year YTD."""
    rows = metric_facts(facts, metric)
    annual = [f for f in rows if 340 <= f.days <= 390]
    if not annual:
        return None
    fy = max(annual, key=lambda f: f.end)
    fallback = {
        "value": fy.value,
        "period_end": fy.end.isoformat(),
        "method": "latest fiscal year",
        "concept": fy.concept,
        "accessions": [fy.accession],
    }
    interim = [f for f in rows if f.form.startswith("10-Q") and f.end > fy.end and 80 <= f.days <= 290]
    if not interim:
        return fallback

    latest_end = max(f.end for f in interim)
    ytd = max((f for f in interim if f.end == latest_end), key=lambda f: f.days)
    if ytd.start and abs((ytd.start - (fy.end + timedelta(days=1))).days) > 7:
        return fallback
    prior = [
        f
        for f in rows
        if abs((f.end - (ytd.end - timedelta(days=364))).days) <= 10 and abs(f.days - ytd.days) <= 10
    ]
    if not prior:
        return fallback
    prior_ytd = max(prior, key=lambda f: f.filed)
    return {
        "value": fy.value + ytd.value - prior_ytd.value,
        "period_end": ytd.end.isoformat(),
        "method": f"FY to {fy.end} + YTD to {ytd.end} - prior-year YTD",
        "concept": ytd.concept if ytd.concept == fy.concept else f"{fy.concept} / {ytd.concept}",
        "accessions": sorted({fy.accession, ytd.accession, prior_ytd.accession}),
    }


def latest_instant(facts: dict, metric: str) -> Fact | None:
    rows = metric_facts(facts, metric)
    return max(rows, key=lambda f: (f.end, f.filed)) if rows else None


def instant_at(facts: dict, metric: str, on: date, tolerance_days: int = 5) -> Fact | None:
    """The balance-sheet fact for ``metric`` dated ``on`` (within a few days), if reported."""
    near = [f for f in metric_facts(facts, metric) if abs((f.end - on).days) <= tolerance_days]
    return max(near, key=lambda f: f.filed) if near else None


def balance_sheet(facts: dict) -> dict:
    """Cash, current securities and total debt on one balance-sheet date.

    The date is the latest one on which cash is reported, and every other item is
    taken from that same date, so enterprise value never mixes quarters. Debt is
    assembled explicitly because filers tag it differently: noncurrent long-term
    debt plus whichever current-debt tags exist, or ``LongTermDebt`` (which by the
    taxonomy's definition already includes the current portion) plus short-term
    borrowings. Lease liabilities are not included. When no debt tag is reported, debt
    is None, not zero: a missing tag says nothing about whether the company borrows.
    """
    cash = latest_instant(facts, "cash")
    if cash is None:
        return {"as_of": None, "cash": None, "securities": None, "debt": None, "debt_method": "no balance sheet in XBRL", "sources": []}
    on = cash.end
    securities = instant_at(facts, "marketable_securities", on)
    pick = {m: instant_at(facts, m, on) for m in ("debt_noncurrent", "debt_current_portion", "debt_current_total", "debt_long_term_total", "short_term_borrowings")}

    parts: list[Fact] = []
    if pick["debt_noncurrent"] and pick["debt_current_total"]:
        parts = [pick["debt_noncurrent"], pick["debt_current_total"]]
    elif pick["debt_noncurrent"] and pick["debt_current_portion"]:
        parts = [pick["debt_noncurrent"], pick["debt_current_portion"]] + [p for p in [pick["short_term_borrowings"]] if p]
    elif pick["debt_long_term_total"]:
        parts = [pick["debt_long_term_total"]] + [p for p in [pick["short_term_borrowings"]] if p]
    elif pick["debt_noncurrent"]:
        parts = [pick["debt_noncurrent"]] + [p for p in [pick["short_term_borrowings"]] if p]
    elif pick["short_term_borrowings"]:
        parts = [pick["short_term_borrowings"]]

    if parts:
        debt = sum(p.value for p in parts)
        method = " + ".join(p.concept for p in parts) + f" as of {on}"
    else:
        debt, method = None, f"no debt tags reported as of {on}"
    used = [cash] + ([securities] if securities else []) + parts
    return {
        "as_of": on.isoformat(),
        "cash": cash.value,
        "securities": securities.value if securities else 0.0,
        "debt": debt,
        "debt_method": method,
        "sources": [f.source() for f in used],
    }


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
    recent = submissions(settings, cik)["filings"]["recent"]
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
    raise DataSourceError(
        f"no 10-K, 20-F or 40-F among the recent filings of CIK {cik} "
        "(a newly registered successor company files only 10-Qs until its first annual report)"
    )


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
