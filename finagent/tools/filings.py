from __future__ import annotations

from ..data import sec
from ..retrieval import FilingIndex, chunk_sections
from . import TICKER, Tool, ToolContext, parse_tickers

PASSAGE_CHARS = 550


def _index(ctx: ToolContext, ticker: str) -> tuple[dict, FilingIndex]:
    key = ("filing_index", ticker)
    if key not in ctx.cache:
        filing, sections = sec.annual_report_sections(ctx.settings, ticker)
        ctx.cache[key] = (filing, FilingIndex(chunk_sections(sections), ctx.settings, filing["accession"]))
    return ctx.cache[key]


def _trim(text: str, limit: int = PASSAGE_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text.rfind(". ", 0, limit)
    return text[: cut + 1 if cut > limit // 2 else limit].rstrip() + " ..."


def search_10k(ctx: ToolContext, ticker: str, query: str, k: int = 3) -> dict:
    ticker = parse_tickers(ticker)[0]
    filing, index = _index(ctx, ticker)
    hits = index.search(query, k=max(1, min(int(k), 5)))
    return {
        "ticker": ticker,
        "filing": f"{filing['form']} filed {filing['filed']} (period {filing['period']})",
        "url": filing["url"],
        "retrieval": index.mode,
        "passages": [{"ref": c.ref, "text": _trim(c.text)} for c, _ in hits],
    }


TOOLS = [
    Tool(
        "search_10k",
        "Search the company's latest annual report (10-K): business, risk factors, management discussion.",
        {
            "type": "object",
            "properties": {
                "ticker": TICKER,
                "query": {"type": "string", "description": "What to look for, e.g. 'supply chain concentration risk'"},
            },
            "required": ["ticker", "query"],
        },
        search_10k,
    ),
]
