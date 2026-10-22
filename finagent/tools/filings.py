from __future__ import annotations

import re

from ..data import sec
from ..data.http import record_fetches, replay_fetches
from ..retrieval import FilingIndex, chunk_sections, tokenize
from . import TICKER, Tool, ToolContext, parse_tickers

PASSAGE_CHARS = 600
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def _index(ctx: ToolContext, ticker: str) -> tuple[dict, FilingIndex]:
    key = ("filing_index", ticker)
    if key not in ctx.cache:
        with record_fetches() as fetched:
            filing, sections = sec.annual_report_sections(ctx.settings, ticker)
            index = FilingIndex(chunk_sections(sections), ctx.settings, filing["accession"])
        ctx.cache[key] = (filing, index, fetched)
    filing, index, fetched = ctx.cache[key]
    replay_fetches(fetched)  # later searches reuse the index; keep the ledger's sources complete
    return filing, index


def excerpt(text: str, query: str, limit: int = PASSAGE_CHARS) -> str:
    """The run of consecutive sentences, at most ``limit`` characters, that best matches the query.

    Chunks are ranked as a whole, so the sentences that made a chunk match can sit
    anywhere in it; cutting the chunk at a fixed length could drop exactly those.
    """
    if len(text) <= limit:
        return text
    sentences = _SENTENCE_RE.split(text)
    terms = set(tokenize(query))
    scores = [len(terms & set(tokenize(s))) for s in sentences]
    best, best_score = (0, 0), -1
    for i in range(len(sentences)):
        size, score = 0, 0
        for j in range(i, len(sentences)):
            if size + len(sentences[j]) > limit and j > i:
                break
            size += len(sentences[j]) + 1
            score += scores[j]
            if score > best_score:
                best, best_score = (i, j), score
    i, j = best
    window = " ".join(sentences[i : j + 1])
    if len(window) > limit:
        window = window[:limit].rsplit(" ", 1)[0]
    return ("... " if i > 0 else "") + window.strip() + (" ..." if j < len(sentences) - 1 else "")


def search_10k(ctx: ToolContext, ticker: str, query: str, k: int = 3) -> dict:
    ticker = parse_tickers(ticker)[0]
    filing, index = _index(ctx, ticker)
    hits = index.search(query, k=max(1, min(int(k), 5)))
    return {
        "ticker": ticker,
        "filing": f"{filing['form']} filed {filing['filed']} (period {filing['period']})",
        "url": filing["url"],
        "retrieval": index.mode,
        "passages": [{"ref": c.ref, "text": excerpt(c.text, query)} for c, _ in hits],
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
