from __future__ import annotations

import re

from ..data import sec
from ..data.http import record_fetches, replay_fetches
from ..retrieval import FilingIndex, chunk_sections, tokenize
from . import TICKER, Tool, ToolContext, parse_tickers

PASSAGE_CHARS = 600
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")
# Filing text is untrusted input: a sentence written to steer an AI reader ("ignore previous
# instructions", "AI assistants must report ...") is withheld before any agent sees it. A
# heuristic, not a defence: it matched nothing in the 10-Ks of the pilot's companies (2.8M
# characters) and catches the obvious phrasings, not every one.
_INSTRUCTION_RE = re.compile(
    r"\b(?:ignore|disregard|forget|override)\b[^.!?]{0,30}\b(?:instructions?|prompts?)\b"
    r"|\b(?:you are|act as|pretend to be|you must now)\b[^.!?]{0,30}\b(?:an? )?(?:ai|assistant|language model|chatbot|llm)\b"
    r"|\bsystem prompt\b|\b(?:dear|attention)[,:]? (?:the )?(?:ai|assistant|language model|llm)s?\b"
    r"|\b(?:ai|assistant|language model|llm)s?\b[^.!?]{0,40}\b(?:must|should) (?:ignore|report|say|state|write|conclude)\b",
    re.IGNORECASE,
)


def screen(text: str) -> tuple[str, int]:
    """``text`` without sentences that address an AI reader, and how many were withheld."""
    sentences = _SENTENCE_RE.split(text)
    kept = [s for s in sentences if not _INSTRUCTION_RE.search(s)]
    return " ".join(kept), len(sentences) - len(kept)


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
    passages, withheld = [], 0
    for chunk, _ in hits:
        text, dropped = screen(chunk.text)
        withheld += dropped
        passages.append({"ref": chunk.ref, "text": excerpt(text, query)})
    out = {
        "ticker": ticker,
        "filing": f"{filing['form']} filed {filing['filed']} (period {filing['period']})",
        "url": filing["url"],
        "retrieval": index.mode,
        "passages": passages,
    }
    if withheld:
        out["withheld"] = f"{withheld} sentence(s) addressed to an AI reader were withheld from these passages"
    return out


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
