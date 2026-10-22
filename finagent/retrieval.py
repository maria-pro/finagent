"""Retrieval over 10-K text.

BM25 runs anywhere with no model at all, which is what the CPU profile uses.
When an embedding model is configured (NeMo Retriever on the NVIDIA API
catalog, or a self-hosted embedding NIM) dense similarity is added and the two
rankings are merged with reciprocal rank fusion.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .config import Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with we our "
    "us may could can such any other these those which their they been also not than into more".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


@dataclass
class Chunk:
    ref: str
    section: str
    text: str


def chunk_sections(sections: dict[str, str], target_chars: int = 900, max_chars: int = 1400) -> list[Chunk]:
    """Pack paragraphs into ~``target_chars`` chunks without splitting sentences mid-paragraph."""
    chunks: list[Chunk] = []
    for section, text in sections.items():
        buf: list[str] = []
        size = 0
        n = 0
        for para in (p.strip() for p in text.split("\n")):
            if len(para) < 40 and not para.endswith((".", ":")):
                # short fragments are usually headings; keep them attached to what follows
                if para:
                    buf.append(para)
                    size += len(para)
                continue
            while len(para) > max_chars:
                cut = para.rfind(". ", 0, max_chars) + 1 or max_chars
                buf.append(para[:cut])
                chunks.append(Chunk(f"{section}#{n}", section, " ".join(buf)))
                n += 1
                buf, size, para = [], 0, para[cut:].strip()
            buf.append(para)
            size += len(para)
            if size >= target_chars:
                chunks.append(Chunk(f"{section}#{n}", section, " ".join(buf)))
                n += 1
                buf, size = [], 0
        if size > 200:
            chunks.append(Chunk(f"{section}#{n}", section, " ".join(buf)))
    return chunks


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.lengths = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = float(self.lengths.mean()) if len(docs) else 0.0
        df = Counter(term for d in self.docs for term in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for i, (doc, length) in enumerate(zip(self.docs, self.lengths)):
            s = 0.0
            for term in query:
                tf = doc.get(term)
                if tf:
                    s += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avgdl))
            out[i] = s
        return out


class FilingIndex:
    """Hybrid (BM25 + optional dense) index over one filing's chunks."""

    def __init__(self, chunks: list[Chunk], settings: Settings, cache_key: str):
        self.chunks = chunks
        self.settings = settings
        self.bm25 = BM25([tokenize(c.text) for c in chunks])
        self._dense: np.ndarray | None = None
        self.mode = "BM25"
        if settings.embed_model:
            try:
                self._dense = self._load_or_embed(cache_key)
                self.mode = f"BM25 + {settings.embed_model} (RRF)"
            except Exception as err:  # embeddings are an enhancement, never a hard dependency
                self.mode = f"BM25 (embeddings unavailable: {type(err).__name__})"

    def _embed(self, texts: list[str], input_type: str) -> np.ndarray:
        from openai import OpenAI

        client = OpenAI(base_url=self.settings.base_url, api_key=self.settings.api_key or "not-needed", timeout=60)
        vectors = []
        for i in range(0, len(texts), 32):
            resp = client.embeddings.create(
                model=self.settings.embed_model,
                input=texts[i : i + 32],
                encoding_format="float",
                extra_body={"input_type": input_type, "truncate": "END"},
            )
            vectors.extend(d.embedding for d in resp.data)
        arr = np.asarray(vectors, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)

    def _load_or_embed(self, cache_key: str) -> np.ndarray:
        digest = hashlib.sha256(f"{self.settings.embed_model}|{cache_key}|{len(self.chunks)}".encode()).hexdigest()[:24]
        path = self.settings.cache_dir / "embeddings" / f"{digest}.npy"
        if path.exists():
            return np.load(path)
        arr = self._embed([c.text for c in self.chunks], "passage")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
        return arr

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        bm = self.bm25.scores(tokenize(query))
        ranks = [np.argsort(-bm)]
        if self._dense is not None:
            q = self._embed([query], "query")[0]
            ranks.append(np.argsort(-(self._dense @ q)))
        fused = np.zeros(len(self.chunks))
        for order in ranks:
            for rank, idx in enumerate(order):
                fused[idx] += 1.0 / (60 + rank)
        top = np.argsort(-fused)[:k]
        return [(self.chunks[i], float(fused[i])) for i in top if bm[i] > 0 or self._dense is not None]
