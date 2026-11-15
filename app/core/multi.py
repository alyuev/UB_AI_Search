"""Search across several books: one query, per-book searchers, one merged list.

Pure logic (no FastAPI): the server opens the books and hands their searchers here.
"""
from __future__ import annotations

import copy
from typing import Optional


class QueryEmbedCache:
    """Wraps an embedder for ONE multi-book request: the query vector is computed once and reused
    by every book that shares this embedder (same model + runtime). Everything else is delegated."""

    def __init__(self, inner):
        self._inner = inner
        self._cache: dict[str, object] = {}
        self.calls = 0

    def embed_query(self, text: str):
        if text not in self._cache:
            self.calls += 1
            self._cache[text] = self._inner.embed_query(text)
        return self._cache[text]

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _normalise(results: list[dict], mode: str, same_model: bool) -> None:
    """Put the scores of ONE book's hits on a scale that can be compared with other books (in place).

    * semantic: the score is a cosine similarity. With the same embedding model it is already
      comparable between books, so it is kept as is. Different models live in different vector
      spaces (a 0.6 of one is not a 0.6 of the other), so then the best hit of each book becomes 1.0.
    * fts (bm25) and hybrid (RRF): bm25 depends on the statistics of the particular book and RRF is
      built from ranks, so the raw numbers mean nothing across books. The best hit of each book is
      scaled to 1.0 and the others relative to it; the merged list therefore interleaves books by
      how strong a hit is *relative to the best one in its own book*.
    The original value stays in `score_raw`.
    """
    top = max((r.get("score") or 0.0 for r in results), default=0.0)
    for r in results:
        raw = float(r.get("score") or 0.0)
        r["score_raw"] = raw
        if mode == "semantic" and same_model:
            r["score"] = raw
        else:
            r["score"] = raw / top if top > 0 else 0.0


def merge_hits(per_book: list[dict], mode: str, k: int) -> list[dict]:
    """per_book: [{book_id, book_name, model, results}] -> the best k hits of all books by normalised
    score. Every hit gets `book_id` and `book_name`."""
    models = {b.get("model") for b in per_book if b["results"]}
    same_model = len(models) <= 1
    single = sum(1 for b in per_book if b["results"]) <= 1   # one book: its own scores are already a ranking
    merged: list[tuple[float, int, int, dict]] = []
    for bi, b in enumerate(per_book):
        res = [dict(r) for r in b["results"]]
        if single:
            for r in res:
                r["score_raw"] = r.get("score")
        else:
            _normalise(res, mode, same_model)
        for ri, r in enumerate(res):
            r["book_id"], r["book_name"] = b["book_id"], b.get("book_name") or b["book_id"]
            merged.append((-r["score"], bi, ri, r))
    merged.sort(key=lambda t: t[:3])
    return [t[3] for t in merged[:k]]


def search_books(handles: list, query: str, mode: str, k: int) -> tuple[list[dict], dict, list[dict]]:
    """handles: objects with book_id, book_name, model, searcher (and optional `problem`).
    Returns (merged hits, timings, warnings). A book that cannot be searched is skipped with a
    warning instead of failing the whole request."""
    per_book, warnings = [], []
    caches: dict[int, QueryEmbedCache] = {}
    timings: dict = {}
    for h in handles:
        sr = h.searcher
        if sr is None:
            warnings.append({"book_id": h.book_id, "message": getattr(h, "problem", None) or "поисковый индекс недоступен"})
            continue
        if getattr(sr, "embedder", None) is None and mode in ("semantic", "hybrid"):
            warnings.append({"book_id": h.book_id, "message": "поиск по смыслу для этой книги недоступен"
                             + (f" ({h.problem})" if getattr(h, "problem", None) else "")
                             + (" — искали только по словам" if mode == "hybrid" else "")})
            if mode == "semantic":
                continue
        view = copy.copy(sr)      # the shared embedder wrapper must not leak into the cached searcher
        if getattr(sr, "embedder", None) is not None:
            emb = sr.embedder
            view.embedder = caches.setdefault(id(emb), QueryEmbedCache(emb))
        fn = {"hybrid": view.hybrid, "fts": view.fts, "semantic": view.semantic}[mode]
        res = fn(query, k)
        for key, v in getattr(view, "last_timings", {}).items():
            if isinstance(v, (int, float)):
                timings[key] = round(timings.get(key, 0) + v, 3)
        per_book.append({"book_id": h.book_id, "book_name": h.book_name, "model": h.model, "results": res})
    timings["books"] = len(per_book)
    timings["query_embeddings"] = sum(c.calls for c in caches.values())
    return merge_hits(per_book, mode, k), timings, warnings
