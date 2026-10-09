"""Fake app.core.search for tests: simple substring FTS + fake semantic scoring."""
from __future__ import annotations

from . import db as fake_db


class Searcher:
    def __init__(self, db_path: str, embedder=None):
        self.db_path = db_path
        self.embedder = embedder
        self.conn = fake_db.open_db(db_path, readonly=True)
        self.last_timings = {}

    def _all(self):
        rows = self.conn.execute("SELECT * FROM paragraphs").fetchall()
        return [fake_db._row_to_para(r) for r in rows]

    def fts(self, query, k=10):
        import time
        t0 = time.time()
        q = (query or "").lower()
        results = []
        for p in self._all():
            en_hit = q in (p["en"] or "").lower()
            ru_hit = q in (p["ru"] or "").lower()
            if en_hit or ru_hit:
                r = dict(p)
                r["score"] = 1.0
                r["fts_rank"] = 1
                r["sem_rank"] = None
                r["match_lang"] = "en" if en_hit else "ru"
                results.append(r)
        self.last_timings = {"fts_ms": (time.time() - t0) * 1000}
        return results[:k]

    def semantic(self, query, k=10):
        import time
        t0 = time.time()
        results = []
        for p in self._all():
            r = dict(p)
            r["score"] = 0.5
            r["fts_rank"] = None
            r["sem_rank"] = 1
            r["match_lang"] = "en"
            results.append(r)
        self.last_timings = {"sem_ms": (time.time() - t0) * 1000}
        return results[:k]

    def hybrid(self, query, k=10):
        import time
        t0 = time.time()
        fts_res = self.fts(query, k)
        if fts_res:
            out = fts_res
        else:
            out = self.semantic(query, k)
        self.last_timings = {"hybrid_ms": (time.time() - t0) * 1000}
        return out[:k]
