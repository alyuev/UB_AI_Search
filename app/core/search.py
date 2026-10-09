"""FTS5, semantic and hybrid (RRF) search over a book.db."""
from __future__ import annotations

import html
import re
import sqlite3
import struct
import time
from typing import Optional

import numpy as np

from app.core.db import _PARA_WITH_TITLES_SELECT, _row_to_para
from app.log import get_logger

logger = get_logger("app")

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

_STOP_RU = set("""
кто что какой какая какое какие каких каким какими какого какому какой чей чья чьё чьи
такой такая такое такие таких таким такими такого такую этот эта это эти этих этим этими этого этой этом эту
тот та то те тех тем тем того той том ту весь вся всё все всех всем всего всей
где куда откуда когда почему зачем как как-то сколько ли ль же бы б ни не нет да или либо но а и
в во на о об обо про при по за из изо от ото до для без безо над под перед между через чрез к ко с со у
ты он она оно мы вы они меня тебя его её ее их нас вас мне тебе ему ей им нам вам мной собой себя себе свой своя своё свои
мой моя моё мои твой наш ваш был была было были будет будут является являются являться
может могут можно нужно надо должен должна должны очень также тоже только еще ещё уже вот вон там тут здесь
если чтобы чтоб что-то кто-то когда-то который которая которое которые которых которым которого которой
какой-то какой-либо ли расскажи объясни скажи такое значит означает
упоминается упоминаются упоминаться упоминалось упоминался упоминалась упоминались упомянут упомянута упомянуты
упомянуто упомянутый упомянуть упомянуться говорится говориться сказано написано встречается встречаются
встречаться существует существуют существовать рассказывается рассказываться речь идёт идет
""".split())

_STOP_EN = set("""
a an the of to in on at by for with from into onto about as is are was were be been being am do does did done doing
have has had having will would shall should can could may might must what who whom whose which when where why how
that this these those there here it its it's i me my we our you your he him his she her they them their
and or but nor so if then than not no yes also very just only
tell explain describe mean means meant say said me please
mentioned mention mentions mentioning says written write writes appears appear appeared occurs occur occurred
exist exists existed talk talks talked tells told
""".split())

_stop_lemmas: Optional[set] = None


def _is_stop(word: str, lemma: Optional[str] = None) -> bool:
    w = word.lower()
    if w in _STOP_EN or w in _STOP_RU:
        return True
    global _stop_lemmas
    if lemma is not None:
        if _stop_lemmas is None:
            m = _get_morph()
            _stop_lemmas = {m.parse(x)[0].normal_form for x in _STOP_RU}
        return lemma.lower() in _stop_lemmas
    return False



def _escape_fts_query(query: str, op: str = " ", stop: bool = True) -> str:
    """Turn arbitrary user input into a safe FTS5 MATCH expression.

    - "quoted phrases" are kept as phrases (inner quotes doubled per FTS5 rules)
    - bare words are individually quoted so operators like OR/AND/NOT, '*', '-'
      and stray punctuation can never be interpreted as FTS5 syntax
    - empty input yields an expression that matches nothing
    """
    query = (query or "").strip()
    if not query:
        return '""'

    tokens: list[str] = []
    bare: list[tuple[int, str]] = []  # (index in tokens, word)
    i = 0
    n = len(query)
    while i < n:
        ch = query[i]
        if ch == '"':
            j = query.find('"', i + 1)
            if j == -1:
                phrase = query[i + 1 :]
                i = n
            else:
                phrase = query[i + 1 : j]
                i = j + 1
            words = _WORD_RE.findall(phrase)
            if words:
                escaped = " ".join(w.replace('"', '""') for w in words)
                tokens.append(f'"{escaped}"')
        else:
            m = _WORD_RE.match(query, i)
            if m:
                w = m.group(0)
                bare.append((len(tokens), w))
                tokens.append('"' + w.replace('"', '""') + '"')
                i = m.end()
            else:
                i += 1

    if stop and bare:
        drop = {ix for ix, w in bare if _is_stop(w)}
        if len(drop) == len(bare) and len(tokens) == len(bare):
            drop = set()  # only stop words and no phrases: keep original terms
        tokens = [t for ix, t in enumerate(tokens) if ix not in drop]
    if not tokens:
        return '""'
    return op.join(tokens)


def _lemmatize_words(text: str, morph) -> list[str]:
    out = []
    for w in _WORD_RE.findall(text):
        try:
            out.append(morph.parse(w)[0].normal_form)
        except Exception:
            out.append(w.lower())
    return out


def _query_lemmas(query: str, stop: bool = True) -> list[str]:
    """Lemmas of the UNQUOTED words, minus stop words (kept if all are stop)."""
    bare = _WORD_RE.findall(re.sub(r'"[^"]*"?', " ", query or ""))
    if not bare:
        return []
    morph = _get_morph()
    pairs = []
    for w in bare:
        try:
            lem = morph.parse(w)[0].normal_form
        except Exception:
            lem = w.lower()
        pairs.append((w, lem))
    kept = [lem for w, lem in pairs if not (stop and _is_stop(w, lem))]
    if kept or '"' in (query or ""):
        return kept
    return [lem for _, lem in pairs]


_morph = None


def _get_morph():
    global _morph
    if _morph is None:
        import pymorphy3

        _morph = pymorphy3.MorphAnalyzer()
    return _morph


def _blob_to_vec(blob: bytes, dtype: str = "f32", scale: Optional[float] = None) -> np.ndarray:
    if dtype == "i8":
        from app.core.db import dequantize_vector

        return dequantize_vector(blob, scale if scale is not None else 1.0)
    n = len(blob) // 4
    return np.array(struct.unpack(f"<{n}f", blob), dtype=np.float32)


def _safe_snippet(raw) -> Optional[str]:
    """FTS5 snippet() with \x01/\x02 markers -> HTML-escaped text with
    <mark></mark>. None if nothing was highlighted."""
    if not raw or "\x01" not in raw:
        return None
    esc = html.escape(raw, quote=True)
    return esc.replace("\x01", "<mark>").replace("\x02", "</mark>")


class Searcher:
    def __init__(self, db_path, embedder=None, fts_weights: Optional[dict] = None,
                 fts_mode_hybrid: Optional[str] = None, fts_mode_standalone: Optional[str] = None,
                 fts_stopwords: Optional[bool] = None):
        from app.core.db import open_db

        self.db_path = db_path
        self.conn: sqlite3.Connection = open_db(db_path, readonly=False)
        self.embedder = embedder
        self.last_timings: dict = {}
        # fts_weights: dict column->weight; None = read from settings per query.
        self.fts_weights: Optional[dict] = fts_weights
        # modes: "and" | "or" | "and_fallback_or". Constructor args override settings.
        try:
            from app.config import load_settings

            st = load_settings()
        except Exception:
            st = {}
        self.fts_stopwords = bool(st.get("fts_stopwords", True) if fts_stopwords is None else fts_stopwords)
        self.fts_mode_hybrid = fts_mode_hybrid or st.get("fts_mode_hybrid") or "or"
        self.fts_mode_standalone = fts_mode_standalone or st.get("fts_mode_standalone") or "and_fallback_or"
        self.fts_titles = True  # match title_* columns when the DB has them
        self._fts_columns: Optional[list[str]] = None

        self._ids: list[str] = []
        self._en_mat: Optional[np.ndarray] = None
        self._ru_mat: Optional[np.ndarray] = None
        if embedder is not None:
            from app.core.db import check_embedder_match

            ok, msg = check_embedder_match(
                self.conn, embedder.model_name, getattr(embedder, "runtime", "onnx")
            )
            if not ok:
                logger.warning(
                    "Searcher(%s): embedder/db mismatch — %s. Semantic/hybrid search "
                    "results will be meaningless (comparing vectors from different "
                    "embedding spaces) until this is fixed.", db_path, msg,
                )
            self._load_vectors(embedder.model_name)

    def with_overrides(self, fts_weights: Optional[dict] = None, fts_mode_hybrid: Optional[str] = None,
                       fts_mode_standalone: Optional[str] = None,
                       fts_stopwords: Optional[bool] = None) -> "Searcher":
        """A light view of this searcher (shares the DB connection and vectors) with its own FTS
        weights/modes/stop-word flag, used for ONE run (exact replay of a saved experiment). The
        original instance and the global settings are not touched."""
        import copy

        view = copy.copy(self)
        view.last_timings = {}
        if fts_weights is not None:
            view.fts_weights = dict(fts_weights)
        if fts_mode_hybrid:
            view.fts_mode_hybrid = fts_mode_hybrid
        if fts_mode_standalone:
            view.fts_mode_standalone = fts_mode_standalone
        if fts_stopwords is not None:
            view.fts_stopwords = bool(fts_stopwords)
        return view

    def _load_vectors(self, model_name: str) -> None:
        t0 = time.time()
        # dtype/scale columns may not exist on an older db (pre-quantization
        # schema); COALESCE lets us read those rows as plain float32 too.
        rows = self.conn.execute(
            "SELECT para_id, lang, vector, "
            "COALESCE(dtype, 'f32') AS dtype, scale "
            "FROM embeddings WHERE model = ?",
            (model_name,),
        ).fetchall()
        by_lang: dict[str, dict[str, np.ndarray]] = {"en": {}, "ru": {}}
        n_i8 = 0
        for r in rows:
            vec = _blob_to_vec(r["vector"], r["dtype"], r["scale"])
            if r["dtype"] == "i8":
                n_i8 += 1
                # Dequantization doesn't exactly preserve unit norm (int8
                # rounding); re-normalize once here so cosine similarity via
                # plain dot product stays correct, rather than per-query.
                norm = np.linalg.norm(vec)
                if norm > 1e-12:
                    vec = vec / norm
            by_lang.setdefault(r["lang"], {})[r["para_id"]] = vec

        ids = sorted(set(by_lang.get("en", {})) | set(by_lang.get("ru", {})))
        self._ids = ids
        dim = self.embedder.dim if self.embedder else (len(next(iter(by_lang["en"].values()))) if by_lang["en"] else 0)
        en_mat = np.zeros((len(ids), dim), dtype=np.float32)
        ru_mat = np.zeros((len(ids), dim), dtype=np.float32)
        for i, pid in enumerate(ids):
            if pid in by_lang.get("en", {}):
                en_mat[i] = by_lang["en"][pid]
            if pid in by_lang.get("ru", {}):
                ru_mat[i] = by_lang["ru"][pid]
        self._en_mat = en_mat
        self._ru_mat = ru_mat
        self._id_to_idx = {pid: i for i, pid in enumerate(ids)}
        logger.info(
            "loaded %d vectors (dim=%d, %d int8-dequantized) in %.2fs",
            len(ids), dim, n_i8, time.time() - t0,
        )

    def _fetch_paras(self, ids: list[str]) -> dict[str, dict]:
        if not ids:
            return {}
        qmarks = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"{_PARA_WITH_TITLES_SELECT} WHERE paragraphs.id IN ({qmarks})", ids
        ).fetchall()
        return {r["id"]: _row_to_para(r) for r in rows}

    # ---- FTS helpers -------------------------------------------------
    def _fts_cols(self) -> list[str]:
        if self._fts_columns is None:
            from app.core.db import fts_columns

            self._fts_columns = fts_columns(self.conn)
        return self._fts_columns

    def _weights(self, cols: list[str]) -> list[float]:
        """Positional bm25 weights for the real FTS columns (UNINDEXED id -> 1)."""
        w = self.fts_weights
        if w is None:
            try:
                from app.config import load_settings

                w = load_settings().get("fts_weights")
            except Exception:
                w = None
        w = w or {}
        return [1.0 if c == "id" else float(w.get(c, 1.0)) for c in cols]

    def fts(self, query: str, k: int = 20, lang: str = "both", mode: Optional[str] = None) -> list[dict]:
        """mode: "and" | "or" | "and_fallback_or" (default = fts_mode_standalone).
        With and_fallback_or, OR is used only if AND found nothing.
        last_timings["fts_mode_used"] is "and" or "or". Quoted phrases stay
        exact phrases in every mode. See _fts_once for result fields."""
        mode = mode or self.fts_mode_standalone
        if mode == "and_fallback_or":
            res = self._fts_once(query, k, lang, "and")
            if res:
                return res
            ms = self.last_timings.get("fts_ms", 0.0)
            res = self._fts_once(query, k, lang, "or")
            self.last_timings["fts_ms"] += ms
            return res
        return self._fts_once(query, k, lang, "or" if mode == "or" else "and")

    def _fts_once(self, query: str, k: int, lang: str, mode: str) -> list[dict]:
        """Weighted-bm25 FTS5 search.

        Extra result fields (None when there is no highlight):
          snippet_en / snippet_ru: ~40-token excerpt, HTML-ESCAPED, with
          <mark>...</mark> around matched terms. Safe to inject as innerHTML:
          the only tags present are <mark>/</mark>.
        """
        t0 = time.time()
        op = " OR " if mode == "or" else " "
        match = _escape_fts_query(query, op, self.fts_stopwords)
        cols = self._fts_cols()
        has_titles = "title_en" in cols and self.fts_titles

        en_cols = ["en"] + (["title_en"] if has_titles else [])
        ru_cols = ["ru", "ru_lemma"] + (["title_ru"] if has_titles and "title_ru" in cols else [])
        lemma_cols = ["ru_lemma"] + (["title_ru"] if has_titles and "title_ru" in cols else [])

        def f(cs, m):
            return "{" + " ".join(cs) + "} : (" + m + ")"

        parts: list[str] = []
        if lang in ("en", "both"):
            parts.append(f(en_cols, match))
        if lang in ("ru", "both"):
            parts.append(f(ru_cols, match))
            lemmas = _query_lemmas(query, self.fts_stopwords)
            if lemmas:
                parts.append(f(lemma_cols, op.join(f'"{w}"' for w in lemmas)))
        fts_query = " OR ".join(parts)
        return self._run_fts(fts_query, k, lang, mode, t0)

    def _run_fts(self, fts_query: str, k: int, lang: str, mode: str, t0: float) -> list[dict]:
        """Run a ready FTS5 MATCH expression with weighted bm25 and build the result rows."""
        cols = self._fts_cols()
        weights = ", ".join(repr(x) for x in self._weights(cols))
        en_idx = cols.index("en") if "en" in cols else 1
        ru_idx = cols.index("ru") if "ru" in cols else 2
        sn = "snippet(paragraphs_fts, {}, '\x01', '\x02', '\u2026', 40)"
        sql = (
            f"SELECT id, bm25(paragraphs_fts, {weights}) AS score, "
            f"{sn.format(en_idx)} AS sn_en, {sn.format(ru_idx)} AS sn_ru "
            "FROM paragraphs_fts WHERE paragraphs_fts MATCH ? ORDER BY score LIMIT ?"
        )
        try:
            rows = self.conn.execute(sql, (fts_query, k)).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning("fts query failed (%r), falling back to empty result: %s", query, e)
            rows = []

        ids = [r["id"] for r in rows]
        by_id = {r["id"]: r for r in rows}
        paras = self._fetch_paras(ids)
        results = []
        for rank, pid in enumerate(ids):
            if pid not in paras:
                continue
            r = by_id[pid]
            item = dict(paras[pid])
            item["score"] = -r["score"]  # bm25: lower is better -> flip sign
            item["fts_rank"] = rank + 1
            item["sem_rank"] = None
            item["match_lang"] = lang
            item["snippet_en"] = _safe_snippet(r["sn_en"])
            item["snippet_ru"] = _safe_snippet(r["sn_ru"])
            results.append(item)
        self.last_timings = {"fts_ms": (time.time() - t0) * 1000, "fts_mode_used": mode}
        return results

    # ---- vocabulary / glossary tools (optional, never required) --------
    @property
    def terms(self):
        """Term tools over this book: vocabulary check, «did you mean», glossary (None if unavailable)."""
        tt = getattr(self, "_term_tools", None)
        if tt is None:
            from app.core.terms import TermTools

            tt = TermTools(self.conn)
            self._term_tools = tt
        return tt

    def fts_terms(self, st, k: int = 20) -> list[dict]:
        """Word search for already prepared terms (exact lemmas, prefixes, phrases in both languages).
        Same result rows as fts()."""
        t0 = time.time()
        return self._run_fts(st.fts_match(self._fts_cols()), k, "both", "or", t0)

    def semantic(self, query: str, k: int = 20) -> list[dict]:
        t0 = time.time()
        if self.embedder is None or self._en_mat is None or len(self._ids) == 0:
            self.last_timings = {"semantic_ms": (time.time() - t0) * 1000}
            return []

        qvec = self.embedder.embed_query(query)
        sim_en = self._en_mat @ qvec
        sim_ru = self._ru_mat @ qvec
        sim = np.maximum(sim_en, sim_ru)
        match_lang = np.where(sim_en >= sim_ru, "en", "ru")

        top_idx = np.argsort(-sim)[:k]
        paras = self._fetch_paras([self._ids[i] for i in top_idx])
        results = []
        for rank, idx in enumerate(top_idx):
            pid = self._ids[idx]
            if pid not in paras:
                continue
            item = dict(paras[pid])
            item["score"] = float(sim[idx])
            item["fts_rank"] = None
            item["sem_rank"] = rank + 1
            item["match_lang"] = match_lang[idx]
            item["snippet_en"] = None
            item["snippet_ru"] = None
            results.append(item)
        self.last_timings = {"semantic_ms": (time.time() - t0) * 1000}
        return results

    def hybrid(self, query: str, k: int = 20, rrf_k: int = 60, cand: int = 50) -> list[dict]:
        t0 = time.time()
        fts_res = self.fts(query, k=cand, lang="both", mode=self.fts_mode_hybrid)
        fts_ms = self.last_timings.get("fts_ms", 0.0)
        fts_mode_used = self.last_timings.get("fts_mode_used")
        sem_res = self.semantic(query, k=cand)
        sem_ms = self.last_timings.get("semantic_ms", 0.0)

        fts_rank = {r["id"]: r["fts_rank"] for r in fts_res}
        sem_rank = {r["id"]: r["sem_rank"] for r in sem_res}
        all_ids = list(dict.fromkeys(list(fts_rank) + list(sem_rank)))

        rrf_score = {}
        for pid in all_ids:
            s = 0.0
            if pid in fts_rank:
                s += 1.0 / (rrf_k + fts_rank[pid])
            if pid in sem_rank:
                s += 1.0 / (rrf_k + sem_rank[pid])
            rrf_score[pid] = s

        ranked_ids = sorted(all_ids, key=lambda pid: -rrf_score[pid])[:k]
        paras = self._fetch_paras(ranked_ids)
        by_id_fts = {r["id"]: r for r in fts_res}
        by_id_sem = {r["id"]: r for r in sem_res}

        results = []
        for pid in ranked_ids:
            if pid not in paras:
                continue
            item = dict(paras[pid])
            item["score"] = rrf_score[pid]
            item["fts_rank"] = fts_rank.get(pid)
            item["sem_rank"] = sem_rank.get(pid)
            lang = None
            if pid in by_id_sem:
                lang = by_id_sem[pid]["match_lang"]
            elif pid in by_id_fts:
                lang = by_id_fts[pid]["match_lang"]
            item["match_lang"] = lang
            f = by_id_fts.get(pid)
            item["snippet_en"] = f["snippet_en"] if f else None
            item["snippet_ru"] = f["snippet_ru"] if f else None
            results.append(item)

        self.last_timings = {
            "fts_ms": fts_ms,
            "fts_mode_used": fts_mode_used,
            "semantic_ms": sem_ms,
            "hybrid_ms": (time.time() - t0) * 1000,
        }
        return results
