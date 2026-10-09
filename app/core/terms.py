"""Vocabulary of the book, «did you mean» hints and the automatic RU<->EN glossary.

Why this exists: a small language model that prepares the search query sometimes REPLACES the
user's own words (e.g. «эдемом» -> «отек»). The book itself is the best judge of which words are
real, so every term (the user's and the model's) is verified against the vocabulary of the book,
unknown words get «did you mean» suggestions, and a glossary built from the parallel EN/RU text
(Dice coefficient over paragraph pairs) translates names and terms in both directions without any LLM.

The glossary lives in the table `term_glossary` of book.db. It is OPTIONAL: nothing here fails
when the table is missing.
"""
from __future__ import annotations

import datetime as _dt
import re
import sqlite3
import threading
from collections import Counter, defaultdict
from typing import Callable, Optional

from app.log import get_logger

logger = get_logger("app")

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_CYR_RE = re.compile(r"[а-яёА-ЯЁ]")

GLOSSARY_VERSION = 1
MIN_PAIR_COUNT = 2          # a pair must meet in at least 2 paragraph pairs (hapaxes: see translit rule)
MAX_DF_FRACTION = 0.2       # words in more than 20% of paragraphs are not terms (stop words, "бог")
PER_RU_TOP = 3
EXPAND_MIN_SCORE = 0.45     # only confident translations are ADDED to the search (all are shown as hints)
MIN_SCORE = 0.2
PAIR_MIN_SCORE = 0.25       # hover highlight: a term counts if its translation is in the same paragraph pair

_TRANSLIT = dict(zip("абвгдежзийклмнопрстуфхцчшщъыьэюяё",
                     ["a", "b", "v", "g", "d", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s",
                      "t", "u", "f", "h", "ts", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya", "e"]))

_SENT_END = (".", "!", "?", ":", "»", '"', "—", "…", "“", "”")


def is_cyrillic(word: str) -> bool:
    return bool(_CYR_RE.search(word or ""))


def translit(word: str) -> str:
    return "".join(_TRANSLIT.get(ch, ch) for ch in (word or "").lower())


# ------------------------------------------------------------ string similarity ---

def levenshtein(a: str, b: str, limit: int = 99) -> int:
    """Edit distance; returns limit+1 as soon as it is certain to exceed `limit`."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        best = i
        for j, cb in enumerate(b, 1):
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            cur.append(v)
            if v < best:
                best = v
        if best > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def jaro_winkler(a: str, b: str) -> float:
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0
    rng = max(0, max(la, lb) // 2 - 1)
    ma, mb = [False] * la, [False] * lb
    m = 0
    for i, ca in enumerate(a):
        for j in range(max(0, i - rng), min(lb, i + rng + 1)):
            if not mb[j] and b[j] == ca:
                ma[i] = mb[j] = True
                m += 1
                break
    if not m:
        return 0.0
    t, k = 0, 0
    for i in range(la):
        if ma[i]:
            while not mb[k]:
                k += 1
            if a[i] != b[k]:
                t += 1
            k += 1
    jaro = (m / la + m / lb + (m - t / 2) / m) / 3
    p = 0
    for x, y in zip(a[:4], b[:4]):
        if x != y:
            break
        p += 1
    return jaro + p * 0.1 * (1 - jaro)


# ------------------------------------------------------------------- helpers ---

def _stop_en() -> set:
    from app.core.search import _STOP_EN
    return _STOP_EN


def _stop_ru() -> set:
    from app.core.search import _STOP_RU
    return _STOP_RU


def _plural(n: int, one: str, few: str, many: str) -> str:
    n10, n100 = n % 10, n % 100
    if n10 == 1 and n100 != 11:
        return one
    if 2 <= n10 <= 4 and not 12 <= n100 <= 14:
        return few
    return many


def paragraphs_word(n: int) -> str:
    return f"{n} {_plural(n, 'абзац', 'абзаца', 'абзацев')}"


def has_glossary_table(conn: sqlite3.Connection) -> bool:
    try:
        return conn.execute("SELECT 1 FROM sqlite_master WHERE name='term_glossary'").fetchone() is not None
    except sqlite3.Error:
        return False


def glossary_info(db_path: str) -> dict:
    """Read-only facts for the «База данных» status card: {"exists", "pairs", "mutual", "built_at"}."""
    out = {"exists": False, "pairs": 0, "mutual": 0, "built_at": None}
    try:
        conn = sqlite3.connect(f"file:{str(db_path).replace(chr(92), '/')}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return out
    try:
        if not has_glossary_table(conn):
            return out
        out["exists"] = True
        out["pairs"] = conn.execute("SELECT COUNT(*) FROM term_glossary").fetchone()[0]
        out["mutual"] = conn.execute("SELECT COUNT(*) FROM term_glossary WHERE mutual = 1").fetchone()[0]
        try:
            r = conn.execute("SELECT built_at FROM term_glossary_meta LIMIT 1").fetchone()
            out["built_at"] = r[0] if r else None
        except sqlite3.Error:
            pass
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return out


# ------------------------------------------------------------- glossary build ---

GLOSSARY_SQL = """
CREATE TABLE term_glossary (
    ru_lemma   TEXT NOT NULL,
    en_term    TEXT NOT NULL,
    score      REAL,       -- Dice coefficient 0..1 (how exclusively the two words appear together)
    df         INTEGER,    -- in how many paragraph pairs both words occur
    df_ru      INTEGER,    -- paragraphs with the Russian word
    df_en      INTEGER,    -- paragraphs with the English word
    mutual     INTEGER,    -- 1 = each is the other's best partner (the most reliable pairs)
    ru_display TEXT,       -- the Russian word as it is usually written (Эдемия / божество)
    en_display TEXT,       -- the English word as it is usually written (Edentia / Adjuster)
    PRIMARY KEY (ru_lemma, en_term)
);
CREATE INDEX idx_term_glossary_en ON term_glossary(en_term);
CREATE TABLE term_glossary_meta (built_at TEXT, paragraphs INTEGER, version INTEGER);
"""


def _clean_token(tok: str) -> str:
    return "".join(ch for ch in tok if ch.isalnum())


def _noop(*_a, **_k):
    return None


def build_glossary(db_path, progress: Optional[Callable] = None, cancel: Optional[threading.Event] = None) -> dict:
    """Build `term_glossary` in the book.db at `db_path` from its parallel paragraphs (EN <-> RU).

    Method: for every Russian lemma (from the lemmatized text) and every English word, count in how many
    paragraph pairs they occur together; the score is the Dice coefficient 2*both/(paragraphs with RU + with EN).
    A pair is kept if the words meet at least twice (words seen once: only if the spelling is clearly the
    same name, e.g. агондонтер / agondonter) and it is among the three best partners of the Russian word.
    «mutual» marks pairs where each word is the other's best partner. The old table is replaced in ONE
    transaction (a cancel or an error leaves the previous glossary intact). Returns statistics.
    """
    progress = progress or _noop
    cancel = cancel or threading.Event()
    conn = sqlite3.connect(str(db_path), timeout=60, isolation_level=None)
    try:
        progress("glossary", 0, 0, None, "чтение абзацев")
        rows = conn.execute("SELECT id, en, ru, ru_lemma FROM paragraphs ORDER BY seq").fetchall()
        n = len(rows)
        if not n:
            raise RuntimeError("В базе нет абзацев — словарь строить не из чего.")
        morph = None
        stop_en, stop_ru = _stop_en(), _stop_ru()
        max_df = n if n < 400 else max(80, int(n * MAX_DF_FRACTION))   # tiny test slices: no cap

        ru_sets: list[list[str]] = []
        en_sets: list[list[str]] = []
        ru_cap: dict[str, list[int]] = defaultdict(lambda: [0, 0])   # lemma -> [capitalized mid-sentence, mid-sentence]
        en_surface: dict[str, Counter] = defaultdict(Counter)
        for i, (pid, en, ru, lem) in enumerate(rows):
            if cancel.is_set():
                raise RuntimeError("cancelled")
            if not lem and ru:
                if morph is None:
                    import pymorphy3
                    morph = pymorphy3.MorphAnalyzer()
                lem = " ".join(morph.parse(c)[0].normal_form for c in (_clean_token(t) for t in ru.split()) if c)
            lemmas = (lem or "").lower().replace("ё", "е").split()   # the index treats «ё» as «е»: one word, not two
            raw = [t for t in (ru or "").split() if _clean_token(t)]
            if len(raw) == len(lemmas):  # aligned: capitalization of the surface word tells proper names
                for k, (rt, lm) in enumerate(zip(raw, lemmas)):
                    if k > 0 and not raw[k - 1].endswith(_SENT_END):
                        c = _clean_token(rt)
                        ru_cap[lm][1] += 1
                        if c[:1].isupper():
                            ru_cap[lm][0] += 1
            ru_sets.append(sorted({lm for lm in lemmas if len(lm) >= 3 and lm.isalpha() and lm not in stop_ru}))
            toks = set()
            for m in _WORD_RE.finditer(en or ""):
                w = m.group(0)
                lw = w.lower()
                if len(lw) < 3 or lw in stop_en:
                    continue
                toks.add(lw)
                en_surface[lw][w] += 1
            en_sets.append(sorted(toks))
            if i % 1000 == 0:
                progress("glossary", i, n * 2, None, "разбор текста")

        ru_df: Counter = Counter()
        en_df: Counter = Counter()
        ru_post: dict[str, list[int]] = defaultdict(list)
        for i in range(n):
            for lm in ru_sets[i]:
                ru_df[lm] += 1
                ru_post[lm].append(i)
            for t in en_sets[i]:
                en_df[t] += 1
        # drop over-frequent words (stop-like) on both sides before counting
        en_ok = [[t for t in s if en_df[t] <= max_df] for s in en_sets]

        cands: dict[str, list[tuple[float, int, str]]] = {}
        best_for_en: dict[str, tuple[float, int, str]] = {}
        lemmas_sorted = sorted(l for l, d in ru_df.items() if d <= max_df)
        total_l = len(lemmas_sorted)
        for li, lm in enumerate(lemmas_sorted):
            if li % 400 == 0:
                if cancel.is_set():
                    raise RuntimeError("cancelled")
                progress("glossary", n + li * n // max(1, total_l), n * 2, None, "подсчёт совпадений слов")
            dr = ru_df[lm]
            cnt: Counter = Counter()
            for p in ru_post[lm]:
                cnt.update(en_ok[p])
            tr = None
            found: list[tuple[float, int, str]] = []
            for e, c in cnt.items():
                de = en_df[e]
                if c >= MIN_PAIR_COUNT:
                    d = 2.0 * c / (dr + de)
                elif dr <= 2 and de <= 2:       # a name seen once: needs the same spelling in both languages
                    if tr is None:
                        tr = translit(lm)
                    if jaro_winkler(tr, e) < 0.85:
                        continue
                    d = 2.0 * c / (dr + de)
                else:
                    continue
                found.append((d, c, e))
                b = best_for_en.get(e)
                if b is None or (d, c) > (b[0], b[1]):
                    best_for_en[e] = (d, c, lm)
            if found:
                found.sort(key=lambda x: (-x[0], -x[1], x[2]))
                cands[lm] = found[:PER_RU_TOP]

        out_rows = []
        mutual_n = 0
        for lm, lst in cands.items():
            top_e = lst[0][2]
            for rank, (d, c, e) in enumerate(lst):
                b = best_for_en.get(e)
                mutual = 1 if (rank == 0 and b is not None and b[2] == lm) else 0
                if d < MIN_SCORE and not mutual:
                    continue
                cap = ru_cap.get(lm)
                ru_disp = lm.capitalize() if cap and cap[1] >= 2 and cap[0] / cap[1] >= 0.5 else lm
                surf = en_surface.get(e)
                en_disp = surf.most_common(1)[0][0] if surf else e
                out_rows.append((lm, e, round(d, 4), c, ru_df[lm], en_df[e], mutual, ru_disp, en_disp))
                mutual_n += mutual
        if cancel.is_set():
            raise RuntimeError("cancelled")
        progress("glossary", n * 2 - 1, n * 2, None, "запись словаря")
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("DROP TABLE IF EXISTS term_glossary")
            conn.execute("DROP TABLE IF EXISTS term_glossary_meta")
            for stmt in [s.strip() for s in GLOSSARY_SQL.split(";") if s.strip()]:
                conn.execute(stmt)
            conn.executemany("INSERT INTO term_glossary VALUES (?,?,?,?,?,?,?,?,?)", out_rows)
            conn.execute("INSERT INTO term_glossary_meta VALUES (?,?,?)",
                         (_dt.datetime.now().isoformat(timespec="seconds"), n, GLOSSARY_VERSION))
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        progress("glossary", n * 2, n * 2, None, f"словарь готов: {len(out_rows)} пар")
        return {"pairs": len(out_rows), "mutual": mutual_n, "paragraphs": n, "ru_terms": len(cands)}
    finally:
        conn.close()


# ------------------------------------------------------------------ run-time ---

class Vocabulary:
    """Document frequencies of the words of the book, read through SQLite fts5vocab (a TEMP table: nothing is
    written to the database file). ru = word forms, lemma = Russian dictionary forms, en = English words."""

    def __init__(self, conn: sqlite3.Connection):
        self.ru: dict[str, int] = {}
        self.lemma: dict[str, int] = {}
        self.en: dict[str, int] = {}
        self.ok = False
        try:
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.vocab_col USING fts5vocab(main, paragraphs_fts, 'col')")
            for term, col, doc in conn.execute("SELECT term, col, doc FROM temp.vocab_col"):
                term = term.replace("ё", "е")      # «ё» and «е» are one letter for the search
                if col == "ru":
                    self.ru[term] = self.ru.get(term, 0) + doc
                elif col == "ru_lemma":
                    self.lemma[term] = self.lemma.get(term, 0) + doc
                elif col == "en":
                    self.en[term] = self.en.get(term, 0) + doc
            self.ok = bool(self.lemma or self.en)
        except sqlite3.Error as exc:
            logger.warning("vocabulary unavailable: %s", exc)
        self._by_first: dict[str, dict[str, list[str]]] = {}

    def _bucket(self, kind: str, first: str) -> list[str]:
        d = self._by_first.get(kind)
        if d is None:
            d = defaultdict(list)
            for w in (self.lemma if kind == "ru" else self.en):
                d[w[:1]].append(w)
            self._by_first[kind] = d
        return d.get(first, [])

    def nearest(self, word: str, lang: str, limit: int = 3) -> list[str]:
        """Real book words that look like `word`: first the words that start the same way, then the ones a
        couple of typos away (edit distance + Jaro-Winkler). Most frequent first."""
        word = word.lower()
        src = self.lemma if lang == "ru" else self.en
        if len(word) < 3 or not src:
            return []
        found: dict[str, tuple] = {}
        stem = word[: max(3, len(word) - 2)]
        if len(word) >= 5:
            for w in self._bucket(lang, word[:1]):
                if w != word and w.startswith(stem) and len(w) <= len(word) + 4:
                    found[w] = (levenshtein(word, w, 9), jaro_winkler(word, w))
        limit_ed = 1 if len(word) <= 4 else 2 if len(word) <= 8 else 3
        for w in self._bucket(lang, word[:1]):
            if w == word or w in found or abs(len(w) - len(word)) > limit_ed:
                continue
            ed = levenshtein(word, w, limit_ed)
            if ed <= limit_ed:
                jw = jaro_winkler(word, w)
                if jw >= SUGGEST_MIN_JW and w[:3] == word[:3]:
                    found[w] = (ed, jw)
        ranked = sorted(found, key=lambda w: (found[w][0], -found[w][1], -src.get(w, 0), w))
        return ranked[:limit]


class Glossary:
    def __init__(self, conn: sqlite3.Connection):
        self.ru2en: dict[str, list[dict]] = defaultdict(list)
        self.en2ru: dict[str, list[dict]] = defaultdict(list)
        self.exists = False
        if not has_glossary_table(conn):
            return
        try:
            for r in conn.execute("SELECT ru_lemma, en_term, score, df, mutual, ru_display, en_display "
                                  "FROM term_glossary ORDER BY score DESC, df DESC"):
                row = {"ru": r[0], "en": r[1], "score": r[2], "df": r[3], "mutual": bool(r[4]),
                       "ru_display": r[5] or r[0], "en_display": r[6] or r[1]}
                self.ru2en[r[0].replace("ё", "е")].append(row)
                self.en2ru[r[1]].append(row)
            self.exists = True
        except sqlite3.Error as exc:
            logger.warning("glossary unreadable: %s", exc)

    @staticmethod
    def _pick(rows: list[dict], limit: int, min_score: float) -> list[dict]:
        if not rows:
            return []
        top = rows[0]["score"] or 0
        out = []
        for r in rows:
            s = r["score"] or 0
            # the best pair always counts if it is reliable; others only when close to it
            if (r is rows[0] and (r["mutual"] or s >= min_score)) or (s >= min_score and s >= 0.6 * top):
                out.append(r)
            if len(out) >= limit:
                break
        return out

    def to_en(self, lemma: str, limit: int = 2, min_score: float = 0.3) -> list[dict]:
        rows = sorted(self.ru2en.get(lemma.replace("ё", "е"), []), key=lambda r: (not r["mutual"], -(r["score"] or 0)))
        return self._pick(rows, limit, min_score)

    def to_ru(self, en: str, limit: int = 2, min_score: float = 0.3) -> list[dict]:
        rows = sorted(self.en2ru.get(en, []), key=lambda r: (not r["mutual"], -(r["score"] or 0)))
        return self._pick(rows, limit, min_score)


class SearchTerms:
    """What goes into the word search: exact lemmas / English words, prefixes (stems) and phrases."""

    def __init__(self):
        self.ru_exact: list[str] = []
        self.ru_prefix: list[str] = []
        self.ru_phrases: list[str] = []
        self.en_exact: list[str] = []
        self.en_prefix: list[str] = []
        self.en_phrases: list[str] = []

    @staticmethod
    def _add(lst: list, v: str) -> None:
        if v and v not in lst:
            lst.append(v)

    def add_ru(self, exact: str = "", prefix: str = "", phrase: str = "") -> None:
        self._add(self.ru_exact, exact)
        self._add(self.ru_prefix, prefix)
        self._add(self.ru_phrases, phrase)

    def add_en(self, exact: str = "", prefix: str = "", phrase: str = "") -> None:
        self._add(self.en_exact, exact)
        self._add(self.en_prefix, prefix)
        self._add(self.en_phrases, phrase)

    def empty(self) -> bool:
        return not any((self.ru_exact, self.ru_prefix, self.ru_phrases, self.en_exact, self.en_prefix, self.en_phrases))

    def is_ru_only(self) -> bool:
        return not (self.en_exact or self.en_prefix or self.en_phrases)

    def to_dict(self) -> dict:
        return {k: list(getattr(self, k)) for k in
                ("ru_exact", "ru_prefix", "ru_phrases", "en_exact", "en_prefix", "en_phrases")}

    def fts_match(self, cols: list[str]) -> str:
        """FTS5 MATCH expression. `cols` = the real columns of paragraphs_fts (title columns may be absent)."""
        parts = []
        ru_c = [c for c in ("ru", "ru_lemma", "title_ru") if c in cols]
        en_c = [c for c in ("en", "title_en") if c in cols]

        def q(w: str) -> str:
            return '"' + w.replace('"', '""') + '"'
        ru_items = [q(w) for w in self.ru_exact] + [q(w) + "*" for w in self.ru_prefix] + \
                   [q(p) for p in self.ru_phrases]
        en_items = [q(w) for w in self.en_exact] + [q(w) + "*" for w in self.en_prefix] + \
                   [q(p) for p in self.en_phrases]
        if ru_items and ru_c:
            parts.append("{" + " ".join(ru_c) + "} : (" + " OR ".join(ru_items) + ")")
        if en_items and en_c:
            parts.append("{" + " ".join(en_c) + "} : (" + " OR ".join(en_items) + ")")
        return " OR ".join(parts) if parts else '""'


SUGGEST_MIN_JW = 0.85        # «did you mean» only for really similar words (same first 3 letters)
FREQUENT_SHARE = 0.02        # a term in more than 2% of paragraphs is too common to select by


def _stem(lemma: str) -> str:
    """Prefix for matching other forms / derived words; only for words of 4+ letters."""
    if len(lemma) < 4:
        return ""
    return lemma[: max(4, len(lemma) - 2)]


class TermTools:
    """Verifies terms against the book, suggests real words, translates with the glossary.

    Needs only a sqlite connection to a book.db (the Searcher's); works without the glossary table."""

    def __init__(self, conn: sqlite3.Connection, count_hits: Optional[Callable[[SearchTerms], int]] = None):
        self.conn = conn
        self._lock = threading.Lock()
        self._vocab: Optional[Vocabulary] = None
        self._gloss: Optional[Glossary] = None
        self._morph = None
        self._count_cache: dict[str, int] = {}
        self._cols: Optional[list[str]] = None

    # -- lazily loaded pieces -------------------------------------------------
    @property
    def vocab(self) -> Vocabulary:
        with self._lock:
            if self._vocab is None:
                self._vocab = Vocabulary(self.conn)
            return self._vocab

    @property
    def glossary(self) -> Glossary:
        with self._lock:
            if self._gloss is None:
                self._gloss = Glossary(self.conn)
            return self._gloss

    def reload(self) -> None:
        with self._lock:
            self._vocab = None
            self._gloss = None
            self._count_cache.clear()

    @property
    def morph(self):
        if self._morph is None:
            import pymorphy3
            self._morph = pymorphy3.MorphAnalyzer()
        return self._morph

    def available(self) -> bool:
        return self.vocab.ok

    # -- word level -------------------------------------------------------------
    def ru_lemma(self, word: str) -> str:
        """Dictionary form of a Russian word; among the possible readings the one that really occurs in the
        book (most often) wins, so «эдемом» becomes «эдем» and not some other word."""
        w = word.lower()
        try:
            forms = [p.normal_form for p in self.morph.parse(w)]
        except Exception:  # noqa: BLE001
            forms = []
        forms = [f.replace("ё", "е") for f in forms]    # the index treats «ё» as «е»: one letter, one word
        forms = list(dict.fromkeys(forms)) or [w.replace("ё", "е")]
        v = self.vocab.lemma
        known = [f for f in forms if f in v]
        if known:
            return max(known, key=lambda f: v[f])
        if w.replace("ё", "е") in v:
            return w.replace("ё", "е")
        return forms[0]

    def _cols_(self) -> list[str]:
        if self._cols is None:
            from app.core.db import fts_columns
            self._cols = fts_columns(self.conn)
        return self._cols

    def count_hits(self, st: SearchTerms) -> int:
        """In how many paragraphs these terms occur (same columns and syntax as the real search)."""
        expr = st.fts_match(self._cols_())
        if expr == '""':
            return 0
        hit = self._count_cache.get(expr)
        if hit is None:
            try:
                hit = self.conn.execute("SELECT COUNT(*) FROM paragraphs_fts WHERE paragraphs_fts MATCH ?",
                                        (expr,)).fetchone()[0]
            except sqlite3.Error:
                hit = 0
            self._count_cache[expr] = hit
        return hit

    def _hits_ru(self, lemma: str, prefix: str = "") -> int:
        st = SearchTerms()
        st.add_ru(exact=lemma, prefix=prefix)
        return self.count_hits(st)

    def _hits_en(self, word: str, prefix: str = "") -> int:
        st = SearchTerms()
        st.add_en(exact=word, prefix=prefix)
        return self.count_hits(st)

    def display_ru(self, lemma: str) -> str:
        g = self.glossary
        rows = g.ru2en.get(lemma.replace("ё", "е"))
        return rows[0]["ru_display"] if rows else lemma

    def display_en(self, token: str) -> str:
        g = self.glossary
        rows = g.en2ru.get(token)
        return rows[0]["en_display"] if rows else token

    def equivalents(self, lang: str, key: str) -> list[dict]:
        """Glossary translations of a word: [{"lang": "en"|"ru", "key", "display", "score"}]."""
        g = self.glossary
        if not g.exists:
            return []
        if lang == "ru":
            return [{"lang": "en", "key": r["en"], "display": r["en_display"], "score": r["score"]}
                    for r in g.to_en(key)]
        return [{"lang": "ru", "key": r["ru"], "display": r["ru_display"], "score": r["score"]}
                for r in g.to_ru(key)]

    @staticmethod
    def _en_forms_like(term: str, forms) -> list[str]:
        """English word forms of a text that are `term` itself or the same word with a short ending."""
        term = term.lower()
        out = [f for f in forms if f == term]
        if not out and len(term) >= 4:
            out = [f for f in forms
                   if (f.startswith(term) or term.startswith(f)) and abs(len(f) - len(term)) <= 2 and len(f) >= 4]
        return out

    def query_marks(self, query: str, ru_texts: list[str], en_texts: list[str]) -> dict:
        """Which words of these texts to highlight for a search query, in both languages:
        {"ru": [word forms], "en": [word forms]} (lowercase, «ё» as «е»). Russian words match by lemma
        (любви = любовь), English ones by form (Adjusters = Adjuster); a word of the query in one language
        also highlights its glossary translations in the other one."""
        stop_ru, stop_en = _stop_ru(), _stop_en()
        ru_lemmas: set = set()
        en_terms: set = set()
        for w in _WORD_RE.findall(query or ""):
            lw = w.lower()
            if len(lw) < 3:
                continue
            if is_cyrillic(lw):
                if lw in stop_ru:
                    continue
                lemma = self.ru_lemma(lw)
                ru_lemmas.add(lemma)
                en_terms.update(e["key"] for e in self.equivalents("ru", lemma))
            elif lw not in stop_en:
                en_terms.add(lw)
                ru_lemmas.update(e["key"].replace("ё", "е") for e in self.equivalents("en", lw))
        out_ru: set = set()
        if ru_lemmas:
            seen: set = set()
            for t in ru_texts:
                for w in _WORD_RE.findall(t or ""):
                    lw = w.lower().replace("ё", "е")
                    if lw in seen or len(lw) < 3:
                        continue
                    seen.add(lw)
                    if self.ru_lemma(lw) in ru_lemmas:
                        out_ru.add(lw)
        out_en: set = set()
        if en_terms:
            forms = {w.lower() for t in en_texts for w in _WORD_RE.findall(t or "") if len(w) >= 3}
            for term in en_terms:
                out_en.update(self._en_forms_like(term, forms))
        return {"ru": sorted(out_ru), "en": sorted(out_en)}

    def paragraph_pairs(self, ru_text: str, en_text: str, min_score: float = PAIR_MIN_SCORE) -> list[dict]:
        """Glossary terms that occur in BOTH halves of one paragraph pair, for the hover highlight in the UI:
        [{"ru": [word forms as written, lowercase], "en": [...], "score"}]. A Russian word is reduced to its
        lemma; its glossary translations are searched among the English words of the same paragraph
        (exact form, or the same word with an ending: Adjuster / Adjusters). Occurring in the paired text
        is itself strong evidence, so a lower score than for search expansion is enough."""
        g = self.glossary
        if not g.exists:
            return []
        stop_ru, stop_en = _stop_ru(), _stop_en()
        en_forms: dict[str, None] = {}
        for t in _WORD_RE.findall(en_text or ""):
            t = t.lower()
            if len(t) >= 3 and t not in stop_en:
                en_forms.setdefault(t)

        def en_matches(term: str) -> list[str]:
            return self._en_forms_like(term, en_forms)

        by_lemma: dict[str, list[str]] = defaultdict(list)
        for w in _WORD_RE.findall(ru_text or ""):
            lw = w.lower()
            if len(lw) < 3 or lw in stop_ru:
                continue
            lemma = self.ru_lemma(lw)
            if lw not in by_lemma[lemma]:
                by_lemma[lemma].append(lw)

        pairs = []
        for lemma, ru_forms in by_lemma.items():
            best = None
            for r in g.ru2en.get(lemma, []):
                s = r["score"] or 0
                if s < min_score:
                    continue
                found = en_matches(r["en"])
                if found and (best is None or (r["mutual"], s) > (best[0]["mutual"], best[0]["score"] or 0)):
                    best = (r, found)
            if best:
                pairs.append({"ru": ru_forms, "en": best[1], "score": round(best[0]["score"] or 0, 3)})
        return pairs

    def suggestions(self, key: str, lang: str) -> list[dict]:
        """«Did you mean»: real book words close to an unknown one, with their translation if the glossary has it."""
        out = []
        for w in self.vocab.nearest(key, lang):
            item = {"lang": lang, "key": w, "display": self.display_ru(w) if lang == "ru" else self.display_en(w)}
            eq = self.equivalents(lang, w)
            if eq:
                item["equiv"] = eq[0]["display"]
            out.append(item)
        return out

    def paragraph_count(self) -> int:
        if getattr(self, "_n_paras", None) is None:
            try:
                self._n_paras = self.conn.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0]
            except sqlite3.Error:
                self._n_paras = 0
        return self._n_paras

    def _is_meta_word(self, w: str) -> bool:
        from app.core.search import _is_stop
        lw = w.lower()
        if not is_cyrillic(lw):
            return _is_stop(lw)
        try:
            lemma = self.morph.parse(lw)[0].normal_form
        except Exception:  # noqa: BLE001
            lemma = lw
        return _is_stop(lw, lemma)

    # -- the user's words -----------------------------------------------------
    def _content_words(self, text: str) -> list[str]:
        from app.core.search import _is_stop
        words = []
        for w in _WORD_RE.findall(text or ""):
            lw = w.lower()
            if len(lw) < 2 or _is_stop(lw):
                continue
            if is_cyrillic(lw):
                try:
                    lemma = self.morph.parse(lw)[0].normal_form
                except Exception:  # noqa: BLE001
                    lemma = lw
                if _is_stop(lw, lemma):
                    continue
            words.append(lw)
        return list(dict.fromkeys(words))

    def analyze_user(self, question: str) -> list[dict]:
        """Content words of the question (stop words removed), each verified in the book."""
        terms = []
        for w in self._content_words(question):
            terms.append(self.check_word(w, src="user"))
        return terms

    def check_word(self, word: str, src: str = "user") -> dict:
        """Verify ONE word (or a short phrase of the model) against the book and look for translations/suggestions."""
        word = word.strip()
        parts = _WORD_RE.findall(word)
        if len(parts) > 1:
            return self._check_phrase(word, parts, src)
        lw = word.lower()
        if is_cyrillic(lw):
            lemma = self.ru_lemma(lw)
            prefix = _stem(lemma) if src == "user" else ""
            exact = self._hits_ru(lemma)
            related = self._hits_ru(lemma, prefix) if prefix else exact
            t = {"text": word, "lang": "ru", "src": src, "key": lemma, "surface": lw, "prefix": prefix,
                 "hits": exact, "hits_related": max(related, exact), "ok": exact > 0,
                 "searchable": max(related, exact) > 0}
            t["display"] = self.display_ru(lemma) if exact else word
        else:
            prefix = _stem(lw) if src == "user" else ""
            exact = self._hits_en(lw)
            related = self._hits_en(lw, prefix) if prefix else exact
            t = {"text": word, "lang": "en", "src": src, "key": lw, "surface": lw, "prefix": prefix,
                 "hits": exact, "hits_related": max(related, exact), "ok": exact > 0,
                 "searchable": max(related, exact) > 0}
            t["display"] = self.display_en(lw) if exact else word
        if t["ok"]:
            t["equiv"] = self.equivalents(t["lang"], t["key"])
        else:
            t["suggest"] = self.suggestions(t["key"], t["lang"])
        return t

    def _check_phrase(self, word: str, parts: list[str], src: str) -> dict:
        st = SearchTerms()
        if is_cyrillic(word):
            lemmas = [self.ru_lemma(p) for p in parts]
            phrase = " ".join(lemmas)
            st.add_ru(phrase=phrase)
            hits = self.count_hits(st)
            return {"text": word, "lang": "ru", "src": src, "key": phrase, "surface": word.lower(), "prefix": "",
                    "hits": hits, "ok": hits > 0, "phrase": True, "display": word}
        phrase = " ".join(p.lower() for p in parts)
        st.add_en(phrase=phrase)
        hits = self.count_hits(st)
        return {"text": word, "lang": "en", "src": src, "key": phrase, "surface": word.lower(), "prefix": "",
                "hits": hits, "ok": hits > 0, "phrase": True, "display": word}

    # -- hints for the model -----------------------------------------------------
    def hints_text(self, user_terms: list[dict]) -> str:
        """The line given to the query-preparation model, e.g.
        «эдем → Eden; эдемия (нет в книге; близкие слова: Эдентия ↔ Edentia)». Empty if there is nothing to say."""
        bits = []
        for t in user_terms:
            if t.get("phrase"):
                continue
            label = t["text"] if t["text"].lower() == t["key"] else f"{t['text']} ({t['key']})"
            if t["ok"]:
                eq = [e["display"] for e in t.get("equiv") or []]
                if eq:
                    bits.append(f"{label} → {', '.join(eq)}")
                else:
                    bits.append(f"{label} → есть в книге ({paragraphs_word(t['hits'])})")
            else:
                sg = t.get("suggest") or []
                if sg:
                    near = ", ".join(s["display"] + (f" ↔ {s['equiv']}" if s.get("equiv") else "") for s in sg)
                    bits.append(f"{label} (нет в книге; близкие слова: {near})")
                else:
                    bits.append(f"{label} (нет в книге; оставь как написал пользователь)")
        return "; ".join(bits)

    # -- merging with the model's plan --------------------------------------------
    def merge(self, user_terms: list[dict], plan: Optional[dict]) -> dict:
        """Combine the user's words, the model's (verified) keywords and glossary translations.

        Returns {"search": SearchTerms, "terms": [term dicts for the status/history], "lines": [str],
        "dropped": [model terms with no hits], "model_all_missing": bool}."""
        st = SearchTerms()
        terms: list[dict] = []
        seen: set = set()

        def use(t: dict) -> None:
            k = (t["lang"], t["key"].replace("ё", "е"))
            if k in seen:
                return
            seen.add(k)
            terms.append(t)

        def add_to_search(t: dict) -> None:
            if not t.get("searchable", t["ok"]):
                return
            if t["lang"] == "ru":
                if t.get("phrase"):
                    st.add_ru(phrase=t["key"])
                else:
                    st.add_ru(exact=t["key"], prefix=t.get("prefix") or "")
            else:
                if t.get("phrase"):
                    st.add_en(phrase=t["key"])
                else:
                    st.add_en(exact=t["key"], prefix=t.get("prefix") or "")

        for t in user_terms:
            use(dict(t))
        model_terms: list[dict] = []
        if plan:
            for w in list(plan.get("keywords_ru") or []) + list(plan.get("keywords_en") or []):
                if not isinstance(w, str) or not w.strip():
                    continue
                if " " not in w.strip() and self._is_meta_word(w):
                    continue
                if (is_cyrillic(w), w.lower()) in [(is_cyrillic(t["text"]), t["text"].lower()) for t in terms]:
                    continue
                mt = self.check_word(w, src="model")
                mt["dropped"] = not mt["ok"]
                model_terms.append(mt)
                use(mt)
        # glossary translations of everything that was found (both directions)
        extra: list[dict] = []
        for t in list(terms):
            if not t["ok"] or t.get("phrase"):
                continue
            for e in t.get("equiv") or []:
                k = (e["lang"], e["key"].replace("ё", "е"))
                if k in seen or (e.get("score") or 0) < EXPAND_MIN_SCORE:
                    continue
                if e["lang"] == "en":
                    hits = self._hits_en(e["key"], _stem(e["key"]))
                    prefix = _stem(e["key"])
                else:
                    hits = self._hits_ru(e["key"], _stem(e["key"]))
                    prefix = _stem(e["key"])
                if not hits:
                    continue
                gt = {"text": e["display"], "lang": e["lang"], "src": "glossary", "key": e["key"],
                      "surface": e["key"], "prefix": prefix, "hits": hits, "ok": True, "display": e["display"],
                      "from": t["display"] if t.get("display") else t["text"]}
                extra.append(gt)
                use(gt)
        model_all_missing = bool(model_terms) and all(not m["ok"] for m in model_terms)
        # terms found in a large share of the book would drown the rare ones in OR mode: leave them out
        okt = [t for t in terms if t.get("searchable", t["ok"])]
        total = self.paragraph_count()
        limit = total * FREQUENT_SHARE if total else 0
        rare_exists = any(t["ok"] and t["hits"] <= limit for t in okt)
        for t in terms:
            if limit and rare_exists and t["ok"] and t["hits"] > limit and t.get("searchable", True):
                t["too_common"] = True
        status_extra = [f"слово «{t.get('display') or t['text']}» слишком частое, не используется для отбора"
                        for t in terms if t.get("too_common")]
        for t in terms:
            if not t.get("too_common"):
                add_to_search(t)
        return {"search": st, "terms": terms, "lines": self.status_lines(terms) + status_extra,
                "dropped": [m["text"] for m in model_terms if not m["ok"]],
                "model_all_missing": model_all_missing}

    @staticmethod
    def status_lines(terms: list[dict]) -> list[str]:
        """Plain-Russian lines for the status panel: ✓ found (with the number of paragraphs) / ✗ not in the book."""
        lines = []
        for t in terms:
            name = t.get("display") or t["text"]
            if t["ok"]:
                extra = ""
                if t["src"] == "glossary":
                    extra = f" · перевод слова «{t.get('from')}»"
                elif t.get("equiv"):
                    extra = " · по-" + ("английски" if t["lang"] == "ru" else "русски") + ": " + \
                            ", ".join(e["display"] for e in t["equiv"])
                rel = t.get("hits_related") or 0
                more = f" (с однокоренными словами: {rel})" if rel > t["hits"] * 1.3 + 2 else ""
                lines.append(f"✓ {name} — {paragraphs_word(t['hits'])}{more}{extra}")
            elif t["src"] == "model":
                lines.append(f"✗ {name} — в книге нет, предложено моделью, не использую")
            else:
                sg = t.get("suggest") or []
                near = ""
                if sg:
                    near = "; возможно: " + ", ".join(s["display"] + (f" ↔ {s['equiv']}" if s.get("equiv") else "")
                                                    for s in sg)
                rel = t.get("hits_related") or 0
                tail = f" (ищу и родственные слова: {paragraphs_word(rel)})" if rel else ""
                lines.append(f"✗ слова «{t['text']}» нет в книге{near}{tail}")
        return lines
