"""SQLite schema and accessors for a book.db (paragraphs + FTS5 + embeddings)."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS books (
    book_id TEXT PRIMARY KEY,
    book_name TEXT,
    language_source TEXT,
    language_translation TEXT,
    source_file TEXT,
    created_at TEXT,
    embedding_model TEXT,
    embedding_dim INTEGER,
    index_scheme TEXT,
    embedding_runtime TEXT,
    vector_dtype TEXT DEFAULT 'f32'
);

CREATE TABLE IF NOT EXISTS paragraphs (
    id TEXT PRIMARY KEY,
    book_id TEXT,
    seq INTEGER,
    paper INTEGER,
    section INTEGER,
    paragraph INTEGER,
    en TEXT,
    ru TEXT,
    ru_lemma TEXT,
    en_lemma TEXT,
    chunk_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_paragraphs_book_seq ON paragraphs(book_id, seq);
CREATE INDEX IF NOT EXISTS idx_paragraphs_paper_section ON paragraphs(paper, section, paragraph);

CREATE VIRTUAL TABLE IF NOT EXISTS paragraphs_fts USING fts5(
    id UNINDEXED,
    en,
    ru,
    ru_lemma,
    title_en,
    title_ru,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS embeddings (
    para_id TEXT,
    lang TEXT,
    model TEXT,
    vector BLOB,
    dtype TEXT DEFAULT 'f32',
    scale REAL,
    PRIMARY KEY (para_id, lang, model)
);

CREATE TABLE IF NOT EXISTS parts (
    part INTEGER PRIMARY KEY,
    en TEXT,
    ru TEXT,
    first_paper INTEGER,
    last_paper INTEGER
);

CREATE TABLE IF NOT EXISTS papers (
    paper INTEGER PRIMARY KEY,
    en TEXT,
    ru TEXT,
    part INTEGER
);

CREATE TABLE IF NOT EXISTS sections (
    paper INTEGER,
    section INTEGER,
    en TEXT,
    ru TEXT,
    PRIMARY KEY (paper, section)
);
"""


# Column order of paragraphs_fts for NEW builds. bm25() weights are positional.
# Old databases lack title_en/title_ru; use fts_columns(conn) to detect.
FTS_COLUMNS = ["id", "en", "ru", "ru_lemma", "title_en", "title_ru"]

FTS_CREATE_SQL = """
CREATE VIRTUAL TABLE {name} USING fts5(
    id UNINDEXED,
    en,
    ru,
    ru_lemma,
    title_en,
    title_ru,
    tokenize = 'unicode61 remove_diacritics 2'
)
"""


def fts_columns(conn: sqlite3.Connection) -> list[str]:
    """Actual column names of paragraphs_fts, in positional order."""
    try:
        return [r[1] for r in conn.execute("PRAGMA table_info(paragraphs_fts)").fetchall()]
    except sqlite3.Error:
        return []


class DictRow(sqlite3.Row):
    def keys(self):  # pragma: no cover - convenience only
        return super().keys()


def quantize_vector(vec) -> tuple[bytes, float]:
    """Symmetric per-vector int8 quantization: scale = max|x|/127,
    q = round(x/scale). Returns (int8 bytes, scale). scale=0 (all-zero
    vector) is stored as scale=1.0 with an all-zero blob to avoid div by 0.
    """
    import numpy as np

    arr = np.asarray(vec, dtype=np.float32)
    max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
    if max_abs == 0.0:
        return (np.zeros(arr.shape, dtype=np.int8)).tobytes(), 1.0
    scale = max_abs / 127.0
    q = np.round(arr / scale).clip(-127, 127).astype(np.int8)
    return q.tobytes(), scale


def dequantize_vector(blob: bytes, scale: float):
    import numpy as np

    q = np.frombuffer(blob, dtype=np.int8)
    return (q.astype(np.float32)) * scale


def open_db(path: str | Path, readonly: bool = False) -> sqlite3.Connection:
    path = Path(path)
    if readonly:
        uri = f"file:{path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if not readonly:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    return conn


_PARA_WITH_TITLES_SELECT = """
SELECT paragraphs.*,
       p.en AS paper_title_en, p.ru AS paper_title_ru,
       s.en AS section_title_en, s.ru AS section_title_ru
FROM paragraphs
LEFT JOIN papers p ON p.paper = paragraphs.paper
LEFT JOIN sections s ON s.paper = paragraphs.paper AND s.section = paragraphs.section
"""


def _row_to_para(row: sqlite3.Row) -> dict:
    keys = row.keys()
    out = {
        "id": row["id"],
        "paper": row["paper"],
        "section": row["section"],
        "paragraph": row["paragraph"],
        "en": row["en"],
        "ru": row["ru"],
    }
    if "paper_title_en" in keys and (row["paper_title_en"] is not None or row["paper_title_ru"] is not None):
        out["paper_title"] = {"en": row["paper_title_en"], "ru": row["paper_title_ru"]}
    if "section_title_en" in keys and (row["section_title_en"] is not None or row["section_title_ru"] is not None):
        out["section_title"] = {"en": row["section_title_en"], "ru": row["section_title_ru"]}
    return out


def get_book_info(conn: sqlite3.Connection, book_id: Optional[str] = None) -> Optional[dict]:
    if book_id:
        row = conn.execute("SELECT * FROM books WHERE book_id = ?", (book_id,)).fetchone()
    else:
        row = conn.execute("SELECT * FROM books LIMIT 1").fetchone()
    return dict(row) if row else None


def check_embedder_match(conn: sqlite3.Connection, model_name: str, runtime: str) -> tuple[bool, str]:
    """Compare the embedder about to be used for search/indexing against
    what this book.db's `books` row records it was built with. Query
    vectors from a different model/runtime are not comparable to the
    stored vectors (different vector space), so callers should warn/refuse
    on mismatch rather than silently return nonsense similarity scores.
    """
    info = get_book_info(conn)
    if not info:
        return False, "no book metadata found in this database"
    stored_model = info.get("embedding_model")
    stored_runtime = info.get("embedding_runtime")
    if stored_model != model_name:
        return False, (
            f"embedding model mismatch: db was built with {stored_model!r}, "
            f"but embedder is {model_name!r}"
        )
    if stored_runtime and runtime and stored_runtime != runtime:
        return False, (
            f"embedding runtime mismatch: db was built with {stored_runtime!r}, "
            f"but embedder runtime is {runtime!r} (vectors should still be numerically "
            f"close if both are correctness-verified, but this is not guaranteed)"
        )
    return True, "ok"


def get_paragraph(conn: sqlite3.Connection, id_: str) -> Optional[dict]:
    row = conn.execute(_PARA_WITH_TITLES_SELECT + " WHERE paragraphs.id = ?", (id_,)).fetchone()
    return _row_to_para(row) if row else None


def get_context(conn: sqlite3.Connection, id_: str, before: int = 2, after: int = 2) -> list:
    center = conn.execute("SELECT book_id, seq FROM paragraphs WHERE id = ?", (id_,)).fetchone()
    if not center:
        return []
    book_id, seq = center["book_id"], center["seq"]
    rows = conn.execute(
        _PARA_WITH_TITLES_SELECT
        + " WHERE paragraphs.book_id = ? AND paragraphs.seq BETWEEN ? AND ? ORDER BY paragraphs.seq",
        (book_id, seq - before, seq + after),
    ).fetchall()
    return [_row_to_para(r) for r in rows]


def get_range(
    conn: sqlite3.Connection,
    paper: int,
    section: Optional[int] = None,
    para_from: Optional[int] = None,
    para_to: Optional[int] = None,
) -> list:
    q = _PARA_WITH_TITLES_SELECT + " WHERE paragraphs.paper = ?"
    args: list = [paper]
    if section is not None:
        q += " AND paragraphs.section = ?"
        args.append(section)
    if para_from is not None:
        q += " AND paragraphs.paragraph >= ?"
        args.append(para_from)
    if para_to is not None:
        q += " AND paragraphs.paragraph <= ?"
        args.append(para_to)
    q += " ORDER BY paragraphs.seq"
    rows = conn.execute(q, args).fetchall()
    return [_row_to_para(r) for r in rows]


def get_titles(conn: sqlite3.Connection, paper: int, section: Optional[int] = None) -> dict:
    """Return {paper_en, paper_ru, section_en, section_ru, part, part_en, part_ru}
    for the given paper (and optionally section). Missing titles yield None
    values rather than raising.
    """
    out = {
        "paper_en": None, "paper_ru": None,
        "section_en": None, "section_ru": None,
        "part": None, "part_en": None, "part_ru": None,
    }
    prow = conn.execute("SELECT * FROM papers WHERE paper = ?", (paper,)).fetchone()
    if prow:
        out["paper_en"] = prow["en"]
        out["paper_ru"] = prow["ru"]
        out["part"] = prow["part"]
        if prow["part"] is not None:
            partrow = conn.execute("SELECT * FROM parts WHERE part = ?", (prow["part"],)).fetchone()
            if partrow:
                out["part_en"] = partrow["en"]
                out["part_ru"] = partrow["ru"]
    if section is not None:
        srow = conn.execute(
            "SELECT * FROM sections WHERE paper = ? AND section = ?", (paper, section)
        ).fetchone()
        if srow:
            out["section_en"] = srow["en"]
            out["section_ru"] = srow["ru"]
    return out


def list_papers(conn: sqlite3.Connection) -> list:
    rows = conn.execute(
        "SELECT DISTINCT paragraphs.paper AS paper, p.en AS en, p.ru AS ru "
        "FROM paragraphs LEFT JOIN papers p ON p.paper = paragraphs.paper "
        "ORDER BY paragraphs.paper"
    ).fetchall()
    out = []
    for r in rows:
        item = {"paper": r["paper"]}
        if r["en"] is not None or r["ru"] is not None:
            item["title"] = {"en": r["en"], "ru": r["ru"]}
        out.append(item)
    return out


def list_sections(conn: sqlite3.Connection, paper: int) -> list:
    rows = conn.execute(
        "SELECT DISTINCT paragraphs.section AS section, s.en AS en, s.ru AS ru "
        "FROM paragraphs LEFT JOIN sections s ON s.paper = paragraphs.paper AND s.section = paragraphs.section "
        "WHERE paragraphs.paper = ? ORDER BY paragraphs.section",
        (paper,),
    ).fetchall()
    out = []
    for r in rows:
        item = {"section": r["section"]}
        if r["en"] is not None or r["ru"] is not None:
            item["title"] = {"en": r["en"], "ru": r["ru"]}
        out.append(item)
    return out


def get_toc(conn: sqlite3.Connection) -> list:
    """Table of contents for the reading page, in one query per level:
    [{"part", "title": {en, ru} | None, "papers": [{"paper", "title", "sections": [{"section", "title"}]}]}].
    Only papers/sections that really have paragraphs are listed; titles may be None (books without them)."""
    def title(r):
        return {"en": r["en"], "ru": r["ru"]} if (r["en"] is not None or r["ru"] is not None) else None

    secs: dict = {}
    for r in conn.execute(
        "SELECT DISTINCT paragraphs.paper AS paper, paragraphs.section AS section, s.en AS en, s.ru AS ru "
        "FROM paragraphs LEFT JOIN sections s ON s.paper = paragraphs.paper AND s.section = paragraphs.section "
        "ORDER BY paragraphs.paper, paragraphs.section"
    ):
        secs.setdefault(r["paper"], []).append({"section": r["section"], "title": title(r)})
    parts: dict = {}
    try:
        for r in conn.execute("SELECT part, en, ru FROM parts ORDER BY part"):
            parts[r["part"]] = {"part": r["part"], "title": title(r), "papers": []}
    except sqlite3.Error:  # older books: no parts table
        pass
    out: list = []
    for r in conn.execute(
        "SELECT DISTINCT paragraphs.paper AS paper, p.en AS en, p.ru AS ru, p.part AS part "
        "FROM paragraphs LEFT JOIN papers p ON p.paper = paragraphs.paper ORDER BY paragraphs.paper"
    ):
        paper = {"paper": r["paper"], "title": title(r), "sections": secs.get(r["paper"], [])}
        part = parts.get(r["part"])
        if part is None:  # paper outside any known part: one anonymous group per run of such papers
            if not out or out[-1]["part"] is not None:
                out.append({"part": None, "title": None, "papers": []})
            out[-1]["papers"].append(paper)
            continue
        if not out or out[-1] is not part:
            out.append(part)
        part["papers"].append(paper)
    return out


def get_paper_text(conn: sqlite3.Connection, paper: int) -> Optional[dict]:
    """All paragraphs of one paper for the reading page, with `n` = 1-based position inside the paper,
    plus the neighbouring paper numbers for «previous / next»."""
    paras = get_range(conn, paper)
    if not paras:
        return None
    for i, p in enumerate(paras, 1):
        p["n"] = i
        p.pop("paper_title", None)  # the same for every paragraph: sent once below
    prev = conn.execute("SELECT MAX(paper) FROM paragraphs WHERE paper < ?", (paper,)).fetchone()[0]
    nxt = conn.execute("SELECT MIN(paper) FROM paragraphs WHERE paper > ?", (paper,)).fetchone()[0]
    t = get_titles(conn, paper)
    return {
        "paper": paper,
        "title": {"en": t["paper_en"], "ru": t["paper_ru"]},
        "part": {"part": t["part"], "en": t["part_en"], "ru": t["part_ru"]} if t["part"] is not None else None,
        "prev": prev, "next": nxt,
        "paragraphs": paras,
    }


def check_integrity(conn: sqlite3.Connection) -> tuple[bool, str]:
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        ok_pragma = row and row[0] == "ok"
        n_para = conn.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0]
        n_fts = conn.execute("SELECT COUNT(*) FROM paragraphs_fts").fetchone()[0]
        if not ok_pragma:
            return False, f"integrity_check failed: {row}"
        if n_para != n_fts:
            return False, f"paragraphs ({n_para}) != fts rows ({n_fts})"
        if n_para == 0:
            return False, "no paragraphs in database"
        return True, f"ok: {n_para} paragraphs"
    except sqlite3.Error as e:
        return False, f"sqlite error: {e}"
