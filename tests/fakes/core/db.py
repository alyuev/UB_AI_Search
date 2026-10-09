"""Fake app.core.db for unit tests: an in-memory sqlite-like store built on plain dicts."""
from __future__ import annotations

import sqlite3
from typing import Optional


def _row_to_para(row) -> dict:
    return {
        "id": row["id"],
        "paper": row["paper"],
        "section": row["section"],
        "paragraph": row["paragraph"],
        "en": row["en"],
        "ru": row["ru"],
    }


def open_db(path: str, readonly: bool = False):
    uri = f"file:{path}?mode=ro" if readonly else path
    conn = sqlite3.connect(uri if readonly else path, uri=readonly)
    conn.row_factory = sqlite3.Row
    return conn


def get_book_info(conn) -> dict:
    row = conn.execute("SELECT * FROM books LIMIT 1").fetchone()
    return dict(row) if row else {}


def get_paragraph(conn, id_: int) -> Optional[dict]:
    row = conn.execute("SELECT * FROM paragraphs WHERE id=?", (id_,)).fetchone()
    return _row_to_para(row) if row else None


def get_context(conn, id_: int, before: int = 2, after: int = 2) -> list[dict]:
    center = get_paragraph(conn, id_)
    if not center:
        return []
    rows = conn.execute(
        "SELECT * FROM paragraphs WHERE paper=? AND section IS ? "
        "ORDER BY paragraph",
        (center["paper"], center["section"]),
    ).fetchall()
    paras = [_row_to_para(r) for r in rows]
    idx = next((i for i, p in enumerate(paras) if p["id"] == id_), None)
    if idx is None:
        return [center]
    lo = max(0, idx - before)
    hi = min(len(paras), idx + after + 1)
    return paras[lo:hi]


def get_range(conn, paper: int, section: Optional[int] = None,
              para_from: Optional[int] = None, para_to: Optional[int] = None) -> list[dict]:
    q = "SELECT * FROM paragraphs WHERE paper=?"
    args: list = [paper]
    if section is not None:
        q += " AND section=?"
        args.append(section)
    if para_from is not None:
        q += " AND paragraph>=?"
        args.append(para_from)
    if para_to is not None:
        q += " AND paragraph<=?"
        args.append(para_to)
    q += " ORDER BY section, paragraph"
    rows = conn.execute(q, args).fetchall()
    return [_row_to_para(r) for r in rows]


def list_papers(conn) -> list[int]:
    rows = conn.execute("SELECT DISTINCT paper FROM paragraphs ORDER BY paper").fetchall()
    return [r["paper"] for r in rows]


def list_sections(conn, paper: int) -> list[int]:
    rows = conn.execute(
        "SELECT DISTINCT section FROM paragraphs WHERE paper=? ORDER BY section", (paper,)
    ).fetchall()
    return [r["section"] for r in rows]


def check_integrity(conn) -> bool:
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        return row[0] == "ok"
    except sqlite3.DatabaseError:
        return False


def create_fake_db(path: str, book_id: str = "urantia", book_name: str = "The Urantia Book"):
    """Helper (test-only) to build a minimal fake book.db with a handful of paragraphs."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE books (id TEXT, name TEXT)")
    conn.execute("INSERT INTO books VALUES (?,?)", (book_id, book_name))
    conn.execute(
        "CREATE TABLE paragraphs (id INTEGER PRIMARY KEY, paper INTEGER, section INTEGER, "
        "paragraph INTEGER, en TEXT, ru TEXT)"
    )
    data = [
        (1, 147, 2, 1, "In the beginning was the Word.", "В начале было Слово."),
        (2, 147, 2, 2, "Jesus spoke of the Father's love.", "Иисус говорил о любви Отца."),
        (3, 147, 2, 3, "The disciples listened intently.", "Ученики слушали внимательно."),
        (4, 147, 2, 4, "It was evening in Jerusalem.", "Был вечер в Иерусалиме."),
        (5, 1, 1, 1, "God is spirit.", "Бог есть дух."),
    ]
    conn.executemany(
        "INSERT INTO paragraphs VALUES (?,?,?,?,?,?)", data
    )
    conn.execute(
        "CREATE VIRTUAL TABLE paragraphs_fts USING fts5(en, ru, content='paragraphs', content_rowid='id')"
    )
    conn.execute("INSERT INTO paragraphs_fts(rowid, en, ru) SELECT id, en, ru FROM paragraphs")
    conn.commit()
    conn.close()
