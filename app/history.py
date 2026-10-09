"""Query history (searches, questions, translation analyses) in its own SQLite file data/history.db.

Deliberately NOT stored inside a book.db: the book database stays portable/shareable and never
contains a user's personal query log. Each entry keeps a compact snapshot of the result so it can
be redrawn instantly without re-running anything (paragraph text is re-fetched by id on open).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sqlite3
from typing import Optional

from app import config

MODES = ("search", "rag", "translation")
MAX_ENTRIES = 1000  # pinned entries are never auto-deleted

_SCHEMA = """
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    book_id TEXT,
    mode TEXT NOT NULL,
    query TEXT NOT NULL,
    params TEXT,
    result TEXT,
    summary TEXT,
    duration_s REAL,
    status TEXT NOT NULL DEFAULT 'ok',
    pinned INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_history_created ON history(created_at);
CREATE INDEX IF NOT EXISTS idx_history_mode ON history(mode);
"""


def _db_path() -> str:
    # resolved on every call: tests (and the packaged app) may point config.DATA_DIR elsewhere
    return os.path.join(str(config.DATA_DIR), "history.db")


def _connect() -> sqlite3.Connection:
    path = _db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.create_function("lc", 1, lambda s: s.lower() if isinstance(s, str) else s)  # Cyrillic-aware
    conn.executescript(_SCHEMA)
    have = {r[1] for r in conn.execute("PRAGMA table_info(history)").fetchall()}
    for col, ddl in (("experiment", "TEXT"), ("rating", "INTEGER NOT NULL DEFAULT 0"), ("note", "TEXT")):
        if col not in have:  # history.db created by an older version of the app
            conn.execute(f"ALTER TABLE history ADD COLUMN {col} {ddl}")
    conn.commit()
    return conn


def _row(r: sqlite3.Row, with_result: bool = False) -> dict:
    d = {
        "id": r["id"], "created_at": r["created_at"], "book_id": r["book_id"], "mode": r["mode"],
        "query": r["query"], "params": json.loads(r["params"]) if r["params"] else {},
        "summary": r["summary"], "duration_s": r["duration_s"], "status": r["status"],
        "pinned": bool(r["pinned"]),
        "rating": r["rating"] or 0, "note": r["note"] or "",
    }
    keys = r.keys()
    if "has_experiment" in keys:
        d["has_experiment"] = bool(r["has_experiment"])
    if with_result:
        d["result"] = json.loads(r["result"]) if r["result"] else None
        d["experiment"] = json.loads(r["experiment"]) if r["experiment"] else None
        d["has_experiment"] = d["experiment"] is not None
    return d


def add(book_id: Optional[str], mode: str, query: str, params: Optional[dict], result: Optional[dict],
        summary: str, duration_s: Optional[float] = None, status: str = "ok",
        experiment: Optional[dict] = None) -> int:
    if mode not in MODES:
        raise ValueError(f"unknown history mode {mode!r}")
    now = _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO history (created_at, book_id, mode, query, params, result, summary, duration_s, status, "
            "experiment) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (now, book_id, mode, query, json.dumps(params or {}, ensure_ascii=False),
             json.dumps(result, ensure_ascii=False) if result is not None else None,
             summary, None if duration_s is None else round(duration_s, 2), status,
             json.dumps(experiment, ensure_ascii=False) if experiment is not None else None),
        )
        new_id = cur.lastrowid
        # cap: drop the oldest UNPINNED entries beyond MAX_ENTRIES
        total = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
        if total > MAX_ENTRIES:
            conn.execute(
                "DELETE FROM history WHERE id IN (SELECT id FROM history WHERE pinned = 0 "
                "ORDER BY created_at ASC, id ASC LIMIT ?)", (total - MAX_ENTRIES,))
        conn.commit()
        return new_id
    finally:
        conn.close()


def list_entries(mode: Optional[str] = None, q: Optional[str] = None, book: Optional[str] = None,
                 limit: int = 50, offset: int = 0, rating: Optional[str] = None,
                 sort: str = "date") -> dict:
    where, args = [], []
    if mode:
        where.append("mode = ?")
        args.append(mode)
    if book:
        where.append("book_id = ?")
        args.append(book)
    if q:
        where.append("lc(query) LIKE lc(?)")
        args.append(f"%{q}%")
    if rating == "up":
        where.append("rating > 0")
    elif rating == "down":
        where.append("rating < 0")
    elif rating == "rated":
        where.append("rating != 0")
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    order = "rating DESC, created_at DESC, id DESC" if sort == "rating" else "created_at DESC, id DESC"
    conn = _connect()
    try:
        total = conn.execute(f"SELECT COUNT(*) FROM history{clause}", args).fetchone()[0]
        rows = conn.execute(
            "SELECT id, created_at, book_id, mode, query, params, NULL AS result, NULL AS experiment, summary, duration_s, "
            "status, pinned, rating, note, (experiment IS NOT NULL) AS has_experiment "
            f"FROM history{clause} ORDER BY {order} LIMIT ? OFFSET ?",
            args + [max(1, min(limit, 500)), max(0, offset)]).fetchall()
        return {"items": [_row(r) for r in rows], "total": total}
    finally:
        conn.close()


def get(entry_id: int) -> Optional[dict]:
    conn = _connect()
    try:
        r = conn.execute("SELECT * FROM history WHERE id = ?", (entry_id,)).fetchone()
        return _row(r, with_result=True) if r else None
    finally:
        conn.close()


def delete(entry_id: int) -> bool:
    conn = _connect()
    try:
        n = conn.execute("DELETE FROM history WHERE id = ?", (entry_id,)).rowcount
        conn.commit()
        return n > 0
    finally:
        conn.close()


def set_pinned(entry_id: int, pinned: Optional[bool] = None) -> Optional[bool]:
    """Set (or toggle when pinned is None) the pin flag; returns the new value or None if not found."""
    conn = _connect()
    try:
        r = conn.execute("SELECT pinned FROM history WHERE id = ?", (entry_id,)).fetchone()
        if r is None:
            return None
        new = (not r["pinned"]) if pinned is None else bool(pinned)
        conn.execute("UPDATE history SET pinned = ? WHERE id = ?", (int(new), entry_id))
        conn.commit()
        return new
    finally:
        conn.close()


def clear_unpinned() -> int:
    conn = _connect()
    try:
        n = conn.execute("DELETE FROM history WHERE pinned = 0").rowcount
        conn.commit()
        return n
    finally:
        conn.close()


def set_rating(entry_id: int, rating: Optional[int] = None, note: Optional[str] = None) -> Optional[dict]:
    """Record the user's verdict on an entry: rating -1 (worse) / 0 (none) / 1 (better) and a free note.
    Each argument is optional: only the given ones change. Returns {rating, note} or None if not found."""
    if rating is not None and rating not in (-1, 0, 1):
        raise ValueError("оценка должна быть -1, 0 или 1")
    if note is not None and len(note) > 4000:
        raise ValueError("заметка слишком длинная (больше 4000 знаков)")
    conn = _connect()
    try:
        r = conn.execute("SELECT rating, note FROM history WHERE id = ?", (entry_id,)).fetchone()
        if r is None:
            return None
        new_rating = r["rating"] if rating is None else rating
        new_note = (r["note"] or "") if note is None else note
        conn.execute("UPDATE history SET rating = ?, note = ? WHERE id = ?", (new_rating, new_note, entry_id))
        conn.commit()
        return {"rating": new_rating, "note": new_note}
    finally:
        conn.close()
