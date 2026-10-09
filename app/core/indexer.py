"""Build a book.db from the source JSON corpus: parse, store, FTS, lemmatize,
embed, and atomically finalize.
"""
from __future__ import annotations

import shutil
import sqlite3
import struct
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from app.config import DATA_DIR
from app.core.db import SCHEMA_SQL, check_integrity, quantize_vector
from app.core.embedder import Embedder, create_embedder
from app.core.parser import parse_json
from app.log import get_logger

logger = get_logger("index")

ProgressFn = Callable[[str, int, int, Optional[str], str], None]


def _noop_progress(stage: str, current: int, total: int, current_id, message: str) -> None:
    pass


def _lower_process_priority() -> None:
    """Best-effort: lower this process's OS scheduling priority so a long
    CPU-bound indexing run leaves headroom for the desktop UI. No-op if
    psutil is unavailable or the platform doesn't support it.
    """
    try:
        import psutil

        p = psutil.Process()
        if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            # POSIX: a small positive niceness, never touching realtime range.
            p.nice(10)
        logger.info("lowered process priority for indexing")
    except Exception as e:
        logger.debug("could not lower process priority: %s", e)


def _lemmatize_ru(text: str, morph) -> str:
    words = []
    for tok in text.split():
        cleaned = "".join(ch for ch in tok if ch.isalnum())
        if not cleaned:
            continue
        try:
            parsed = morph.parse(cleaned)[0]
            words.append(parsed.normal_form)
        except Exception:
            words.append(cleaned.lower())
    return " ".join(words)


def _vec_to_blob(vec) -> bytes:
    return struct.pack(f"<{len(vec)}f", *[float(x) for x in vec])


def _load_titles(titles_path: Optional[Path], conn: sqlite3.Connection) -> None:
    """Load a titles JSON ({"parts":[...], "papers":{"<n>":{en,ru,part,sections:{...}}}})
    into the parts/papers/sections tables. Missing file is a no-op.
    """
    import json

    if titles_path is None or not Path(titles_path).exists():
        return

    with open(titles_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for part in data.get("parts", []) or []:
        papers_range = part.get("papers") or [None, None]
        first_p, last_p = (papers_range + [None, None])[:2]
        conn.execute(
            "INSERT OR REPLACE INTO parts (part, en, ru, first_paper, last_paper) VALUES (?,?,?,?,?)",
            (part.get("part"), part.get("en"), part.get("ru"), first_p, last_p),
        )

    for paper_str, pinfo in (data.get("papers") or {}).items():
        try:
            paper_num = int(paper_str)
        except (TypeError, ValueError):
            continue
        conn.execute(
            "INSERT OR REPLACE INTO papers (paper, en, ru, part) VALUES (?,?,?,?)",
            (paper_num, pinfo.get("en"), pinfo.get("ru"), pinfo.get("part")),
        )
        for sec_str, sinfo in (pinfo.get("sections") or {}).items():
            try:
                sec_num = int(sec_str)
            except (TypeError, ValueError):
                continue
            conn.execute(
                "INSERT OR REPLACE INTO sections (paper, section, en, ru) VALUES (?,?,?,?)",
                (paper_num, sec_num, sinfo.get("en"), sinfo.get("ru")),
            )
    conn.commit()


def _fts_rows(conn: sqlite3.Connection, morph) -> list[tuple]:
    """(id, en, ru, ru_lemma, title_en, title_ru) for every paragraph, in seq
    order. title_en = paper + section title (EN); title_ru = the same in RU,
    raw AND lemmatized. Missing titles tables/rows yield empty strings.
    """
    try:
        titles = conn.execute(
            "SELECT paragraphs.id AS id, p.en AS pen, p.ru AS pru, s.en AS sen, s.ru AS sru "
            "FROM paragraphs LEFT JOIN papers p ON p.paper = paragraphs.paper "
            "LEFT JOIN sections s ON s.paper = paragraphs.paper AND s.section = paragraphs.section"
        ).fetchall()
    except sqlite3.Error:
        titles = []
    tmap = {}
    cache: dict[tuple, tuple[str, str]] = {}
    for t in titles:
        key = (t[1], t[2], t[3], t[4])
        if key not in cache:
            en = " ".join(x for x in (t[1], t[3]) if x)
            ru = " ".join(x for x in (t[2], t[4]) if x)
            ru_all = (ru + " " + _lemmatize_ru(ru, morph)).strip() if ru else ""
            cache[key] = (en, ru_all)
        tmap[t[0]] = cache[key]
    out = []
    for r in conn.execute("SELECT id, en, ru, ru_lemma FROM paragraphs ORDER BY seq").fetchall():
        te, tr = tmap.get(r[0], ("", ""))
        out.append((r[0], r[1], r[2], r[3], te, tr))
    return out


def rebuild_fts(
    db_path: str | Path,
    progress: Optional[ProgressFn] = None,
    cancel: Optional[threading.Event] = None,
) -> int:
    """Drop and recreate paragraphs_fts (new schema incl. title columns) from the
    existing paragraphs + papers/sections tables. Embeddings are NOT touched.

    Rows are computed first; the DROP/CREATE/INSERT then runs in ONE write
    transaction, so a failure/cancel leaves the old index intact and readers
    never see a half-built table. Missing paragraphs.ru_lemma values are
    filled in. Returns the number of indexed paragraphs.
    """
    from app.core.db import FTS_CREATE_SQL

    progress = progress or _noop_progress
    cancel = cancel or threading.Event()
    conn = sqlite3.connect(str(db_path), timeout=60, isolation_level=None)
    try:
        import pymorphy3

        morph = pymorphy3.MorphAnalyzer()
        progress("fts", 0, 0, None, "preparing rows")
        missing = conn.execute(
            "SELECT id, ru FROM paragraphs WHERE ru_lemma IS NULL OR ru_lemma = ''"
        ).fetchall()
        fills = [(_lemmatize_ru(ru or "", morph), pid) for pid, ru in missing]
        rows = _fts_rows(conn, morph)
        total = len(rows)
        if cancel.is_set():
            raise RuntimeError("cancelled")
        progress("fts", 0, total, None, "rebuilding full-text index")
        conn.execute("BEGIN IMMEDIATE")
        try:
            if fills:
                conn.executemany("UPDATE paragraphs SET ru_lemma = ? WHERE id = ?", fills)
                rows = _fts_rows(conn, morph)
            conn.execute("DROP TABLE IF EXISTS paragraphs_fts")
            conn.execute(FTS_CREATE_SQL.format(name="paragraphs_fts"))
            conn.executemany(
                "INSERT INTO paragraphs_fts (id, en, ru, ru_lemma, title_en, title_ru) "
                "VALUES (?,?,?,?,?,?)",
                rows,
            )
            if cancel.is_set():
                raise RuntimeError("cancelled")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        progress("fts", total, total, None, "fts index rebuilt")
        return total
    finally:
        conn.close()


def build_index(
    json_path: str | Path,
    db_path: str | Path,
    model_name: str,
    scheme: str = "D",
    book_id: str = "urantia",
    book_name: str = "The Urantia Book",
    progress: Optional[ProgressFn] = None,
    cancel: Optional[threading.Event] = None,
    device: str = "auto",
    embed_batch_size: int = 64,
    titles_path: Optional[str | Path] = None,
    intra_op_threads: Optional[int] = None,
    lower_priority: bool = True,
    runtime: str = "auto",
    vector_dtype: str = "int8",
) -> Path:
    """Parse json_path, build a fresh SQLite db with FTS + embeddings, and
    atomically move it into place at db_path. Returns db_path.

    `lower_priority` (default True): drop this process's OS scheduling
    priority to BELOW_NORMAL (Windows) while indexing runs, so a long CPU
    embedding pass doesn't make the desktop UI feel sluggish. Best-effort;
    silently skipped if psutil isn't available or the platform doesn't
    support it.

    `vector_dtype` ("int8" default, or "float32"): int8 (symmetric per-vector
    quantization, scale = max|x|/127) roughly quarters embeddings storage
    with no measurable quality loss on this corpus (pair retrieval identical,
    scheme-D top-10 overlap 98.9%, top-1 98% vs float32 — see
    docs/03_benchmark_search.md). Dequantized to float32 once at Searcher
    load time, not per-query.
    """
    progress = progress or _noop_progress
    cancel = cancel or threading.Event()
    json_path = Path(json_path)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if lower_priority:
        _lower_process_priority()

    if titles_path is None:
        default_titles = DATA_DIR / "titles" / f"{book_id}_titles.json"
        titles_path = default_titles if default_titles.exists() else None
    else:
        titles_path = Path(titles_path)

    t0 = time.time()
    logger.info(
        "build_index start: json=%s db=%s model=%s scheme=%s", json_path, db_path, model_name, scheme
    )

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".db", dir=str(db_path.parent))
    import os

    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    tmp_path.unlink(missing_ok=True)

    try:
        conn = sqlite3.connect(str(tmp_path))
        conn.executescript(SCHEMA_SQL)

        # --- stage: parsing ---
        progress("parsing", 0, 0, None, "parsing source JSON")
        records = list(parse_json(json_path))
        total = len(records)
        progress("parsing", total, total, None, f"parsed {total} paragraphs")
        if cancel.is_set():
            raise RuntimeError("cancelled")

        # --- stage: database ---
        progress("database", 0, total, None, "inserting paragraphs")
        for i, rec in enumerate(records):
            conn.execute(
                "INSERT INTO paragraphs (id, book_id, seq, paper, section, paragraph, en, ru) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    rec["id"], book_id, i, rec["paper"], rec["section"], rec["paragraph"],
                    rec["en"], rec["ru"],
                ),
            )
            if i % 2000 == 0:
                progress("database", i, total, rec["id"], "inserting paragraphs")
            if cancel.is_set():
                raise RuntimeError("cancelled")
        conn.commit()
        progress("database", total, total, None, "paragraphs inserted")

        # --- stage: lemmatize ---
        progress("lemmatize", 0, total, None, "lemmatizing russian text")
        import pymorphy3

        morph = pymorphy3.MorphAnalyzer()
        cur = conn.execute("SELECT id, ru FROM paragraphs ORDER BY seq")
        rows = cur.fetchall()
        for i, (pid, ru) in enumerate(rows):
            lemma = _lemmatize_ru(ru or "", morph)
            conn.execute("UPDATE paragraphs SET ru_lemma = ? WHERE id = ?", (lemma, pid))
            if i % 2000 == 0:
                progress("lemmatize", i, total, pid, "lemmatizing russian text")
            if cancel.is_set():
                raise RuntimeError("cancelled")
        conn.commit()
        progress("lemmatize", total, total, None, "lemmatization done")

        # --- stage: fts (titles must be loaded first: they are indexed too) ---
        progress("fts", 0, total, None, "building full-text index")
        _load_titles(titles_path, conn)
        rows = _fts_rows(conn, morph)
        conn.executemany(
            "INSERT INTO paragraphs_fts (id, en, ru, ru_lemma, title_en, title_ru) "
            "VALUES (?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
        progress("fts", total, total, None, "fts index built")

        # --- glossary of terms RU<->EN (optional bonus; a failure must not break the build) ---
        try:
            from app.core.terms import build_glossary

            ginfo = build_glossary(
                tmp_path, cancel=cancel,
                progress=lambda st_, cur, tot, cid, msg: progress("fts", cur, tot, cid, "словарь терминов: " + msg))
            logger.info("glossary built: %s", ginfo)
        except RuntimeError as exc:
            if str(exc) == "cancelled":
                raise
            logger.warning("glossary skipped: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("glossary skipped: %s", exc)
        if cancel.is_set():
            raise RuntimeError("cancelled")

        # --- stage: embedding ---
        progress("embedding", 0, total, None, f"loading embedding model {model_name}")
        embedder = create_embedder(
            model_name, runtime=runtime, device=device, intra_op_threads=intra_op_threads
        )
        ids_ordered = [r["id"] for r in records]
        en_texts = [r["en"] for r in records]
        ru_texts = [r["ru"] for r in records]

        for lang, texts in (("en", en_texts), ("ru", ru_texts)):
            done = 0
            for start in range(0, total, embed_batch_size):
                if cancel.is_set():
                    raise RuntimeError("cancelled")
                batch_ids = ids_ordered[start : start + embed_batch_size]
                batch_texts = texts[start : start + embed_batch_size]
                vecs = embedder.embed_passages(batch_texts, batch_size=embed_batch_size)
                if vector_dtype == "int8":
                    rows = []
                    for pid, vec in zip(batch_ids, vecs):
                        blob, scale = quantize_vector(vec)
                        rows.append((pid, lang, model_name, blob, "i8", scale))
                else:
                    rows = [
                        (pid, lang, model_name, _vec_to_blob(vec), "f32", None)
                        for pid, vec in zip(batch_ids, vecs)
                    ]
                conn.executemany(
                    "INSERT INTO embeddings (para_id, lang, model, vector, dtype, scale) VALUES (?,?,?,?,?,?)",
                    rows,
                )
                done += len(batch_ids)
                current = done if lang == "en" else total + done
                progress("embedding", current, total * 2, batch_ids[-1], f"embedding {lang}")
            conn.commit()
        progress("embedding", total * 2, total * 2, None, "embedding done")

        # --- stage: finalize ---
        progress("finalize", 0, 1, None, "writing book metadata")
        import datetime

        conn.execute(
            "INSERT OR REPLACE INTO books (book_id, book_name, language_source, "
            "language_translation, source_file, created_at, embedding_model, embedding_dim, "
            "index_scheme, embedding_runtime, vector_dtype) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                book_id, book_name, "en", "ru", str(json_path),
                datetime.datetime.utcnow().isoformat(), model_name, embedder.dim, scheme,
                getattr(embedder, "runtime", "onnx"),
                "i8" if vector_dtype == "int8" else "f32",
            ),
        )
        conn.commit()

        if hasattr(embedder, "close"):
            embedder.close()

        ok, msg = check_integrity(conn)
        logger.info("integrity check: ok=%s msg=%s", ok, msg)
        conn.close()

        if not ok:
            raise RuntimeError(f"integrity check failed: {msg}")

        # atomic replace
        if db_path.exists():
            db_path.unlink()
        shutil.move(str(tmp_path), str(db_path))
        progress("finalize", 1, 1, None, "index build complete")

        elapsed = time.time() - t0
        logger.info(
            "build_index done in %.1fs: %d paragraphs, model=%s, db=%s", elapsed, total, model_name, db_path
        )
        return db_path

    except Exception:
        logger.exception("build_index failed")
        try:
            conn.close()
        except Exception:
            pass
        tmp_path.unlink(missing_ok=True)
        raise
