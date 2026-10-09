"""Plain-language health report of the active book database (GET /api/db/status).

Everything is computed read-only from the book.db file itself (own `mode=ro` connection, so it
works even when the app refused to load a damaged database) plus a few facts from the running
app (loaded embedder, running jobs). The report contains ready-to-render rows (status
ok/warn/error/info + Russian text) and a list of warnings each with a recommended action id
that the UI turns into a button.
"""
from __future__ import annotations

import datetime as _dt
import os
import sqlite3
from typing import Optional

STATE_READY, STATE_ATTENTION, STATE_MISSING = "ready", "attention", "missing"
STATE_LABELS = {STATE_READY: "готова", STATE_ATTENTION: "требует внимания", STATE_MISSING: "не построена"}

ACTIONS = {
    "rebuild_fts": "Перестроить полнотекстовый индекс",
    "rebuild_all": "Построить базу заново",
    "open_embeddings": "Открыть настройки модели",
    "select_book": "Выбрать другую книгу",
    "build_glossary": "Построить словарь терминов",
}


def _n(x: int) -> str:
    return f"{x:,}".replace(",", " ")


def _size(b: int) -> str:
    return f"{b / 1e6:.1f} МБ".replace(".", ",")


def _fmt_dt(iso: Optional[str]) -> Optional[str]:
    if not iso:
        return None
    try:
        return _dt.datetime.fromisoformat(iso).strftime("%d.%m.%Y %H:%M")
    except ValueError:
        return iso


def _q1(conn: sqlite3.Connection, sql: str, args=()) -> Optional[int]:
    try:
        row = conn.execute(sql, args).fetchone()
        return row[0] if row else None
    except sqlite3.Error:
        return None


def build_db_status(
    db_path: str,
    book_id: str,
    *,
    fts_weights: Optional[dict] = None,
    fts_mode_hybrid: Optional[str] = None,
    fts_mode_standalone: Optional[str] = None,
    embedder_key: Optional[tuple] = None,
    embedder_error: Optional[str] = None,
    busy_job: Optional[dict] = None,
) -> dict:
    out: dict = {
        "book_id": book_id, "path": str(db_path), "exists": os.path.isfile(db_path),
        "rows": [], "warnings": [], "busy": busy_job,
        "search_settings": {"fts_weights": fts_weights, "fts_mode_hybrid": fts_mode_hybrid,
                            "fts_mode_standalone": fts_mode_standalone},
    }
    rows, warnings = out["rows"], out["warnings"]

    def row(status: str, text: str):
        rows.append({"status": status, "text": text})

    def warn(level: str, text: str, action: Optional[str]):
        warnings.append({"level": level, "text": text,
                         "action": {"id": action, "label": ACTIONS[action]} if action else None})

    if not out["exists"]:
        out.update(state=STATE_MISSING, state_label=STATE_LABELS[STATE_MISSING],
                   summary="База ещё не построена. Постройте её из исходного файла с текстом книги.")
        row("error", "Файл базы не найден: " + str(db_path))
        warn("error", "Базы для этой книги ещё нет.", "rebuild_all")
        return out

    st = os.stat(db_path)
    out["size_bytes"] = st.st_size
    out["modified"] = _dt.datetime.fromtimestamp(st.st_mtime).strftime("%d.%m.%Y %H:%M")

    try:
        conn = sqlite3.connect(f"file:{str(db_path).replace(chr(92), '/')}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        return _unreadable(out, f"не удалось открыть файл ({exc})")

    try:
        # --- integrity (quick_check is a cheap single pass over the file) -------------------
        try:
            qc = conn.execute("PRAGMA quick_check(1)").fetchone()
            integrity_ok, integrity_msg = (qc[0] == "ok"), str(qc[0])
        except sqlite3.Error as exc:
            return _unreadable(out, str(exc), conn)
        out["integrity"] = {"ok": integrity_ok, "message": integrity_msg}
        if not integrity_ok:
            return _unreadable(out, integrity_msg, conn)

        book = None
        try:
            r = conn.execute("SELECT * FROM books WHERE book_id = ?", (book_id,)).fetchone() \
                or conn.execute("SELECT * FROM books LIMIT 1").fetchone()
            book = dict(r) if r else None
        except sqlite3.Error:
            book = None
        n_par = _q1(conn, "SELECT COUNT(*) FROM paragraphs") or 0
        n_papers = _q1(conn, "SELECT COUNT(DISTINCT paper) FROM paragraphs") or 0
        n_sections = _q1(conn, "SELECT COUNT(*) FROM (SELECT DISTINCT paper, section FROM paragraphs)") or 0
        out.update(book_name=(book or {}).get("book_name"), built_at=_fmt_dt((book or {}).get("created_at")),
                   source_file=(book or {}).get("source_file"), paragraphs=n_par,
                   papers=n_papers, sections=n_sections)

        # --- file + book ----------------------------------------------------------------------
        built = f", построена {out['built_at']} (UTC)" if out["built_at"] else ""
        row("ok", f"Файл базы: {_size(st.st_size)}{built}")
        row("ok" if book else "warn",
            f"Книга: {(book or {}).get('book_name') or '—'} (код «{(book or {}).get('book_id') or book_id}»)")
        if not book:
            warn("warn", "В базе нет сведений о книге (таблица books пуста).", "rebuild_all")

        # --- paragraphs ------------------------------------------------------------------------
        if n_par == 0:
            row("error", "Абзацев: 0 — база пустая")
            warn("error", "В базе нет ни одного абзаца.", "rebuild_all")
        else:
            row("ok", f"Абзацев: {_n(n_par)} (глав: {_n(n_papers)}, разделов: {_n(n_sections)})")

        # --- titles ----------------------------------------------------------------------------
        t_papers = _q1(conn, "SELECT COUNT(*) FROM papers WHERE coalesce(en,'')!='' OR coalesce(ru,'')!=''") or 0
        t_sections = _q1(conn, "SELECT COUNT(*) FROM sections WHERE coalesce(en,'')!='' OR coalesce(ru,'')!=''") or 0
        out["titles"] = {"papers": t_papers, "sections": t_sections}
        if t_papers or t_sections:
            untitled = max(0, (n_sections or 0) - t_sections)
            note = f"; у {_n(untitled)} вводных частей названия нет, как и в книге" if untitled else ""
            row("ok", f"Названия глав и разделов: загружены ({_n(t_papers)} глав, "
                      f"{_n(t_sections)} разделов с названиями{note})")
        else:
            row("info", "Названий глав и разделов нет — в результатах поиска они не показываются")
            warn("info", "Названия глав не загружены. Положите файл data\\titles\\<код книги>_titles.json "
                 "и постройте базу заново (см. docs\\new_book_howto.md).", None)

        # --- full-text index ---------------------------------------------------------------------
        cols = [r[1] for r in conn.execute("PRAGMA table_info(paragraphs_fts)").fetchall()]
        fts_rows = _q1(conn, "SELECT COUNT(*) FROM paragraphs_fts") if cols else None
        has_title_cols = "title_en" in cols and "title_ru" in cols
        with_titles = None
        if has_title_cols:
            with_titles = _q1(conn, "SELECT COUNT(*) FROM paragraphs_fts WHERE title_en != '' OR title_ru != ''")
        out["fts"] = {"exists": bool(cols), "rows": fts_rows, "columns": cols,
                      "has_title_columns": has_title_cols, "rows_with_titles": with_titles,
                      "matches_paragraphs": fts_rows == n_par}
        has_titles_anywhere = bool(t_papers or t_sections)
        if not cols:
            row("error", "Полнотекстовый индекс (поиск по словам): отсутствует")
            warn("error", "Нет полнотекстового индекса — поиск по словам не работает.", "rebuild_fts")
        elif fts_rows != n_par:
            row("error", f"Полнотекстовый индекс: неполный ({_n(fts_rows or 0)} из {_n(n_par)} абзацев)")
            warn("error", "Полнотекстовый индекс не соответствует абзацам.", "rebuild_fts")
        elif not has_title_cols:
            row("warn", "Полнотекстовый индекс: есть, но без названий глав (старый формат)")
            warn("warn", "Индекс создан старой версией: названия глав и разделов в нём не учитываются "
                 "при поиске.", "rebuild_fts")
        elif has_titles_anywhere and not with_titles:
            row("warn", "Полнотекстовый индекс: есть, но названия глав в нём пустые")
            warn("warn", "Названия глав загружены в базу, но не попали в полнотекстовый индекс.", "rebuild_fts")
        else:
            row("ok", f"Полнотекстовый индекс: есть, {_n(fts_rows)} абзацев"
                + (", с названиями глав" if with_titles else ""))

        # --- vectors ---------------------------------------------------------------------------------
        model = (book or {}).get("embedding_model")
        runtime = (book or {}).get("embedding_runtime")
        scheme = (book or {}).get("index_scheme") or "D"
        expected = n_par * 2 if scheme == "D" else n_par
        v_count = _q1(conn, "SELECT COUNT(*) FROM embeddings WHERE model = ?", (model,)) if model else \
            _q1(conn, "SELECT COUNT(*) FROM embeddings")
        v_count = v_count or 0
        dtype = (book or {}).get("vector_dtype") or "f32"
        dtype_label = {"i8": "int8", "int8": "int8", "f32": "float32", "float32": "float32"}.get(dtype, dtype)
        coverage = round(100 * v_count / expected, 1) if expected else 0.0
        out["vectors"] = {"model": model, "runtime": runtime, "dtype": dtype_label, "dim": (book or {}).get("embedding_dim"),
                          "count": v_count, "expected": expected, "coverage_pct": coverage, "scheme": scheme}
        if v_count == 0:
            row("warn", "Векторы для поиска по смыслу: нет — работает только поиск по словам")
            warn("warn", "В базе нет векторов, поиск по смыслу недоступен.", "rebuild_all")
        elif v_count < expected:
            row("warn", f"Векторы для поиска по смыслу: {_n(v_count)} из {_n(expected)} ({coverage}%) — неполные".replace(".", ","))
            warn("warn", "Векторы построены не для всех абзацев.", "rebuild_all")
        else:
            row("ok", f"Векторы для поиска по смыслу: {_n(v_count)} из {_n(expected)} ({model}, {dtype_label})")

        # --- embedder (the model that turns the user's query into a vector) ---------------------------
        if v_count:
            if embedder_key and embedder_key[0] == model:
                same_rt = (not runtime) or embedder_key[1] == runtime
                row("ok" if same_rt else "info",
                    f"Модель смыслового поиска запущена ({embedder_key[0]}, {embedder_key[1]})"
                    + ("" if same_rt else f"; база построена в среде «{runtime}» — результаты почти совпадают"))
            elif embedder_error:
                row("warn", "Модель смыслового поиска не запустилась: " + embedder_error)
                warn("warn", "Поиск по смыслу сейчас не работает: модель не запустилась.", "open_embeddings")
            else:
                row("info", "Модель смыслового поиска ещё не запущена (запустится при первом поиске)")

        # --- glossary of terms (optional: search works without it) ---------------------------------
        from app.core.terms import glossary_info

        g = glossary_info(db_path)
        out["glossary"] = g
        if g["exists"]:
            when = f", построен {_fmt_dt(g['built_at'])}" if g.get("built_at") else ""
            row("ok", f"Словарь терминов (русские ↔ английские слова): есть, {_n(g['pairs'])} пар{when}")
        else:
            row("warn", "Словарь терминов (русские ↔ английские слова): не построен — поиск работает, но без "
                        "перевода имён и терминов между языками")
            warn("info", "Словарь терминов улучшает поиск по именам и понятиям (Эдемия ↔ Edentia). "
                         "Строится за несколько секунд из уже имеющегося текста.", "build_glossary")

        if fts_weights:
            w = ", ".join(f"{k} {v:g}" for k, v in fts_weights.items())
            row("info", f"Настройки поиска по словам: веса — {w}")
    finally:
        conn.close()

    levels = {w["level"] for w in warnings}
    if "error" in levels or "warn" in levels:
        state = STATE_ATTENTION
        out["summary"] = "База работает, но есть замечания — см. ниже." if "error" not in levels else \
            "В базе есть проблемы — см. ниже."
    else:
        state = STATE_READY
        out["summary"] = "База построена и готова к работе."
    if busy_job:
        out["summary"] = "Сейчас выполняется задача: " + {"index": "построение базы",
                                                             "rebuild-fts": "перестроение индекса",
                                                             "build-glossary": "построение словаря терминов"}.get(busy_job["type"], busy_job["type"])
    out["state"], out["state_label"] = state, STATE_LABELS[state]
    return out


def _unreadable(out: dict, reason: str, conn: Optional[sqlite3.Connection] = None) -> dict:
    if conn is not None:
        conn.close()
    out["rows"].append({"status": "error", "text": f"База данных повреждена или не читается: {reason}"})
    out["warnings"].append({"level": "error", "text": "Файл базы испорчен.",
                            "action": {"id": "rebuild_all", "label": ACTIONS["rebuild_all"]}})
    out["warnings"].append({"level": "info", "text": "Можно переключиться на другую книгу.",
                            "action": {"id": "select_book", "label": ACTIONS["select_book"]}})
    out.update(state=STATE_ATTENTION, state_label=STATE_LABELS[STATE_ATTENTION],
               summary="Файл базы повреждён или не читается.")
    return out
