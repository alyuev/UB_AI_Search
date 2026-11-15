"""FastAPI application: static UI + search/RAG/translation-analysis/indexing API.

Everything is lazy-loaded: the app must start and FTS must keep working even if the
embedding model or LM Studio is unavailable; a corrupted DB surfaces a clear error to the UI.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import re
import threading
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
import mimetypes

from fastapi.staticfiles import StaticFiles

# Windows registry often lacks .wasm; Hyphenopoly's patterns must be served as application/wasm
mimetypes.add_type("application/wasm", ".wasm")
from pydantic import BaseModel

from app import config
from app.version import __version__
from app.log import get_logger
from app.core import db as core_db
from app.core import embedder as core_embedder
from app.core import indexer as core_indexer
from app.core import terms as core_terms
from app.core import multi as core_multi
from app.core import parser as core_parser
from app.core.search import Searcher
from app import history as hist
from app import options as opts_mod
from app import prompts as prompts_mod
from app.history_compare import compare as compare_entries
from app.db_status import build_db_status
from app.jobs import manager as job_manager
from app.llm.lmstudio import LMStudioError
from app.llm import providers
from app import secrets_store
from app.llm.rag import RagPipeline, resolve_reasoning_effort
from app.llm.translation import TranslationAnalyzer, export_markdown
from app.llm.refparse import parse_reference

logger = get_logger("app")

app = FastAPI(title="SmartSearch")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class AppState:
    """Holds lazily-created, mutable singletons: DB connection, embedder, searcher.

    The embedder (in particular a LlamaCppEmbedder, which owns a live llama-server
    subprocess) is deliberately kept alive and REUSED across searches and across
    reload_searcher() calls -- only closed/recreated when the (model, runtime) pair
    actually changes. This avoids re-spawning llama-server on every search.
    """

    def __init__(self):
        self.conn = None
        self.book_id = None
        self.db_error: Optional[str] = None
        self.embedder = None
        self.embedder_key: Optional[tuple] = None  # (model_name, runtime) the current embedder was built with
        self.embedder_error: Optional[str] = None
        self.searcher = None
        self.lock = threading.Lock()
        # Other opened books for multi-book search (the ACTIVE book lives in conn/searcher above):
        # book_id -> BookHandle, opened lazily. Embedders of other models are kept per (model, runtime).
        self.handles: dict[str, "BookHandle"] = {}
        self.extra_embedders: dict[tuple, object] = {}
        self.handles_lock = threading.RLock()

    def book_db_path(self, book_id: str) -> str:
        return os.path.join(config.LIBRARY_DIR, book_id, "book.db")

    def load_book(self, book_id: str):
        with self.lock:
            path = self.book_db_path(book_id)
            self.db_error = None
            self._close_conn()
            self._close_searcher()
            if not os.path.exists(path):
                self.db_error = f"База книги не найдена: {path}"
                return
            try:
                conn = core_db.open_db(path, readonly=True)
                ok, detail = core_db.check_integrity(conn)
                if not ok:
                    self.db_error = f"База данных повреждена: {detail}"
                    conn.close()
                    return
                self.conn = conn
                self.book_id = book_id
            except Exception as exc:  # noqa: BLE001
                self.db_error = f"Ошибка открытия базы: {exc}"
                return
            self._rebuild_searcher()

    def _close_conn(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:  # noqa: BLE001
                pass
            self.conn = None

    def _close_searcher(self):
        """Close the Searcher's own sqlite connection explicitly (it opens a second,
        writable connection to book.db) instead of waiting for GC."""
        sr = self.searcher
        self.searcher = None
        if sr is not None:
            try:
                sr.conn.close()
            except Exception:  # noqa: BLE001
                pass

    def rebuild_searcher(self):
        """Recreate only the Searcher (keeps the DB connection and the embedder/llama-server):
        needed after the FTS table was rebuilt (column detection is cached per instance) or
        after FTS search settings (weights/modes) changed."""
        with self.lock:
            self._close_searcher()
            self._rebuild_searcher()
        self.drop_handles()   # the other books pick up the new FTS settings when opened again

    def release_for_reindex(self):
        """Close the DB connection (and the searcher that holds its own writable
        connection) *before* build_index atomically replaces book.db. On Windows,
        an open sqlite3 handle on the target file can make os.replace()/MoveFileEx
        fail or silently block, so this must run before the index job starts
        writing. The embedder itself (llama-server subprocess, if any) is left
        running -- it doesn't hold a handle on book.db, only on the GGUF model
        file, so it's safe (and desirable, per "reuse not re-spawn") to keep it up
        across a reindex; reload_searcher() will just start using it again."""
        with self.lock:
            self._close_conn()
            self._close_searcher()

    def ensure_embedder(self, model_name: Optional[str] = None, runtime: Optional[str] = None,
                         device: Optional[str] = None):
        settings = config.load_settings()
        model_name = model_name or settings.get("embedding_model")
        runtime = runtime or settings.get("embedding_runtime", "auto")
        device = device or settings.get("device", "auto")
        key = (model_name, runtime)
        if self.embedder is not None and self.embedder_key == key:
            return  # already have the right embedder loaded (and its subprocess, if any) -- reuse it
        if self.embedder is not None and hasattr(self.embedder, "close"):
            try:
                self.embedder.close()
            except Exception:  # noqa: BLE001
                pass
        self.embedder = None
        self.embedder_error = None
        try:
            self.embedder = core_embedder.create_embedder(model_name, runtime=runtime, device=device)
            self.embedder_key = key
        except Exception as exc:  # noqa: BLE001
            self.embedder_error = str(exc)
            self.embedder_key = None
            logger.warning("embedder unavailable (%s/%s): %s", model_name, runtime, exc)

    def _rebuild_searcher(self):
        if self.conn is None:
            self.searcher = None
            return
        # Use the runtime/model this book.db was actually built with (recorded in its
        # `books` row), not just whatever the global settings currently say, so query
        # vectors are comparable to the stored ones.
        info = core_db.get_book_info(self.conn, self.book_id) or {}
        model_name = info.get("embedding_model")
        runtime = info.get("embedding_runtime") or "auto"
        if model_name:
            self.ensure_embedder(model_name=model_name, runtime=runtime)
        else:
            self.ensure_embedder()
        path = self.book_db_path(self.book_id)
        try:
            self.searcher = Searcher(path, self.embedder)
        except Exception as exc:  # noqa: BLE001
            logger.warning("searcher init failed, FTS-only fallback: %s", exc)
            self.searcher = None

    # ---- other books (multi-book search) ----
    def embedder_for(self, model_name: Optional[str], runtime: Optional[str]):
        """(embedder, problem) for a book built with this model/runtime: the active embedder when it is the
        same one (the query vector is then computed once for all such books), else one cached per pair."""
        runtime = runtime or "auto"
        if not model_name:
            return None, "в базе не записана модель эмбеддингов"
        key = (model_name, runtime)
        if self.embedder is not None and self.embedder_key == key:
            return self.embedder, None
        if key in self.extra_embedders:
            return self.extra_embedders[key], None
        try:
            emb = core_embedder.create_embedder(model_name, runtime=runtime,
                                                device=config.load_settings().get("device", "auto"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("embedder for %s/%s unavailable: %s", model_name, runtime, exc)
            return None, f"модель {model_name} недоступна: {exc}"
        self.extra_embedders[key] = emb
        return emb, None

    def handle(self, book_id: str) -> "BookHandle":
        """Opened book (connection + searcher + info) for any book in the library; KeyError if it has no database."""
        with self.handles_lock:
            if book_id == self.book_id and self.conn is not None:
                info = core_db.get_book_info(self.conn, book_id) or {}
                return BookHandle(book_id, info.get("book_name") or book_id, self.conn, self.searcher, info,
                                  None if self.searcher is not None else self.embedder_error)
            h = self.handles.get(book_id)
            if h is not None:
                return h
            path = self.book_db_path(book_id)
            if not os.path.exists(path):
                raise KeyError(book_id)
            conn = core_db.open_db(path, readonly=True)
            info = core_db.get_book_info(conn, book_id) or core_db.get_book_info(conn) or {}
            emb, problem = self.embedder_for(info.get("embedding_model"), info.get("embedding_runtime"))
            searcher = None
            try:
                searcher = Searcher(path, emb)
            except Exception as exc:  # noqa: BLE001
                problem = problem or f"поисковый индекс не открылся: {exc}"
            h = BookHandle(book_id, info.get("book_name") or book_id, conn, searcher, info, problem)
            self.handles[book_id] = h
            return h

    def drop_handle(self, book_id: str) -> None:
        """Forget a cached non-active book (it is being re-indexed or deleted) and release its files."""
        with self.handles_lock:
            h = self.handles.pop(book_id, None)
        if h is not None:
            h.close()

    def drop_handles(self) -> None:
        with self.handles_lock:
            hs, self.handles = list(self.handles.values()), {}
        for h in hs:
            h.close()

    def reload_searcher(self):
        if self.book_id:
            self.load_book(self.book_id)

    def shutdown(self):
        """Close everything on app shutdown -- in particular, terminate a
        LlamaCppEmbedder's llama-server subprocess rather than leaving it running."""
        self._close_conn()
        self.drop_handles()
        for emb in list(self.extra_embedders.values()):
            if hasattr(emb, "close"):
                try:
                    emb.close()
                except Exception:  # noqa: BLE001
                    pass
        self.extra_embedders = {}
        if self.embedder is not None and hasattr(self.embedder, "close"):
            try:
                self.embedder.close()
            except Exception:  # noqa: BLE001
                pass
        self.embedder = None
        self.embedder_key = None
        self.searcher = None


class BookHandle:
    """One opened book: read-only connection, searcher (None = no index) and its `books` row."""

    def __init__(self, book_id, book_name, conn, searcher, info, problem=None):
        self.book_id, self.book_name, self.conn, self.searcher = book_id, book_name, conn, searcher
        self.info, self.problem = info or {}, problem
        self.model = self.info.get("embedding_model")
        self.owned = True

    @property
    def mono(self) -> bool:
        return not (self.info.get("language_translation") or "")

    def close(self):
        for c in (self.searcher.conn if self.searcher is not None else None, self.conn):
            try:
                if c is not None:
                    c.close()
            except Exception:  # noqa: BLE001
                pass


state = AppState()


@app.on_event("startup")
def _startup():
    settings = config.load_settings()
    active = settings.get("active_book")
    if active:
        state.load_book(active)


@app.on_event("shutdown")
def _shutdown():
    state.shutdown()


_BOOT_ID = f"{os.getpid()}-{time.time():.0f}"   # changes with every start: the page waits for a new one
_STARTED_AT = _dt.datetime.now()


def _app_info() -> dict:
    """Where the program runs from (settings page → «Программа»): folders, launcher, Python, start time."""
    import sys
    server = getattr(app.state, "uvicorn_server", None)
    port = getattr(getattr(server, "config", None), "port", None)
    return {"version": __version__, "root": str(config.ROOT), "data_dir": str(config.DATA_DIR),
            "logs_dir": str(config.LOGS_DIR), "launcher": os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else "",
            "cwd": os.getcwd(), "python": sys.executable, "pid": os.getpid(), "port": port,
            "started": _STARTED_AT.strftime("%d.%m.%Y %H:%M:%S")}


@app.get("/api/restart")
def api_restart_info():
    """Whether a restart is possible and which jobs it would interrupt."""
    running = [j.type for j in job_manager.list() if j.status in ("queued", "running")]
    return {"available": getattr(app.state, "uvicorn_server", None) is not None, "running_jobs": running,
            "boot_id": _BOOT_ID, "app": _app_info()}


@app.post("/api/restart")
def api_restart():
    """Stop the server; main.py then starts a fresh process on the same port (new code, settings, models)."""
    server = getattr(app.state, "uvicorn_server", None)
    if server is None:
        raise HTTPException(409, "Перезапуск отсюда доступен, только если программа запущена через start.bat / main.py")
    app.state.restart_requested = True
    logger.info("restart requested from the settings page")

    def stop():
        time.sleep(0.5)   # let this response reach the browser first
        server.should_exit = True
    threading.Thread(target=stop, daemon=True).start()
    return {"ok": True}


def get_llm_client(provider_id: Optional[str] = None):
    """Client of the active provider (or of `provider_id`); raises a plain-Russian error for a blocked cloud one."""
    settings = config.load_settings()
    prof = providers.find_profile(settings, provider_id)
    if prof is None:
        raise providers.ProviderError("Провайдер не найден — выберите другой в Настройки → Модель LLM.")
    return providers.client_for(settings, prof)


def _client_for(prof: dict):
    """The same, via the (monkeypatchable) get_llm_client: the active provider uses the zero-argument call."""
    return get_llm_client() if prof["id"] == providers.active_id(config.load_settings()) else get_llm_client(prof["id"])


def _llm_plan(settings: dict, task: str) -> tuple[dict, str]:
    """(profile, model preference) for a task; HTTP 400 in Russian if the provider is missing / cloud is not allowed."""
    try:
        prof = providers.profile_for_task(settings, task)
        providers.check_allowed(settings, prof)
    except providers.ProviderError as exc:
        raise HTTPException(400, str(exc))
    return prof, providers.model_for_task(settings, prof, task)


def _eff_params(settings: dict, prof: dict) -> dict:
    """temperature / max_tokens: the provider's own override, else the general settings."""
    p = prof.get("params") or {}
    return {"temperature": p.get("temperature", settings.get("temperature", 0.2)),
            "max_tokens": p.get("max_tokens", settings.get("max_tokens", 1024)),
            "timeout": prof.get("timeout") or settings.get("llm_timeout", 120)}


def _provider_snapshot(settings: dict, prof: dict, model: str, stats: Optional[dict]) -> dict:
    cost = (stats or {}).get("cost")
    return {**providers.public_info(settings, prof, model), "cost": cost}


@app.middleware("http")
async def no_cache_for_ui(request, call_next):
    """The UI files change between versions: make browsers revalidate them every time
    (a stale cached app.js/index.html looked like 'buttons do nothing')."""
    resp = await call_next(request)
    if not request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.get("/api/db/status")
def api_db_status():
    """Plain-language health report of the active book's database (read-only)."""
    settings = config.load_settings()
    book_id = settings.get("active_book") or state.book_id or "urantia"
    busy = next(({"job_id": j.job_id, "type": j.type} for j in job_manager.list()
                 if j.type in ("index", "rebuild-fts", "build-glossary") and j.status in ("queued", "running")), None)
    return build_db_status(
        state.book_db_path(book_id), book_id,
        fts_weights=settings.get("fts_weights"),
        fts_mode_hybrid=settings.get("fts_mode_hybrid"),
        fts_mode_standalone=settings.get("fts_mode_standalone"),
        embedder_key=state.embedder_key, embedder_error=state.embedder_error, busy_job=busy,
    )


# ---------------------------------------------------------------- status ---

@app.get("/api/status")
def api_status():
    settings = config.load_settings()
    llm_ok = False
    llm_message = ""
    active = providers.find_profile(settings, None)
    cloud = providers.is_cloud(active["kind"], active["base_url"])
    try:
        if cloud:  # no network request on every status poll: a cloud provider is checked by its button
            llm_ok, llm_message = None, "Облачный провайдер: проверка по кнопке «Проверить соединение»"
            providers.check_allowed(settings, active)
        else:
            res = get_llm_client().check_connection()
            llm_ok, llm_message = res["ok"], res["message"]
    except Exception as exc:  # noqa: BLE001
        llm_ok, llm_message = False, str(exc)
    llm_active = {**providers.public_info(settings, active, settings.get("llm_model") or active.get("default_model") or None),
                  "blocked": cloud and not settings.get("allow_cloud_llm")}
    return {
        "db_loaded": state.conn is not None,
        "db_error": state.db_error,
        "active_book": state.book_id,
        "embedder_ready": state.embedder is not None,
        "embedder_error": state.embedder_error,
        "embedder_model": state.embedder_key[0] if state.embedder_key else None,
        "embedder_runtime": state.embedder_key[1] if state.embedder_key else None,
        "searcher_ready": state.searcher is not None,
        "lmstudio_reachable": llm_ok,
        "lmstudio_message": llm_message,
        "llm_active": llm_active,
        "settings": settings,
    }


# ---------------------------------------------------------------- search ---

def _search_experiment(q: str, mode: str, k: int, lang, timings: dict) -> dict:
    st = config.load_settings()
    return {
        "kind": "search", "app": {"version": __version__, "date": _dt.date.today().isoformat()},
        "options": {}, "prompts": {}, "prompt_texts": {}, "prompt_versions": {},
        "search_settings": {"fts_weights": st.get("fts_weights"), "fts_mode_hybrid": st.get("fts_mode_hybrid"),
                            "fts_mode_standalone": st.get("fts_mode_standalone"),
                            "fts_stopwords": st.get("fts_stopwords")},
        "retrieval": {"mode": mode, "k": k, "lang": lang, "fts_mode_used": timings.get("fts_mode_used")},
        "embedder": {"model": state.embedder_key[0] if state.embedder_key else None,
                     "runtime": state.embedder_key[1] if state.embedder_key else None},
        "timings": {k_: v for k_, v in timings.items() if isinstance(v, (int, float))},
    }


def _library_ids() -> list[str]:
    root = str(config.LIBRARY_DIR)
    if not os.path.isdir(root):
        return []
    return [n for n in sorted(os.listdir(root)) if os.path.exists(os.path.join(root, n, "book.db"))]


def _parse_ids(book_ids) -> Optional[list[str]]:
    if book_ids is None or not isinstance(book_ids, (str, list, tuple)):
        return None
    if isinstance(book_ids, str):
        book_ids = book_ids.split(",")
    return [b.strip() for b in book_ids if isinstance(b, str) and b.strip()]


def _resolve_books(book_ids=None) -> list[str]:
    """Books to search: the explicit list (unknown id -> 400), else the saved selection («search_books»;
    books that vanished are dropped), else just the active book."""
    lib = _library_ids()
    explicit = _parse_ids(book_ids)
    if explicit:
        bad = [b for b in explicit if b not in lib]
        if bad:
            raise HTTPException(400, f"Книга не найдена в библиотеке: {', '.join(bad)}")
        return list(dict.fromkeys(explicit))
    settings = config.load_settings()
    saved = [b for b in (settings.get("search_books") or []) if b in lib]
    if saved:
        return list(dict.fromkeys(saved))
    active = state.book_id or settings.get("active_book")
    return [active] if active else lib[:1]


def _handle_or_error(book_id: str) -> BookHandle:
    try:
        return state.handle(book_id)
    except KeyError:
        raise HTTPException(404, f"База книги не найдена: {book_id}")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"Ошибка открытия базы «{book_id}»: {exc}")


def _conn_for(book_id: Optional[str] = None):
    """Read connection of the given book; without an id - of the active one (the reading page, TOC, DB tab)."""
    if book_id and book_id != state.book_id:
        return _handle_or_error(book_id).conn
    if state.conn is None:
        raise HTTPException(503, state.db_error or "База не загружена")
    return state.conn


def _searcher_for(book_id: Optional[str] = None):
    if book_id and book_id != state.book_id:
        return _handle_or_error(book_id).searcher
    return state.searcher


def _book_tag(items: list[dict], book_id: str, book_name: str) -> list[dict]:
    for it in items:
        it["book_id"], it["book_name"] = book_id, book_name
    return items


@app.get("/api/search")
def api_search(q: str = Query(...), mode: str = Query("hybrid"), k: int = Query(10),
               lang: Optional[str] = Query(None), history: int = Query(1),
               book_ids: Optional[str] = None):
    """`book_ids`: comma-separated books to search (default: the saved selection). Several books ->
    every book is searched and the hits are merged by normalised score (see app/core/multi.py)."""
    if mode not in ("hybrid", "fts", "semantic"):
        raise HTTPException(400, f"Неизвестный режим поиска: {mode}")
    ids = _resolve_books(book_ids)
    t0 = time.monotonic()
    warnings: list[dict] = []
    if ids == [state.book_id]:
        if state.conn is None:
            raise HTTPException(503, state.db_error or "База не загружена")
        if state.searcher is None:
            raise HTTPException(503, "Поисковый индекс недоступен")
        fn = {"hybrid": state.searcher.hybrid, "fts": state.searcher.fts, "semantic": state.searcher.semantic}[mode]
        try:
            results = fn(q, k)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"Ошибка поиска: {exc}")
        timings = getattr(state.searcher, "last_timings", {})
        name = (core_db.get_book_info(state.conn, state.book_id) or {}).get("book_name") or state.book_id
        _book_tag(results, state.book_id, name)
    else:
        handles = []
        for b in ids:
            try:
                handles.append(state.handle(b))
            except Exception as exc:  # noqa: BLE001
                warnings.append({"book_id": b, "message": f"книга не открылась: {exc}"})
        if not any(h.searcher is not None for h in handles):
            raise HTTPException(503, "Поисковый индекс выбранных книг недоступен")
        try:
            results, timings, w = core_multi.search_books(handles, q, mode, k)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"Ошибка поиска: {exc}")
        warnings += w
    if history:
        n = len(results)
        _record_history("search", q, {"mode": mode, "k": k, "lang": lang, "book_ids": ids},
                        {"results": _compact(results), "timings": timings},
                        f"{n} {_plural(n, 'результат', 'результата', 'результатов')}" if n else "ничего не найдено",
                        time.monotonic() - t0, experiment=_search_experiment(q, mode, k, lang, timings),
                        book_id=ids[0] if ids else None)
    return {"results": results, "timings": timings, "book_ids": ids, "warnings": warnings}


@app.get("/api/paragraph/{para_id}")
def api_paragraph(para_id: str, book_id: Optional[str] = None):
    p = core_db.get_paragraph(_conn_for(book_id), para_id)
    if not p:
        raise HTTPException(404, "Абзац не найден")
    return p


@app.get("/api/paragraph/{para_id}/context")
def api_paragraph_context(para_id: str, before: int = 2, after: int = 2, book_id: Optional[str] = None):
    ctx = core_db.get_context(_conn_for(book_id), para_id, before, after)
    if not ctx:
        raise HTTPException(404, "Абзац не найден")
    return {"context": ctx}


@app.get("/api/toc")
def api_toc(book_id: Optional[str] = None):
    """Table of contents of the active book (or `book_id`): parts > papers > sections, titles in both languages."""
    return {"parts": core_db.get_toc(_conn_for(book_id))}


@app.get("/api/read/{paper}")
def api_read(paper: int, q: str = "", book_id: Optional[str] = None):
    """The whole text of one paper for the reading page. With `q`, also the words to highlight
    for that search query in both languages (`marks`); without it nothing is highlighted."""
    data = core_db.get_paper_text(_conn_for(book_id), paper)
    if not data:
        raise HTTPException(404, "Документ не найден")
    data["marks"] = {"ru": [], "en": []}
    sr = _searcher_for(book_id)
    if q.strip() and sr is not None:
        paras = data["paragraphs"]
        data["marks"] = sr.terms.query_marks(q, [p["ru"] for p in paras], [p["en"] for p in paras])
    return data


@app.get("/api/paragraph/{para_id}/pairs")
def api_paragraph_pairs(para_id: str, book_id: Optional[str] = None):
    """RU<->EN glossary pairs present in this paragraph, for the hover highlight in the result cards.
    Empty list (not an error) when the glossary has not been built."""
    p = core_db.get_paragraph(_conn_for(book_id), para_id)
    if not p:
        raise HTTPException(404, "Абзац не найден")
    sr = _searcher_for(book_id)
    if sr is None:
        return {"pairs": [], "glossary": False}
    tt = sr.terms
    return {"pairs": tt.paragraph_pairs(p.get("ru", ""), p.get("en", "")), "glossary": tt.glossary.exists}


@app.get("/api/papers")
def api_papers():
    if state.conn is None:
        raise HTTPException(503, state.db_error or "База не загружена")
    return {"papers": core_db.list_papers(state.conn)}


@app.get("/api/papers/{paper}/sections")
def api_sections(paper: int):
    if state.conn is None:
        raise HTTPException(503, state.db_error or "База не загружена")
    return {"sections": core_db.list_sections(state.conn, paper)}


# ---------------------------------------------------------------- jobs -----

class IndexJobRequest(BaseModel):
    json_path: str
    model_name: str
    scheme: str = "D"
    book_id: str
    book_name: str
    runtime: str = "auto"       # "auto" | "onnx" | "llamacpp-vulkan"
    device: str = "auto"        # "auto" | "cpu" | "gpu"
    vector_dtype: str = "int8"  # "int8" | "float32"
    embed_batch_size: int = 32  # smaller batches -> smoother progress reporting from the UI


@app.post("/api/jobs/index")
def api_job_index(req: IndexJobRequest):
    # The UI sends just a file name from /api/source-files: resolve it against the program
    # folder (not the process cwd, which differs when launched from elsewhere).
    json_path = req.json_path if os.path.isabs(req.json_path) else os.path.join(str(config.ROOT), req.json_path)
    if not os.path.isfile(json_path):
        raise HTTPException(400, f"Исходный файл не найден: {json_path}")
    if not _is_book_file(json_path):
        raise HTTPException(400, "Формат файла не поддерживается. Нужны: " + ", ".join(core_parser.BOOK_EXTENSIONS))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", req.book_id or ""):
        raise HTTPException(400, "ID книги: только латинские буквы, цифры, «_», «-» и «.»")

    def worker(job):
        progress = job_manager.make_progress_callback(job)
        cancel = job_manager.cancel_event(job)
        db_path = state.book_db_path(req.book_id)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        # Building this book_id's book.db might be the one currently open for search
        # (re-indexing the active/configured book): release our connection/searcher
        # *before* build_index atomically replaces the file, since an open sqlite3
        # handle can make os.replace()/MoveFileEx fail or silently block on Windows.
        is_active = req.book_id == (config.load_settings().get("active_book") or state.book_id)
        if is_active:
            state.release_for_reindex()
        state.drop_handle(req.book_id)   # a cached (non-active) copy would hold book.db open
        try:
            built_path = core_indexer.build_index(
                json_path, db_path, req.model_name, scheme=req.scheme,
                book_id=req.book_id, book_name=req.book_name,
                progress=progress, cancel=cancel,
                runtime=req.runtime, device=req.device, vector_dtype=req.vector_dtype,
                embed_batch_size=req.embed_batch_size,
            )
        except Exception as exc:  # noqa: BLE001
            # build_index raises RuntimeError("cancelled") on cancel and works on a temp file,
            # so the previous book.db (if any) is intact: reopen it so the app keeps working.
            if is_active and os.path.exists(db_path):
                state.book_id = req.book_id
                state.reload_searcher()
            if isinstance(exc, RuntimeError) and str(exc) == "cancelled":
                job.result = {"db_path": str(db_path), "db_size_bytes": None, "cancelled": True}
                return
            raise
        result = {
            "db_path": str(built_path),
            "db_size_bytes": os.path.getsize(built_path) if built_path and os.path.exists(built_path) else None,
            "cancelled": False,
        }
        job.result = result
        if is_active:
            state.book_id = req.book_id
            state.reload_searcher()

    job = job_manager.create(
        "index",
        ["parsing", "database", "lemmatize", "fts", "embedding", "finalize"],
        worker,
    )
    return {"job_id": job.job_id}


@app.post("/api/jobs/rebuild-fts")
def api_job_rebuild_fts():
    """Rebuild paragraphs_fts (new columns/titles/lemmas) from the existing paragraphs --
    embeddings are untouched. Runs while the app keeps book.db open; afterwards the
    Searcher is recreated (its FTS column detection is cached per instance)."""
    book_id = state.book_id or config.load_settings().get("active_book")
    db_path = state.book_db_path(book_id)
    if not os.path.exists(db_path):
        raise HTTPException(404, f"База книги не найдена: {db_path}")
    busy = [j for j in job_manager.list() if j.type in ("index", "rebuild-fts", "build-glossary")
            and j.status in ("queued", "running")]
    if busy:
        raise HTTPException(409, "Уже выполняется индексация или перестроение индекса — дождитесь завершения.")

    def worker(job):
        progress = job_manager.make_progress_callback(job)
        cancel = job_manager.cancel_event(job)
        try:
            n = core_indexer.rebuild_fts(db_path, progress=progress, cancel=cancel)
        except RuntimeError as exc:
            if str(exc) == "cancelled":  # old index is intact by design (single transaction)
                job.result = {"cancelled": True, "rows": 0}
                return
            raise
        job.result = {"cancelled": False, "rows": n}
        state.rebuild_searcher()

    job = job_manager.create("rebuild-fts", ["fts"], worker)
    return {"job_id": job.job_id}


@app.post("/api/jobs/build-glossary")
def api_job_build_glossary():
    """Build the RU<->EN glossary of terms (table term_glossary) from the parallel text of the active book.
    Does not touch paragraphs, the full-text index or vectors; the old glossary stays until the new one is ready."""
    book_id = state.book_id or config.load_settings().get("active_book")
    db_path = state.book_db_path(book_id)
    if not os.path.exists(db_path):
        raise HTTPException(404, f"База книги не найдена: {db_path}")
    busy = [j for j in job_manager.list() if j.type in ("index", "rebuild-fts", "build-glossary")
            and j.status in ("queued", "running")]
    if busy:
        raise HTTPException(409, "Уже выполняется другая задача с базой — дождитесь завершения.")

    def worker(job):
        progress = job_manager.make_progress_callback(job)
        cancel = job_manager.cancel_event(job)
        try:
            info = core_terms.build_glossary(db_path, progress=progress, cancel=cancel)
        except RuntimeError as exc:
            if str(exc) == "cancelled":  # the previous glossary is intact (single transaction)
                job.result = {"cancelled": True}
                return
            raise
        job.result = {"cancelled": False, **info}
        sr = state.searcher
        if sr is not None and getattr(sr, "_term_tools", None) is not None:
            sr._term_tools.reload()   # use the new glossary from the next question on

    job = job_manager.create("build-glossary", ["glossary"], worker)
    return {"job_id": job.job_id}


def _job_status(job):
    """A status line both in the job panel and in its log (e.g. «загружаю модель…»)."""
    def say(msg: str):
        job.message = msg
        job.add_log(msg)
    return say


class RagJobRequest(BaseModel):
    question: str
    k: Optional[int] = None          # None -> the «сколько абзацев передавать» setting
    replay: Optional[int] = None     # history entry id: re-run with EXACTLY the saved prompts/parameters
    overview: Optional[bool] = None  # True - «Обзор по всей книге»; None - the program decides
    book_ids: Optional[list[str]] = None   # books selected for search; the question is answered over ONE of them


def _replay_setup(settings: dict, replay_id: int, mode: str) -> tuple[dict, "prompts_mod.ResolvedPrompts", str]:
    """Settings + prompts rebuilt from a saved experiment snapshot (for «Повторить запрос»)."""
    e = hist.get(replay_id)
    if not e or e["mode"] != mode:
        raise HTTPException(404, "Запись истории для повтора не найдена")
    exp = e.get("experiment")
    if not exp or not exp.get("prompt_texts"):
        raise HTTPException(400, "Для этой записи параметры не сохранены — повторить точно нельзя. "
                                 "Используйте «Повторить с текущими настройками».")
    s = dict(settings)
    s.update({k: v for k, v in (exp.get("options") or {}).items() if k in opts_mod.SPEC})
    for k_src, k_dst in (("temperature", "temperature"), ("max_tokens", "max_tokens")):
        if exp.get(k_src) is not None:
            s[k_dst] = exp[k_src]
    if exp.get("model"):
        s["llm_model"] = exp["model"]
        s["llm_model_" + ("rag" if mode == "rag" else "translation")] = exp["model"]
    pv = (exp.get("provider") or {}).get("id")
    if pv and providers.find_profile(s, pv) is not None:  # the provider of the original run, if it still exists
        s["llm_provider_" + ("rag" if mode == "rag" else "translation")] = pv
    rs = (exp.get("reasoning") or {}).get("setting")
    if rs:
        s["reasoning_rag" if mode == "rag" else "reasoning_translation"] = rs
    # FTS weights / modes / stop words used originally (older snapshots may lack some keys)
    for key, val in (exp.get("search_settings") or {}).items():
        if key in _FTS_REPLAY_KEYS and val is not None:
            s[key] = val
    # the number of paragraphs really sent (the request's k, not necessarily the saved option)
    k_saved = (exp.get("retrieval") or {}).get("k")
    if isinstance(k_saved, int) and k_saved >= 1:
        s["rag_top_k"] = k_saved
    prompts = prompts_mod.ResolvedPrompts(exp["prompt_texts"], preset=exp.get("preset"), replay=True,
                                          sources=exp.get("prompt_versions"))
    return s, prompts, f"Точный повтор записи №{replay_id}: промты и параметры взяты из сохранённого снимка."


_FTS_REPLAY_KEYS = ("fts_weights", "fts_mode_hybrid", "fts_mode_standalone", "fts_stopwords")


def _cloud_params(result: dict) -> dict:
    """Marks a history row as a cloud run (icon in the list); only the provider NAME, never a key."""
    pr = result.get("provider") or {}
    return {"cloud": True, "provider": pr.get("name")} if pr.get("is_cloud") else {}


def _search_meta(settings: dict, reasoning_key: str) -> dict:
    return {"reasoning_setting": settings.get(reasoning_key) or "auto",
            "fts_weights": settings.get("fts_weights"), "fts_mode_hybrid": settings.get("fts_mode_hybrid"),
            "fts_mode_standalone": settings.get("fts_mode_standalone"),
            "fts_stopwords": settings.get("fts_stopwords")}


@app.post("/api/jobs/rag")
def api_job_rag(req: RagJobRequest):
    # A question is answered over ONE book: paragraph ids («1:0.1») repeat between books, so citations
    # would be ambiguous. With several books selected the active one is used if it is among them.
    ids = _resolve_books(req.book_ids)
    rag_book = state.book_id if state.book_id in ids else ids[0]
    rag_handle = _handle_or_error(rag_book)
    if rag_handle.searcher is None:
        raise HTTPException(503, "Поисковый индекс недоступен")
    multi_note = (f"Выбрано книг: {len(ids)}. Вопрос к тексту отвечает по одной книге — «{rag_handle.book_name}»; "
                  "поиск по нескольким книгам доступен в режиме «Обычный поиск».") if len(ids) > 1 else None
    settings = config.load_settings()
    prompts = prompts_mod.from_settings(settings)
    replay_note = None
    if req.replay is not None:
        settings, prompts, replay_note = _replay_setup(settings, req.replay, "rag")
    prof, model_pref = _llm_plan(settings, "rag")
    rw_plan = None
    if opts_mod.effective(settings)["rag_query_rewrite"]:
        rw_plan = _llm_plan(settings, "rewrite")

    def worker(job):
        progress = job_manager.make_progress_callback(job)
        client = _client_for(prof)
        client.on_status = _job_status(job)   # LM Studio may have to load the model first: show it
        # Resolve the model actually usable at the provider (for LM Studio: the LOADED one, or the configured one
        # loaded automatically when nothing is loaded) and its context length, used for context fitting;
        # raises a clear error -> job error.
        resolved = client.resolve_model(model_pref)
        if resolved["note"]:
            job.add_log(resolved["note"])
        rw_client, rw_model, rw_info = None, None, None
        if rw_plan and (rw_plan[0]["id"], rw_plan[1]) != (prof["id"], model_pref):
            rw_client = _client_for(rw_plan[0])
            rw_client.on_status = _job_status(job)
            rw_model = rw_client.resolve_model(rw_plan[1])["model"]
            rw_info = providers.public_info(settings, rw_plan[0], rw_model)
            job.add_log(f"Подготовка запроса: {rw_plan[0]['name']} · {rw_model}")
        job.add_log(("Облако: " if providers.is_cloud(prof["kind"], prof["base_url"]) else "Локально: ")
                    + f"{prof['name']} · {resolved['model']}")
        for w in prompts.warnings:
            job.add_log(w)
        if replay_note:
            job.add_log(replay_note)
        options = opts_mod.effective(settings)
        searcher = rag_handle.searcher
        if multi_note:
            job.add_log(multi_note)
        if req.replay is not None and hasattr(searcher, "with_overrides"):
            # exact replay: the FTS weights/modes/stop words of the ORIGINAL run, for this run only
            searcher = searcher.with_overrides(
                fts_weights=settings.get("fts_weights"), fts_mode_hybrid=settings.get("fts_mode_hybrid"),
                fts_mode_standalone=settings.get("fts_mode_standalone"), fts_stopwords=settings.get("fts_stopwords"))
        ep = _eff_params(settings, prof)
        pipeline = RagPipeline(
            searcher, client, model=resolved["model"],
            temperature=ep["temperature"], max_tokens=ep["max_tokens"],
            timeout=ep["timeout"], rewrite_llm=rw_client, rewrite_model=rw_model,
            max_context_length=resolved["context"] or 8192,
            reasoning_effort=resolve_reasoning_effort(options["reasoning_rag"], auto_default="none"),
            options=options, prompts=prompts, conn=rag_handle.conn,
            meta=_search_meta(settings, "reasoning_rag"),
        )
        t_start = time.monotonic()
        # replay: the saved k; otherwise the request's k, else the «сколько абзацев» setting
        k_eff = options["rag_top_k"] if req.replay is not None else (req.k or options["rag_top_k"])
        result = pipeline.answer(req.question, k=k_eff, progress=progress, overview=req.overview)
        result["model_note"] = resolved["note"]
        result["replay_note"] = replay_note
        result["provider"] = _provider_snapshot(settings, prof, resolved["model"], result.get("stats"))
        if isinstance(result.get("experiment"), dict):
            result["experiment"]["provider"] = result["provider"]
            result["experiment"]["rewrite_provider"] = rw_info
        job.result = result
        partial = bool(result.get("error") and result.get("answer"))
        if result.get("error") and not partial:
            raise RuntimeError(result["error"])
        n_cited = len(result.get("cited_ids") or [])
        _record_history(
            "rag", req.question, {"k": k_eff, "replay": req.replay, "book_ids": ids, **_cloud_params(result),
                                  **({"overview": True} if result.get("overview") else {})},
            {**{k: result.get(k) for k in ("answer", "cited_ids", "unknown_citations", "no_answer", "has_inference",
                                          "plan", "candidates", "stats_summary", "model_note", "prompt_warnings",
                                          "warnings", "provider", "truncated", "error", "answered_model",
                                          "fallback_note", "overview", "followup")},
             "sources": _compact(_book_tag(list(result.get("sources") or []), rag_book, rag_handle.book_name))},
            "ответ оборван на середине" if partial else
            "в книге нет данных для ответа" if result.get("no_answer")
            else f"ответ, {n_cited} {_plural(n_cited, 'источник', 'источника', 'источников')}",
            time.monotonic() - t_start, experiment=result.get("experiment"), book_id=rag_book)
        if partial:   # the job is an error (the reason is shown), the partial answer stays in job.result
            raise RuntimeError(result["error"])

    job = job_manager.create("rag", ["analysis", "fts", "semantic", "hybrid", "context", "generation", "followup"],
                             worker)
    return {"job_id": job.job_id}


class TranslationJobRequest(BaseModel):
    reference: str
    replay: Optional[int] = None


@app.post("/api/jobs/translation-analysis")
def api_job_translation(req: TranslationJobRequest):
    if state.conn is None:
        raise HTTPException(503, state.db_error or "База не загружена")
    ref = parse_reference(req.reference)
    if not ref:
        raise HTTPException(400, f"Не удалось разобрать ссылку: {req.reference}")
    settings = config.load_settings()
    prompts = prompts_mod.from_settings(settings)
    replay_note = None
    if req.replay is not None:
        settings, prompts, replay_note = _replay_setup(settings, req.replay, "translation")
    prof, model_pref = _llm_plan(settings, "translation")

    def worker(job):
        progress = job_manager.make_progress_callback(job)
        client = _client_for(prof)
        client.on_status = _job_status(job)
        resolved = client.resolve_model(model_pref)
        if resolved["note"]:
            job.add_log(resolved["note"])
        job.add_log(("Облако: " if providers.is_cloud(prof["kind"], prof["base_url"]) else "Локально: ")
                    + f"{prof['name']} · {resolved['model']}")
        for w in prompts.warnings:
            job.add_log(w)
        if replay_note:
            job.add_log(replay_note)
        options = opts_mod.effective(settings)
        ep = _eff_params(settings, prof)
        analyzer = TranslationAnalyzer(
            state.conn, client, model=resolved["model"],
            temperature=0.1, max_tokens=ep["max_tokens"] if "max_tokens" in (prof.get("params") or {})
            else settings.get("max_tokens", 2048),
            timeout=ep["timeout"],
            max_context_length=resolved["context"] or 8192,
            reasoning_effort=resolve_reasoning_effort(options["reasoning_translation"], auto_default="none"),
            options=options, prompts=prompts, meta=_search_meta(settings, "reasoning_translation"),
        )
        t_start = time.monotonic()
        result = analyzer.analyze(ref.paper, ref.section, ref.para_from, ref.para_to,
                                  progress=progress)
        result["model_note"] = resolved["note"]
        result["replay_note"] = replay_note
        result["provider"] = _provider_snapshot(settings, prof, resolved["model"],
                                                (result.get("experiment") or {}).get("stats"))
        if isinstance(result.get("experiment"), dict):
            result["experiment"]["provider"] = result["provider"]
        job.result = result
        if result.get("error"):
            raise RuntimeError(result["error"])
        n_issues = len(result.get("issues") or [])
        stored = {k: v for k, v in result.items() if k != "experiment"}  # the snapshot has its own column
        _record_history(
            "translation", req.reference, {"reference": req.reference, "replay": req.replay, **_cloud_params(result)}, stored,
            f"{n_issues} {_plural(n_issues, 'замечание', 'замечания', 'замечаний')}" if n_issues else "замечаний нет",
            time.monotonic() - t_start, experiment=result.get("experiment"))

    job = job_manager.create("translation-analysis",
                              ["Сравнение EN/RU", "Анализ смысловых расхождений", "Формирование отчёта"],
                              worker)
    return {"job_id": job.job_id}


@app.get("/api/jobs")
def api_jobs_list():
    return {"jobs": [j.to_dict() for j in job_manager.list()]}


@app.get("/api/jobs/{job_id}")
def api_job_get(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(404, "Задача не найдена")
    return job.to_dict()


@app.get("/api/jobs/{job_id}/events")
def api_job_events(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(404, "Задача не найдена")

    def gen():
        q = job_manager.subscribe(job_id)
        try:
            yield f"data: {json.dumps(job.to_dict())}\n\n"
            while True:
                try:
                    data = q.get(timeout=1.0)
                    yield f"data: {json.dumps(data)}\n\n"
                    if data["status"] in ("done", "error", "cancelled"):
                        break
                except Exception:  # queue.Empty and others
                    current = job_manager.get(job_id)
                    if current is None:
                        break
                    if current.status in ("done", "error", "cancelled"):
                        yield f"data: {json.dumps(current.to_dict())}\n\n"
                        break
        finally:
            job_manager.unsubscribe(job_id, q)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/jobs/{job_id}/cancel")
def api_job_cancel(job_id: str):
    if not job_manager.cancel(job_id):
        raise HTTPException(404, "Задача не найдена")
    return {"ok": True}


@app.get("/api/jobs/{job_id}/translation-report.md")
def api_job_translation_report(job_id: str):
    job = job_manager.get(job_id)
    if not job or not job.result:
        raise HTTPException(404, "Отчёт не найден")
    md = export_markdown(job.result, title=f"Анализ перевода — задача {job_id}")
    return StreamingResponse(iter([md]), media_type="text/markdown")


# ------------------------------------------------------------- settings ---

@app.get("/api/settings")
def api_settings_get():
    return config.load_settings()


_FTS_MODES = ("and", "or", "and_fallback_or")
_FTS_WEIGHT_KEYS = ("en", "ru", "ru_lemma", "title_en", "title_ru")


_EXTRA_SETTING_KEYS = {"prompts", "prompt_preset"}
_PROVIDER_ONLY_KEYS = ("llm_providers", "active_llm_provider")


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")


def _check_basic_settings(p: dict) -> None:
    """Types and ranges of the plain (non «Параметры ответа») settings; raises HTTP 400 in Russian."""
    for key, lo, hi, label in (("temperature", 0, 2, "Температура"), ("max_tokens", 1, 200000, "Максимум токенов"),
                               ("llm_timeout", 1, 36000, "Таймаут, с")):
        if key in p:
            v = p[key]
            if not _is_num(v):
                raise HTTPException(400, f"{label}: ожидается число")
            if not lo <= v <= hi:
                raise HTTPException(400, f"{label}: допустимо от {lo} до {hi}")
            if key != "temperature" and v != int(v):
                raise HTTPException(400, f"{label}: ожидается целое число")
    for key, label in (("lmstudio_url", "Адрес LM Studio"), ("llm_model", "Модель"),
                       ("embedding_model", "Модель эмбеддингов"), ("active_book", "Книга")):
        if key in p and p[key] is not None and not isinstance(p[key], str):
            raise HTTPException(400, f"{label}: ожидается строка")
    for key in _PROVIDER_ONLY_KEYS:
        if key in p:
            raise HTTPException(400, "Провайдеры меняются в разделе «Модель LLM → Провайдеры», не через общие настройки")
    if "allow_cloud_llm" in p:
        if not isinstance(p["allow_cloud_llm"], bool):
            raise HTTPException(400, "Облачные модели: ожидается «вкл» или «выкл»")
        if p["allow_cloud_llm"] and not config.load_settings().get("allow_cloud_llm"):
            raise HTTPException(400, "Облачные модели включаются только в настройках провайдеров, с подтверждением")
    all_ids = {x["id"] for x in providers.get_profiles(config.load_settings())}
    for t in providers.TASKS:
        v = p.get(f"llm_provider_{t}")
        if f"llm_provider_{t}" in p and (not isinstance(v, str) or (v and v not in all_ids)):
            raise HTTPException(400, "Провайдер для задачи: выберите существующий или «как в активном»")
        if f"llm_model_{t}" in p and not isinstance(p[f"llm_model_{t}"], str):
            raise HTTPException(400, "Модель для задачи: ожидается строка")
    for key in ("openrouter_referer", "openrouter_title"):
        if key in p and (not isinstance(p[key], str) or len(p[key]) > 200 or "\n" in p[key]):
            raise HTTPException(400, "Заголовок OpenRouter: строка до 200 символов")
    if "llm_hidden_models" in p and not (isinstance(p["llm_hidden_models"], list)
                                         and all(isinstance(x, str) for x in p["llm_hidden_models"])):
        raise HTTPException(400, "Скрытые модели: ожидается список названий")
    if "fts_stopwords" in p and not isinstance(p["fts_stopwords"], bool):
        raise HTTPException(400, "Стоп-слова в поиске: ожидается «вкл» или «выкл»")
    if "prompt_preset" in p and p["prompt_preset"] is not None and not isinstance(p["prompt_preset"], str):
        raise HTTPException(400, "prompt_preset: ожидается строка")


@app.post("/api/settings")
def api_settings_post(payload: dict):
    # validate FTS search settings (Settings -> Search) before persisting anything
    payload = dict(payload)
    known = set(config.DEFAULT_SETTINGS) | set(opts_mod.SPEC) | set(config.load_settings()) | _EXTRA_SETTING_KEYS
    for key in payload:
        if key not in known:
            raise HTTPException(400, f"Неизвестная настройка «{key}» — не сохранено")
    _check_basic_settings(payload)
    for key in list(payload):
        if key in opts_mod.SPEC:  # «Параметры ответа»: validated and normalised (types, min/max)
            try:
                payload[key] = opts_mod.coerce(key, payload[key])
            except ValueError as exc:
                raise HTTPException(400, str(exc))
    if "prompts" in payload:
        if not isinstance(payload["prompts"], dict):
            raise HTTPException(400, "prompts: ожидается объект {промт: текст}")
        for pid, text in payload["prompts"].items():
            err = prompts_mod.validate(pid, text)
            if err:
                raise HTTPException(400, f"Промт «{pid}»: {err}")
    for key in ("fts_mode_hybrid", "fts_mode_standalone"):
        if key in payload and payload[key] not in _FTS_MODES:
            raise HTTPException(400, f"{key}: допустимо одно из {', '.join(_FTS_MODES)}")
    if "fts_weights" in payload:
        w = payload["fts_weights"]
        if not isinstance(w, dict):
            raise HTTPException(400, "fts_weights: ожидается объект {колонка: вес}")
        clean = {}
        for k, v in w.items():
            if k not in _FTS_WEIGHT_KEYS:
                raise HTTPException(400, f"fts_weights: неизвестная колонка {k!r}")
            try:
                fv = float(v)
            except (TypeError, ValueError):
                raise HTTPException(400, f"fts_weights.{k}: ожидается число")
            if fv < 0:
                raise HTTPException(400, f"fts_weights.{k}: вес не может быть отрицательным")
            clean[k] = fv
        payload = {**payload, "fts_weights": {**(config.load_settings().get("fts_weights") or {}), **clean}}
    merged = config.load_settings()
    merged.update(payload)
    if isinstance(merged.get("llm_providers"), list):  # keep the profile in step with the old single-provider keys
        act = providers.active_id(merged)
        for pr in merged["llm_providers"]:
            if "llm_model" in payload and pr["id"] == act:
                pr["default_model"] = payload["llm_model"] or ""
            if "lmstudio_url" in payload and pr["id"] == providers.LOCAL_ID and payload["lmstudio_url"]:
                pr["base_url"] = payload["lmstudio_url"]
    config.save_settings(merged)
    if "active_book" in payload:
        state.load_book(payload["active_book"])
    elif any(k in payload for k in ("fts_weights", "fts_mode_hybrid", "fts_mode_standalone")):
        state.rebuild_searcher()
    return config.load_settings()


# ------------------------------------------------------- LLM providers ---

CLOUD_WARNING = ("Тексты абзацев книги и ваши вопросы будут отправляться на внешний сервер {host}. "
                 "Не включайте, если это не допускается.")


def _prof_or_404(settings: dict, pid: Optional[str]) -> dict:
    prof = providers.find_profile(settings, pid)
    if prof is None:
        raise HTTPException(404, "Провайдер не найден")
    return prof


def _public_profile(settings: dict, prof: dict) -> dict:
    kind = providers.KINDS[prof["kind"]]
    cloud = providers.is_cloud(prof["kind"], prof["base_url"])
    is_active = prof["id"] == providers.active_id(settings)
    return {**prof, "host": providers.profile_host(prof), "is_cloud": cloud, "active": is_active,
            "icon": "☁" if cloud else "🖥", "kind_title": kind["title"], "needs_key": kind["needs_key"],
            "model": (settings.get("llm_model") or prof.get("default_model") or "") if is_active
            else prof.get("default_model") or "",
            **secrets_store.key_info(prof["id"], prof["kind"], prof["base_url"])}


def _providers_state(settings: Optional[dict] = None) -> dict:
    settings = settings or config.load_settings()
    return {"providers": [_public_profile(settings, p) for p in providers.get_profiles(settings)],
            "active": providers.active_id(settings), "allow_cloud_llm": bool(settings.get("allow_cloud_llm")),
            "kinds": [{"id": k, **{f: v[f] for f in ("title", "icon", "default_url", "needs_key", "desc")}}
                      for k, v in providers.KINDS.items()],
            "protection": secrets_store.protection(), "cloud_warning": CLOUD_WARNING}


def _save_profiles(profiles: list[dict], active: Optional[str] = None, extra: Optional[dict] = None) -> dict:
    merged = config.load_settings()
    merged["llm_providers"] = [{k: p.get(k) for k in providers.PROFILE_FIELDS} for p in profiles]
    if active is not None:
        merged["active_llm_provider"] = active
        ap = next(p for p in profiles if p["id"] == active)
        merged["llm_model"] = ap.get("default_model") or ""
        if ap["id"] == providers.LOCAL_ID:
            merged["lmstudio_url"] = ap["base_url"]
    merged.update(extra or {})
    config.save_settings(merged)
    return config.load_settings()


def _client_or_error(pid: Optional[str], check_gate: bool = True):
    settings = config.load_settings()
    prof = _prof_or_404(settings, pid)
    try:
        if check_gate:
            providers.check_allowed(settings, prof)
        return providers.client_for(settings, prof) if check_gate else None
    except providers.ProviderError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/llm/providers")
def api_providers_list():
    return _providers_state()


def _apply_key(pid: str, body: dict) -> None:
    if body.get("clear_key"):
        secrets_store.delete_key(pid)
    key = body.get("api_key")
    if isinstance(key, str) and key.strip():
        try:
            secrets_store.set_key(pid, key)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    elif key is not None and not isinstance(key, str):
        raise HTTPException(400, "Ключ API: ожидается строка")


@app.post("/api/llm/providers")
def api_providers_create(body: dict):
    settings = config.load_settings()
    try:
        clean = providers.validate_profile(body)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    profs = providers.get_profiles(settings)
    if len(profs) >= 30:
        raise HTTPException(400, "Слишком много провайдеров (максимум 30)")
    pid = providers.make_id(clean["name"], {p["id"] for p in profs})
    prof = providers.normalize_profile({**clean, "id": pid})
    _apply_key(pid, body)
    st = _save_profiles(profs + [prof], extra={"active_llm_provider": providers.active_id(settings)})
    return {"id": pid, **_providers_state(st)}


@app.put("/api/llm/providers/{pid}")
def api_providers_update(pid: str, body: dict):
    settings = config.load_settings()
    profs = providers.get_profiles(settings)
    cur = next((p for p in profs if p["id"] == pid), None)
    if cur is None:
        raise HTTPException(404, "Провайдер не найден")
    fields = {k: v for k, v in body.items() if k in providers.PROFILE_FIELDS and k not in ("id", "kind")}
    try:
        clean = providers.validate_profile({**fields, "kind": cur["kind"]}, partial=True)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    clean.pop("kind", None)
    profs = [{**cur, **clean} if p["id"] == pid else p for p in profs]
    _apply_key(pid, body)
    act = providers.active_id(settings)
    st = _save_profiles(profs, active=act if pid == act and "default_model" in clean else None,
                        extra={"active_llm_provider": act})
    return {"id": pid, **_providers_state(st)}


@app.delete("/api/llm/providers/{pid}")
def api_providers_delete(pid: str):
    settings = config.load_settings()
    profs = providers.get_profiles(settings)
    if not any(p["id"] == pid for p in profs):
        raise HTTPException(404, "Провайдер не найден")
    if len(profs) <= 1:
        raise HTTPException(400, "Нельзя удалить последнего провайдера")
    rest = [p for p in profs if p["id"] != pid]
    act = providers.active_id(settings)
    extra = {f"llm_provider_{t}": "" for t in providers.TASKS if settings.get(f"llm_provider_{t}") == pid}
    new_active = None
    if act == pid:
        new_active = next((p for p in rest if not providers.is_cloud(p["kind"], p["base_url"])), rest[0])["id"]
    secrets_store.delete_key(pid)
    st = _save_profiles(rest, active=new_active, extra={**extra, "active_llm_provider": new_active or act})
    return _providers_state(st)


@app.post("/api/llm/providers/{pid}/activate")
def api_providers_activate(pid: str):
    settings = config.load_settings()
    prof = _prof_or_404(settings, pid)
    try:
        providers.check_allowed(settings, prof)
    except providers.ProviderError as exc:
        raise HTTPException(400, str(exc))
    st = _save_profiles(providers.get_profiles(settings), active=pid)
    return _providers_state(st)


@app.post("/api/llm/cloud-access")
def api_cloud_access(body: dict):
    """Switch the «allow cloud models» master setting. Turning ON needs an explicit confirmation flag."""
    settings = config.load_settings()
    allow = body.get("allow")
    if not isinstance(allow, bool):
        raise HTTPException(400, "allow: ожидается «вкл» или «выкл»")
    if allow and body.get("confirm") is not True:
        raise HTTPException(400, "Нужно подтверждение: " + CLOUD_WARNING.format(host="облачного провайдера"))
    profs = providers.get_profiles(settings)
    act = providers.active_id(settings)
    new_active = None
    if not allow:  # back to offline: an active cloud provider is replaced by a local one
        ap = next(p for p in profs if p["id"] == act)
        if providers.is_cloud(ap["kind"], ap["base_url"]):
            new_active = next((p["id"] for p in profs if not providers.is_cloud(p["kind"], p["base_url"])), None)
    st = _save_profiles(profs, active=new_active, extra={"allow_cloud_llm": allow, "active_llm_provider": new_active or act})
    return _providers_state(st)


@app.post("/api/llm/providers/{pid}/check")
def api_providers_check(pid: str):
    try:
        return _client_or_error(pid).check_connection()
    except HTTPException as exc:
        return {"ok": False, "message": exc.detail}


@app.post("/api/llm/providers/{pid}/test")
def api_providers_test(pid: str, body: Optional[dict] = None):
    settings = config.load_settings()
    prof = _prof_or_404(settings, pid)
    try:
        client = _client_or_error(pid)
        model = (body or {}).get("model") or providers.model_for_task(settings, prof, None)
        if not model or prof["kind"] == "lmstudio":
            model = client.resolve_model(model or None)["model"]
        return client.test_completion(model)
    except HTTPException as exc:
        return {"ok": False, "message": exc.detail}
    except LMStudioError as exc:
        return {"ok": False, "message": str(exc)}


@app.get("/api/llm/providers/{pid}/account")
def api_providers_account(pid: str):
    """OpenRouter: state of the account behind the key (free tier / spent / limit), in plain words.
    Only when cloud models are allowed and a key exists; never raises - the card just shows the reason."""
    settings = config.load_settings()
    prof = _prof_or_404(settings, pid)
    if prof["kind"] != "openrouter":
        return {"available": False, "reason": "Состояние счёта показывается только для OpenRouter."}
    if not settings.get("allow_cloud_llm"):
        return {"available": False, "reason": "Облачные модели выключены — к сервису никто не обращается."}
    if not secrets_store.key_info(prof["id"], prof["kind"], prof["base_url"]).get("has_key"):
        return {"available": False, "reason": "Ключ не введён."}
    try:
        return providers.client_for(settings, prof).account_state()
    except LMStudioError as exc:
        return {"available": False, "reason": str(exc)}


@app.get("/api/llm/check")
def api_llm_check(provider: Optional[str] = None):
    try:
        return _client_or_error(provider).check_connection()
    except HTTPException as exc:
        if exc.status_code == 404:
            raise
        return {"ok": False, "message": exc.detail}


class SwitchModelRequest(BaseModel):
    provider: Optional[str] = None
    model: str


@app.post("/api/llm/switch-model")
def api_llm_switch_model(req: SwitchModelRequest):
    """LM Studio: unload the current model and load the chosen one (as a job: loading takes a while)."""
    client = _client_or_error(req.provider)
    if getattr(client, "kind", None) != "lmstudio":
        return {"job_id": None}   # other providers pick the model per request: nothing to load

    def worker(job):
        def say(msg: str):
            job.message = msg
            job.add_log(msg)
        client.switch_lmstudio_model(req.model, say)
        job.message = f"Модель «{req.model}» загружена"

    job = job_manager.create("llm-switch", ["Смена модели в LM Studio"], worker)
    return {"job_id": job.job_id}


@app.get("/api/llm/models")
def api_llm_models(provider: Optional[str] = None, q: str = "", free_only: bool = False, min_context: int = 0,
                   refresh: bool = False, text_only: bool = False):
    client = _client_or_error(provider)
    try:
        try:
            models = client.list_models(refresh=refresh)
        except TypeError:  # clients without caching
            models = client.list_models()
    except LMStudioError as exc:
        raise HTTPException(503, str(exc))
    total = len(models)
    models = providers.filter_models(models, q, free_only, min_context, text_only)
    if text_only:  # for the picker on the main page: the user's hidden models out, the model in memory first
        hidden = {h.strip().lower() for h in config.load_settings().get("llm_hidden_models") or []}
        models = [m for m in models if str(m.get("id") or "").lower() not in hidden]
        if any(m.get("loaded") is not None for m in models):   # LM Studio; others (Claude CLI) keep their own order
            models = sorted(models, key=lambda m: (m.get("loaded") is not True, str(m.get("id")).lower()))
    return {"models": models, "total": total,
            "provider": client.profile["id"] if hasattr(client, "profile") else provider}


@app.get("/api/embeddings/models")
def api_embedding_models():
    models = core_embedder.list_available_models()
    for m in models:
        m["runtime_compatibility"] = core_embedder.model_runtime_compatibility(m["name"])
    return {"models": models}


@app.get("/api/embeddings/runtimes")
def api_embedding_runtimes():
    return {"runtimes": core_embedder.list_runtimes()}


class EmbeddingLoadRequest(BaseModel):
    model_name: str
    runtime: str = "auto"
    device: str = "auto"


@app.post("/api/embeddings/load")
def api_embedding_load(req: EmbeddingLoadRequest):
    try:
        state.ensure_embedder(model_name=req.model_name, runtime=req.runtime, device=req.device)
        if state.embedder is None:
            raise RuntimeError(state.embedder_error or "неизвестная ошибка")
        state._rebuild_searcher()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Не удалось загрузить модель: {exc}")
    settings = config.load_settings()
    settings["embedding_model"] = req.model_name
    settings["embedding_runtime"] = req.runtime
    settings["device"] = req.device
    config.save_settings(settings)
    return {
        "ok": True,
        "dim": getattr(state.embedder, "dim", None),
        "status": getattr(state.embedder, "status", None),
        "runtime": getattr(state.embedder, "runtime", None),
    }


# ---------------------------------------------------------------- history ---

def _plural(n: int, one: str, few: str, many: str) -> str:
    n10, n100 = n % 10, n % 100
    if n10 == 1 and n100 != 11:
        return one
    if 2 <= n10 <= 4 and not 12 <= n100 <= 14:
        return few
    return many


def _history_book() -> Optional[str]:
    return state.book_id or config.load_settings().get("active_book")


def _record_history(mode: str, query: str, params: dict, result: dict, summary: str, duration_s: float,
                    experiment: Optional[dict] = None, book_id: Optional[str] = None):
    """Saving history must never break a search/job: any problem is only logged."""
    try:
        hist.add(book_id or _history_book(), mode, query, params, result, summary, duration_s, experiment=experiment)
    except Exception:  # noqa: BLE001
        logger.warning("could not save history entry", exc_info=True)


_PARA_KEEP = ("id", "score", "fts_rank", "sem_rank", "match_lang", "snippet_en", "snippet_ru", "book_id", "book_name")


def _compact(items: list[dict]) -> list[dict]:
    """What is needed to redraw a result list; paragraph text is re-fetched by id on open."""
    return [{k: it.get(k) for k in _PARA_KEEP if it.get(k) is not None} for it in items]


def _conn_quiet(book_id: Optional[str]):
    """Connection of a book for redrawing history, None when it cannot be opened (never raises)."""
    if not book_id or book_id == state.book_id:
        return state.conn
    try:
        return state.handle(book_id).conn
    except Exception:  # noqa: BLE001
        return None


def _hydrate(items: list[dict], default_book: Optional[str] = None) -> tuple[list[dict], int]:
    out, missing = [], 0
    for it in items:
        conn = _conn_quiet(it.get("book_id") or default_book)
        p = core_db.get_paragraph(conn, it["id"]) if conn is not None else None
        if p is None:
            missing += 1
            p = {"id": it["id"], "paper": None, "section": None, "paragraph": None, "en": "", "ru": ""}
        out.append({"book_id": it.get("book_id") or default_book, **p, **it})
    return out, missing


@app.get("/api/history")
def api_history_list(mode: Optional[str] = None, q: Optional[str] = None, book: Optional[str] = None,
                     limit: int = 50, offset: int = 0, rating: Optional[str] = None, sort: str = "date"):
    return hist.list_entries(mode=mode or None, q=q or None, book=book or None, limit=limit, offset=offset,
                             rating=rating or None, sort=sort)


@app.get("/api/history/compare")
def api_history_compare(a: int, b: int):
    ea, eb = hist.get(a), hist.get(b)
    if not ea or not eb:
        raise HTTPException(404, "Одна из записей истории не найдена")
    out = compare_entries(ea, eb)
    return {"a": _entry_for_view(ea), "b": _entry_for_view(eb), **out}


@app.get("/api/history/{entry_id}")
def api_history_get(entry_id: int):
    e = hist.get(entry_id)
    if not e:
        raise HTTPException(404, "Запись истории не найдена")
    return _entry_for_view(e)


def _entry_for_view(e: dict) -> dict:
    res = e.get("result") or {}
    missing = 0
    used = (e.get("params") or {}).get("book_ids") or [e["book_id"]]
    e["book_loaded"] = all(_conn_quiet(b) is not None for b in used)
    if e["mode"] == "search":
        items, missing = _hydrate(res.get("results", []), e["book_id"])
        e["result"] = {"results": items, "timings": res.get("timings", {})}
    elif e["mode"] == "rag":
        items, missing = _hydrate(res.get("sources", []), e["book_id"])
        e["result"] = {**res, "sources": items, "error": None}
    e["missing_paragraphs"] = missing
    return e


@app.delete("/api/history/{entry_id}")
def api_history_delete(entry_id: int):
    if not hist.delete(entry_id):
        raise HTTPException(404, "Запись истории не найдена")
    return {"ok": True}


class PinRequest(BaseModel):
    pinned: Optional[bool] = None


@app.post("/api/history/{entry_id}/pin")
def api_history_pin(entry_id: int, req: PinRequest = PinRequest()):
    new = hist.set_pinned(entry_id, req.pinned)
    if new is None:
        raise HTTPException(404, "Запись истории не найдена")
    return {"ok": True, "pinned": new}


@app.delete("/api/history")
def api_history_clear():
    """Clear the history; pinned entries are kept."""
    return {"deleted": hist.clear_unpinned()}


@app.get("/api/history/{entry_id}/report.md")
def api_history_report(entry_id: int):
    e = hist.get(entry_id)
    if not e or e["mode"] != "translation" or not e.get("result"):
        raise HTTPException(404, "Отчёт не найден")
    md = export_markdown(e["result"], title=f"Анализ перевода — {e['query']}")
    return StreamingResponse(iter([md]), media_type="text/markdown")


# ----------------------------------------------------- options, prompts, presets ---

@app.get("/api/options")
def api_options():
    """«Параметры ответа»: spec (groups, ranges, hints) + the current values."""
    return opts_mod.public_spec(config.load_settings())


@app.get("/api/prompts")
def api_prompts():
    return prompts_mod.public_list(config.load_settings())


class PromptText(BaseModel):
    text: str


def _save_prompt_overrides(overrides: dict, preset="keep"):
    settings = config.load_settings()
    merged = {**settings, "prompts": overrides}
    if preset != "keep":
        merged["prompt_preset"] = preset
    config.save_settings(merged)
    return config.load_settings()


@app.put("/api/prompts/{pid}")
def api_prompt_save(pid: str, req: PromptText):
    sp = prompts_mod.specs().get(pid)
    if sp is None:
        raise HTTPException(404, "Такого промта нет")
    err = prompts_mod.validate(pid, req.text)
    if err:
        raise HTTPException(400, f"Промт «{sp['title']}» не сохранён: {err}")
    settings = config.load_settings()
    overrides = dict(settings.get("prompts") or {})
    if not req.text.strip() or req.text == sp["default"]:
        overrides.pop(pid, None)  # back to the default: no override stored
    else:
        overrides[pid] = req.text
    return next(p for p in prompts_mod.public_list(_save_prompt_overrides(overrides))["prompts"] if p["id"] == pid)


@app.post("/api/prompts/{pid}/reset")
def api_prompt_reset(pid: str):
    if pid not in prompts_mod.specs():
        raise HTTPException(404, "Такого промта нет")
    overrides = dict(config.load_settings().get("prompts") or {})
    overrides.pop(pid, None)
    return next(p for p in prompts_mod.public_list(_save_prompt_overrides(overrides))["prompts"] if p["id"] == pid)


@app.post("/api/prompts/reset-all")
def api_prompts_reset_all():
    return prompts_mod.public_list(_save_prompt_overrides({}, preset=None))


def _preset_rows(settings: dict) -> list[dict]:
    active = settings.get("prompt_preset")
    cur = {k: v for k, v in sorted((settings.get("prompts") or {}).items())}
    rows = []
    for name, p in prompts_mod.load_presets().items():
        rows.append({"name": name, "saved_at": p.get("saved_at"), "note": p.get("note", ""),
                     "count": len(p.get("prompts") or {}), "active": name == active,
                     "dirty": (name == active) and cur != {k: v for k, v in sorted((p.get("prompts") or {}).items())}})
    return rows


@app.get("/api/prompt-presets")
def api_presets():
    return {"presets": _preset_rows(config.load_settings()), "active": config.load_settings().get("prompt_preset")}


class PresetSave(BaseModel):
    name: str
    note: str = ""


@app.post("/api/prompt-presets")
def api_preset_save(req: PresetSave):
    settings = config.load_settings()
    try:
        name = prompts_mod.save_preset(req.name, settings.get("prompts") or {}, req.note)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    config.save_settings({**settings, "prompt_preset": name})
    return {"presets": _preset_rows(config.load_settings()), "active": name}


@app.post("/api/prompt-presets/{name}/load")
def api_preset_load(name: str):
    p = prompts_mod.get_preset(name)
    if p is None:
        raise HTTPException(404, "Такого набора нет")
    rp = prompts_mod.ResolvedPrompts(p.get("prompts") or {}, preset=name)
    cfg = _save_prompt_overrides(dict(rp.overrides), preset=name)
    out = prompts_mod.public_list(cfg)
    out["load_warnings"] = rp.warnings
    return out


class PresetRename(BaseModel):
    new_name: str


@app.put("/api/prompt-presets/{name}")
def api_preset_rename(name: str, req: PresetRename):
    try:
        new = prompts_mod.rename_preset(name, req.new_name)
    except KeyError:
        raise HTTPException(404, "Такого набора нет")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    settings = config.load_settings()
    if settings.get("prompt_preset") == name:
        config.save_settings({**settings, "prompt_preset": new})
    return {"presets": _preset_rows(config.load_settings()), "active": config.load_settings().get("prompt_preset")}


@app.delete("/api/prompt-presets/{name}")
def api_preset_delete(name: str):
    if not prompts_mod.delete_preset(name):
        raise HTTPException(404, "Такого набора нет")
    settings = config.load_settings()
    if settings.get("prompt_preset") == name:
        config.save_settings({**settings, "prompt_preset": None})
    return {"presets": _preset_rows(config.load_settings()), "active": config.load_settings().get("prompt_preset")}


@app.get("/api/prompt-presets-export")
def api_presets_export(name: Optional[str] = None):
    data = prompts_mod.export_json([name] if name else None)
    if name and not data["presets"]:
        raise HTTPException(404, "Такого набора нет")
    fname = "prompt-presets.json" if not name else "prompt-preset.json"
    return Response(json.dumps(data, ensure_ascii=False, indent=1), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


class PresetImport(BaseModel):
    data: dict


@app.post("/api/prompt-presets-import")
def api_presets_import(req: PresetImport):
    try:
        res = prompts_mod.import_json(req.data)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {**res, "presets": _preset_rows(config.load_settings())}


# -------------------------------------------------- history: rating, compare ---

class RatingRequest(BaseModel):
    rating: Optional[int] = None
    note: Optional[str] = None


@app.post("/api/history/{entry_id}/rating")
def api_history_rating(entry_id: int, req: RatingRequest):
    try:
        r = hist.set_rating(entry_id, req.rating, req.note)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if r is None:
        raise HTTPException(404, "Запись истории не найдена")
    return {"ok": True, **r}


# ------------------------------------------------------------- library -----

_BOOK_ROW_CACHE: dict[str, tuple[float, dict]] = {}


def _book_row(book_id: str) -> dict:
    """book_id + what the `books` table says about it (name, language, size); cached by file time."""
    path = state.book_db_path(book_id)
    row = {"book_id": book_id, "book_name": book_id, "language": None, "bilingual": False, "paragraphs": None,
           "embedding_model": None}
    try:
        mt = os.path.getmtime(path)
        cached = _BOOK_ROW_CACHE.get(book_id)
        if cached and cached[0] == mt:
            return dict(cached[1])
        conn = core_db.open_db(path, readonly=True)
        try:
            info = core_db.get_book_info(conn, book_id) or core_db.get_book_info(conn) or {}
            n = conn.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0]
        finally:
            conn.close()
        mono = not (info.get("language_translation") or "")
        row.update({"book_name": info.get("book_name") or book_id, "language": info.get("language_source") or None,
                    "bilingual": not mono, "paragraphs": n, "embedding_model": info.get("embedding_model"),
                    "embedding_runtime": info.get("embedding_runtime"), "index_scheme": info.get("index_scheme"),
                    "vector_dtype": info.get("vector_dtype")})
        _BOOK_ROW_CACHE[book_id] = (mt, dict(row))
    except Exception as exc:  # noqa: BLE001
        row["error"] = str(exc)
    return row


@app.get("/api/books")
def api_books():
    ids = _library_ids()
    selected = _resolve_books(None) if ids else []
    colors = config.load_settings().get("book_colors") or {}
    books = [{**_book_row(b), "active": b == state.book_id, "selected": b in selected, "color": colors.get(b)}
             for b in ids]
    return {"books": books, "search_books": selected, "active_book": state.book_id}


class BookSelectRequest(BaseModel):
    book_id: str


@app.post("/api/books/select")
def api_books_select(req: BookSelectRequest):
    state.load_book(req.book_id)
    settings = config.load_settings()
    settings["active_book"] = req.book_id
    config.save_settings(settings)
    return {"ok": True, "db_error": state.db_error}


def _book_or_404(book_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", book_id) or book_id not in _library_ids():
        raise HTTPException(404, "Книга не найдена")


@app.get("/api/books/{book_id}/info")
def api_book_info(book_id: str):
    """Everything known about one book for the «О книге» window: name, source file, author, language,
    size, structure, how and when it was indexed, its colour."""
    _book_or_404(book_id)
    path = state.book_db_path(book_id)
    conn = core_db.open_db(path, readonly=True)
    try:
        info = core_db.get_book_info(conn, book_id) or core_db.get_book_info(conn) or {}
        n_par = conn.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0]
        n_ch = conn.execute("SELECT COUNT(DISTINCT paper) FROM paragraphs").fetchone()[0]
    finally:
        conn.close()
    src = info.get("source_file") or ""
    src_abs = src if (not src or os.path.isabs(src)) else os.path.join(str(config.ROOT), src)
    src_exists = bool(src_abs) and os.path.isfile(src_abs)
    meta = {}
    if src_exists:
        try:
            meta = core_parser.read_book_meta(src_abs) or {}
        except Exception:  # noqa: BLE001  (old json/md sources have no metadata)
            meta = {}
    def _short(p):  # relative to the program folder when inside it, otherwise as is
        try:
            r = os.path.relpath(p, str(config.ROOT))
            return p if r.startswith("..") else r
        except ValueError:  # another drive
            return p
    rel = _short(src_abs) if src_abs else ""
    colors = config.load_settings().get("book_colors") or {}
    return {
        "book_id": book_id, "book_name": info.get("book_name") or book_id,
        "author": meta.get("author") or "", "title_in_file": meta.get("title") or "",
        "source_file": rel, "source_exists": src_exists,
        "source_size_bytes": os.path.getsize(src_abs) if src_exists else None,
        "language": info.get("language_source"), "translation": info.get("language_translation") or "",
        "paragraphs": n_par, "chapters": n_ch,
        "db_path": _short(path), "db_size_bytes": os.path.getsize(path),
        "created_at": info.get("created_at"), "embedding_model": info.get("embedding_model"),
        "embedding_runtime": info.get("embedding_runtime"), "index_scheme": info.get("index_scheme"),
        "vector_dtype": info.get("vector_dtype"), "color": colors.get(book_id),
        "active": book_id == state.book_id,
    }


class BookEditRequest(BaseModel):
    book_name: Optional[str] = None
    color: Optional[str] = None   # "#rrggbb"; "" = back to the automatic colour


@app.patch("/api/books/{book_id}")
def api_book_edit(book_id: str, req: BookEditRequest):
    """Rename a book (stored in its book.db) and/or set its colour (stored in settings)."""
    _book_or_404(book_id)
    if req.book_name is not None:
        name = req.book_name.strip()
        if not name:
            raise HTTPException(400, "Название не может быть пустым")
        import sqlite3
        c = sqlite3.connect(state.book_db_path(book_id), timeout=10)
        try:
            c.execute("UPDATE books SET book_name = ?", (name[:300],))
            c.commit()
        finally:
            c.close()
        _BOOK_ROW_CACHE.pop(book_id, None)
        if book_id != state.book_id:
            state.drop_handle(book_id)   # the cached handle carries the old name
    if req.color is not None:
        col = req.color.strip()
        if col and not re.fullmatch(r"#[0-9a-fA-F]{6}", col):
            raise HTTPException(400, "Цвет должен быть в виде #rrggbb")
        settings = config.load_settings()
        colors = dict(settings.get("book_colors") or {})
        if col:
            colors[book_id] = col.lower()
        else:
            colors.pop(book_id, None)
        settings["book_colors"] = colors
        config.save_settings(settings)
    return api_book_info(book_id)


class SearchBooksRequest(BaseModel):
    book_ids: list[str]


@app.post("/api/books/search-selection")
def api_books_search_selection(req: SearchBooksRequest):
    """Remember which books the search covers («все» = every library book)."""
    lib = _library_ids()
    ids = [b for b in dict.fromkeys(req.book_ids) if b in lib]
    if not ids:
        raise HTTPException(400, "Выберите хотя бы одну книгу")
    settings = config.load_settings()
    settings["search_books"] = ids
    config.save_settings(settings)
    return {"ok": True, "search_books": ids}


def _is_book_file(path: str) -> bool:
    return str(path).lower().endswith(tuple(core_parser.BOOK_EXTENSIONS))


_TRANSLIT = dict(zip("абвгдежзийклмнопрстуфхцчшщыэюя",
                     ["a", "b", "v", "g", "d", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s", "t",
                      "u", "f", "h", "ts", "ch", "sh", "sch", "y", "e", "yu", "ya"]))


def suggest_book_id(*texts: Optional[str]) -> str:
    """A latin slug for the library folder from the first non-empty text (title, file name...), unique among the books."""
    base = ""
    for t in texts:
        t = (t or "").lower().replace("ё", "е")
        base = re.sub(r"[^a-z0-9]+", "_", "".join(_TRANSLIT.get(c, "" if c in "ьъ" else c) for c in t)).strip("_")[:40].strip("_")
        if base:
            break
    base = base or "book"
    taken = set(_library_ids())
    cand, n = base, 2
    while cand in taken:
        cand, n = f"{base}_{n}", n + 1
    return cand


def _safe_filename(name: str) -> str:
    base = os.path.basename((name or "").replace("\\", "/")).strip()
    return re.sub(r"[^\w.\- ()]", "_", base, flags=re.UNICODE).strip(". ")


MAX_UPLOAD_BYTES = 300 * 1024 * 1024


_FILE_HASH_CACHE: dict = {}   # path -> (mtime, size, sha256)


def _file_sha256(path: str) -> Optional[str]:
    import hashlib
    try:
        st = os.stat(path)
    except OSError:
        return None
    c = _FILE_HASH_CACHE.get(path)
    if c and c[0] == st.st_mtime and c[1] == st.st_size:
        return c[2]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    _FILE_HASH_CACHE[path] = (st.st_mtime, st.st_size, h.hexdigest())
    return h.hexdigest()


def _norm_title(s: Optional[str]) -> str:
    return re.sub(r"[\W_]+", " ", (s or "").lower().replace("ё", "е")).strip()


def _find_duplicate_book(body: bytes, filename: str) -> Optional[dict]:
    """Is this book already in the library? Same file content as a book's source, or the same title.
    Returns {book_id, book_name, reason} or None."""
    import hashlib
    import tempfile
    digest = hashlib.sha256(body).hexdigest()
    title = ""
    try:  # the title the new file declares (parsed from a temp copy; metadata only)
        suffix = ".fb2.zip" if filename.lower().endswith(".fb2.zip") else os.path.splitext(filename)[1]
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(body)
        try:
            title = (core_parser.read_book_meta(tmp) or {}).get("title") or ""
        finally:
            os.remove(tmp)
    except Exception:  # noqa: BLE001  (unreadable file: the normal upload path reports it)
        pass
    want = _norm_title(title)
    for b in _library_ids():
        path = state.book_db_path(b)
        try:
            conn = core_db.open_db(path, readonly=True)
            try:
                info = core_db.get_book_info(conn, b) or core_db.get_book_info(conn) or {}
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            continue
        name = info.get("book_name") or b
        src = info.get("source_file") or ""
        src_abs = src if (not src or os.path.isabs(src)) else os.path.join(str(config.ROOT), src)
        if src_abs and os.path.isfile(src_abs) and _file_sha256(src_abs) == digest:
            return {"book_id": b, "book_name": name, "reason": "тот же файл"}
        if want and want == _norm_title(name):
            return {"book_id": b, "book_name": name, "reason": "то же название"}
        if want and src_abs and os.path.isfile(src_abs):
            try:
                if want == _norm_title((core_parser.read_book_meta(src_abs) or {}).get("title")):
                    return {"book_id": b, "book_name": name, "reason": "то же название"}
            except Exception:  # noqa: BLE001
                pass
    return None


@app.post("/api/books/upload")
async def api_books_upload(request: Request, filename: str = Query(...), force: bool = False):
    """Receive a book file (raw request body; the name in `?filename=`), keep it in corpus/ and report what the file
    says about itself, with a suggested book id. Nothing is indexed here."""
    name = _safe_filename(filename)
    if not name or not _is_book_file(name):
        raise HTTPException(400, "Формат файла не поддерживается. Нужны: " + ", ".join(core_parser.BOOK_EXTENSIONS))
    body = await request.body()
    if not body:
        raise HTTPException(400, "Файл пустой")
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Файл слишком большой (больше 300 МБ)")
    dup = None if force else _find_duplicate_book(body, name)
    if dup:
        # already in the library: nothing is saved, the caller shows a note and moves on
        return {"duplicate_of": dup, "file": None, "title": dup["book_name"], "size_bytes": len(body)}
    os.makedirs(config.CORPUS_DIR, exist_ok=True)
    if name.lower().endswith(".fb2.zip"):
        stem, ext = name[:-len(".fb2.zip")], name[-len(".fb2.zip"):]
    else:
        stem, ext = os.path.splitext(name)
    # the very same file is already in corpus/ (e.g. dropped twice): reuse it instead of storing a copy
    import hashlib
    digest = hashlib.sha256(body).hexdigest()
    same = next((p for p in glob.glob(os.path.join(str(config.CORPUS_DIR), "*"))
                 if os.path.isfile(p) and os.path.getsize(p) == len(body) and _file_sha256(p) == digest), None)
    target, n = os.path.join(str(config.CORPUS_DIR), name), 2
    while same is None and os.path.exists(target):
        target, n = os.path.join(str(config.CORPUS_DIR), f"{stem}_{n}{ext}"), n + 1
    if same is not None:
        target = same
    else:
        with open(target, "wb") as f:
            f.write(body)
    try:
        meta = core_parser.read_book_meta(target)
    except Exception as exc:  # noqa: BLE001
        if same is None:
            os.remove(target)
        raise HTTPException(400, f"Не удалось прочитать файл как книгу: {exc}")
    stem_name = os.path.splitext(os.path.basename(target))[0]
    return {"file": "corpus/" + os.path.basename(target), "title": meta.get("title") or stem_name,
            "author": meta.get("author"), "language": meta.get("language"),
            "book_id": suggest_book_id(meta.get("title"), stem_name), "size_bytes": len(body)}


@app.get("/api/books/preview")
def api_books_preview(file: str):
    """What a book source contains (chapters / paragraphs / language) - parses the file, no database is touched."""
    if os.path.isabs(file):
        path = file
    elif file.replace("\\", "/").startswith("corpus/"):
        path = os.path.join(str(config.CORPUS_DIR), file.replace("\\", "/")[len("corpus/"):])
    else:
        path = os.path.join(str(config.ROOT), file)
    path = os.path.realpath(path)
    allowed = (os.path.realpath(str(config.CORPUS_DIR)), os.path.realpath(str(config.ROOT)))
    if not any(path == a or path.startswith(a + os.sep) for a in allowed) or not os.path.isfile(path) \
            or not _is_book_file(path):
        raise HTTPException(400, "Файл не найден среди исходных книг")
    try:
        data = core_parser.load_book(path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Не удалось разобрать книгу: {exc}")
    recs = data.records
    texts = [(r.get(data.lang or "ru") or r.get("ru") or r.get("en") or "") for r in recs[len(recs) // 3:] + recs]
    sample = next((t for t in texts if len(t) >= 40), "")
    return {"file": file, "paragraphs": len(recs), "chapters": len({r["paper"] for r in recs}),
            "language": data.lang or "bilingual", "title": data.meta.get("title"), "author": data.meta.get("author"),
            "sample": (sample or "")[:300], "trial": _looks_like_trial(recs)}


def _looks_like_trial(recs) -> bool:
    """A shop's free excerpt (LitRes etc.) rather than the whole book: it ends with «Конец ознакомительного
    фрагмента» / a «buy the full version» note."""
    tail = " ".join((r.get("ru") or r.get("en") or "") for r in recs[-6:]).lower()
    return any(s in tail for s in ("ознакомительного фрагмента", "полную версию", "литрес", "end of sample",
                                   "end of this sample"))


@app.delete("/api/books/{book_id}")
def api_books_delete(book_id: str):
    """Remove a book from the library (its book.db folder). The source file in corpus/ stays."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", book_id) or book_id not in _library_ids():
        raise HTTPException(404, "Книга не найдена")
    if book_id == state.book_id or book_id == config.load_settings().get("active_book"):
        raise HTTPException(409, "Это активная книга: сначала сделайте активной другую.")
    if any(j.type == "index" and j.status in ("queued", "running") for j in job_manager.list()):
        raise HTTPException(409, "Сейчас идёт индексация — дождитесь завершения.")
    state.drop_handle(book_id)
    import shutil
    shutil.rmtree(os.path.join(str(config.LIBRARY_DIR), book_id), ignore_errors=True)
    _BOOK_ROW_CACHE.pop(book_id, None)
    settings = config.load_settings()
    if book_id in (settings.get("search_books") or []):
        settings["search_books"] = [b for b in settings["search_books"] if b != book_id]
        config.save_settings(settings)
    return {"ok": True}


@app.get("/api/source-files")
def api_source_files():
    """List *.json/*.md/*.epub/*.fb2/*.fb2.zip source files for the index-source picker (the browser can't hand us an
    arbitrary local path): the «corpus» folder first, then the program folder (older layout).
    Names are relative to the program folder, e.g. «corpus/book.json»."""
    names = []
    for folder, prefix in ((config.CORPUS_DIR, "corpus/"), (config.ROOT, "")):
        for pat in ("*.json", "*.md", "*.epub", "*.fb2", "*.fb2.zip"):
            names += sorted(prefix + os.path.basename(f) for f in glob.glob(os.path.join(folder, pat)))
    return {"files": [n for n in names if n.lower() not in ("readme.md",)]}


# ------------------------------------------------------------- agents ------

from app import agent_api  # noqa: E402

app.include_router(agent_api.router)


@app.middleware("http")
async def only_agents_from_outside(request, call_next):
    """When the program listens on the network (--host), other computers reach only the agent API
    (which has its own key check); the UI, settings and keys stay local."""
    host = request.client.host if request.client else ""
    path = request.url.path
    if host not in agent_api._LOOPBACK and not (path == "/mcp" or path.startswith("/api/agent/")):
        return Response("Доступно только с этого компьютера", status_code=403)
    return await call_next(request)


# ------------------------------------------------------------- static ------

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


_real_get_llm_client = get_llm_client  # tests swap get_llm_client for fakes and may need the real one back
