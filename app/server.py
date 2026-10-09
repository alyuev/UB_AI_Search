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

from fastapi import FastAPI, HTTPException, Query, Response
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

    def reload_searcher(self):
        if self.book_id:
            self.load_book(self.book_id)

    def shutdown(self):
        """Close everything on app shutdown -- in particular, terminate a
        LlamaCppEmbedder's llama-server subprocess rather than leaving it running."""
        self._close_conn()
        if self.embedder is not None and hasattr(self.embedder, "close"):
            try:
                self.embedder.close()
            except Exception:  # noqa: BLE001
                pass
        self.embedder = None
        self.embedder_key = None
        self.searcher = None


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


@app.get("/api/search")
def api_search(q: str = Query(...), mode: str = Query("hybrid"), k: int = Query(10),
               lang: Optional[str] = Query(None), history: int = Query(1)):
    if state.conn is None:
        raise HTTPException(503, state.db_error or "База не загружена")
    if state.searcher is None:
        raise HTTPException(503, "Поисковый индекс недоступен")
    fn = {"hybrid": state.searcher.hybrid, "fts": state.searcher.fts,
          "semantic": state.searcher.semantic}.get(mode)
    if fn is None:
        raise HTTPException(400, f"Неизвестный режим поиска: {mode}")
    t0 = time.monotonic()
    try:
        results = fn(q, k)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Ошибка поиска: {exc}")
    timings = getattr(state.searcher, "last_timings", {})
    if history:
        n = len(results)
        _record_history("search", q, {"mode": mode, "k": k, "lang": lang},
                        {"results": _compact(results), "timings": timings},
                        f"{n} {_plural(n, 'результат', 'результата', 'результатов')}" if n else "ничего не найдено",
                        time.monotonic() - t0, experiment=_search_experiment(q, mode, k, lang, timings))
    return {"results": results, "timings": timings}


@app.get("/api/paragraph/{para_id}")
def api_paragraph(para_id: str):
    if state.conn is None:
        raise HTTPException(503, state.db_error or "База не загружена")
    p = core_db.get_paragraph(state.conn, para_id)
    if not p:
        raise HTTPException(404, "Абзац не найден")
    return p


@app.get("/api/paragraph/{para_id}/context")
def api_paragraph_context(para_id: str, before: int = 2, after: int = 2):
    if state.conn is None:
        raise HTTPException(503, state.db_error or "База не загружена")
    ctx = core_db.get_context(state.conn, para_id, before, after)
    if not ctx:
        raise HTTPException(404, "Абзац не найден")
    return {"context": ctx}


@app.get("/api/toc")
def api_toc():
    """Table of contents of the active book (parts > papers > sections, titles in both languages)."""
    if state.conn is None:
        raise HTTPException(503, state.db_error or "База не загружена")
    return {"parts": core_db.get_toc(state.conn)}


@app.get("/api/read/{paper}")
def api_read(paper: int, q: str = ""):
    """The whole text of one paper for the reading page. With `q`, also the words to highlight
    for that search query in both languages (`marks`); without it nothing is highlighted."""
    if state.conn is None:
        raise HTTPException(503, state.db_error or "База не загружена")
    data = core_db.get_paper_text(state.conn, paper)
    if not data:
        raise HTTPException(404, "Документ не найден")
    data["marks"] = {"ru": [], "en": []}
    if q.strip() and state.searcher is not None:
        paras = data["paragraphs"]
        data["marks"] = state.searcher.terms.query_marks(q, [p["ru"] for p in paras], [p["en"] for p in paras])
    return data


@app.get("/api/paragraph/{para_id}/pairs")
def api_paragraph_pairs(para_id: str):
    """RU<->EN glossary pairs present in this paragraph, for the hover highlight in the result cards.
    Empty list (not an error) when the glossary has not been built."""
    if state.conn is None:
        raise HTTPException(503, state.db_error or "База не загружена")
    p = core_db.get_paragraph(state.conn, para_id)
    if not p:
        raise HTTPException(404, "Абзац не найден")
    sr = state.searcher
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


class RagJobRequest(BaseModel):
    question: str
    k: Optional[int] = None          # None -> the «сколько абзацев передавать» setting
    replay: Optional[int] = None     # history entry id: re-run with EXACTLY the saved prompts/parameters


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
    if state.searcher is None:
        raise HTTPException(503, "Поисковый индекс недоступен")
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
        # Resolve the model actually usable at the provider (for LM Studio: the LOADED one) and its context
        # length, used for context fitting; raises a clear error -> job error.
        resolved = client.resolve_model(model_pref)
        if resolved["note"]:
            job.add_log(resolved["note"])
        rw_client, rw_model, rw_info = None, None, None
        if rw_plan and (rw_plan[0]["id"], rw_plan[1]) != (prof["id"], model_pref):
            rw_client = _client_for(rw_plan[0])
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
        searcher = state.searcher
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
            options=options, prompts=prompts, conn=state.conn,
            meta=_search_meta(settings, "reasoning_rag"),
        )
        t_start = time.monotonic()
        # replay: the saved k; otherwise the request's k, else the «сколько абзацев» setting
        k_eff = options["rag_top_k"] if req.replay is not None else (req.k or options["rag_top_k"])
        result = pipeline.answer(req.question, k=k_eff, progress=progress)
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
            "rag", req.question, {"k": k_eff, "replay": req.replay, **_cloud_params(result)},
            {**{k: result.get(k) for k in ("answer", "cited_ids", "unknown_citations", "no_answer", "has_inference",
                                          "plan", "candidates", "stats_summary", "model_note", "prompt_warnings",
                                          "warnings", "provider", "truncated", "error", "answered_model",
                                          "fallback_note")},
             "sources": _compact(result.get("sources") or [])},
            "ответ оборван на середине" if partial else
            "в книге нет данных для ответа" if result.get("no_answer")
            else f"ответ, {n_cited} {_plural(n_cited, 'источник', 'источника', 'источников')}",
            time.monotonic() - t_start, experiment=result.get("experiment"))
        if partial:   # the job is an error (the reason is shown), the partial answer stays in job.result
            raise RuntimeError(result["error"])

    job = job_manager.create("rag", ["analysis", "fts", "semantic", "hybrid", "context", "generation"], worker)
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
    return {**prof, "host": providers.host_of(prof["base_url"]), "is_cloud": cloud, "active": is_active,
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


@app.get("/api/llm/models")
def api_llm_models(provider: Optional[str] = None, q: str = "", free_only: bool = False, min_context: int = 0,
                   refresh: bool = False):
    client = _client_or_error(provider)
    try:
        try:
            models = client.list_models(refresh=refresh)
        except TypeError:  # clients without caching
            models = client.list_models()
    except LMStudioError as exc:
        raise HTTPException(503, str(exc))
    total = len(models)
    return {"models": providers.filter_models(models, q, free_only, min_context), "total": total,
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
                    experiment: Optional[dict] = None):
    """Saving history must never break a search/job: any problem is only logged."""
    try:
        hist.add(_history_book(), mode, query, params, result, summary, duration_s, experiment=experiment)
    except Exception:  # noqa: BLE001
        logger.warning("could not save history entry", exc_info=True)


_PARA_KEEP = ("id", "score", "fts_rank", "sem_rank", "match_lang", "snippet_en", "snippet_ru")


def _compact(items: list[dict]) -> list[dict]:
    """What is needed to redraw a result list; paragraph text is re-fetched by id on open."""
    return [{k: it.get(k) for k in _PARA_KEEP if it.get(k) is not None} for it in items]


def _hydrate(items: list[dict]) -> tuple[list[dict], int]:
    out, missing = [], 0
    for it in items:
        p = core_db.get_paragraph(state.conn, it["id"]) if state.conn is not None else None
        if p is None:
            missing += 1
            p = {"id": it["id"], "paper": None, "section": None, "paragraph": None, "en": "", "ru": ""}
        out.append({**p, **it})
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
    e["book_loaded"] = e["book_id"] == state.book_id and state.conn is not None
    if e["mode"] == "search":
        items, missing = _hydrate(res.get("results", []))
        e["result"] = {"results": items, "timings": res.get("timings", {})}
    elif e["mode"] == "rag":
        items, missing = _hydrate(res.get("sources", []))
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

@app.get("/api/books")
def api_books():
    books = []
    if os.path.isdir(config.LIBRARY_DIR):
        for name in sorted(os.listdir(config.LIBRARY_DIR)):
            db_path = os.path.join(config.LIBRARY_DIR, name, "book.db")
            if os.path.exists(db_path):
                books.append({"book_id": name, "active": name == state.book_id})
    return {"books": books}


class BookSelectRequest(BaseModel):
    book_id: str


@app.post("/api/books/select")
def api_books_select(req: BookSelectRequest):
    state.load_book(req.book_id)
    settings = config.load_settings()
    settings["active_book"] = req.book_id
    config.save_settings(settings)
    return {"ok": True, "db_error": state.db_error}


@app.get("/api/source-files")
def api_source_files():
    """List *.json/*.md files in the project root, for the index-source picker
    (the browser can't hand us an arbitrary local path)."""
    patterns = ["*.json", "*.md"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(config.ROOT, pat)))
    names = sorted(os.path.basename(f) for f in files)
    return {"files": [n for n in names if n.lower() not in ("readme.md",)]}


# ------------------------------------------------------------- static ------

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


_real_get_llm_client = get_llm_client  # tests swap get_llm_client for fakes and may need the real one back
