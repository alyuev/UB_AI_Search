
import os
import sys

import pytest


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from app import config
    from tests.helpers import create_fake_book_db

    # IMPORTANT: use isolated, temporary paths for BOTH the library dir and the
    # settings file, so tests never touch the real library/<book_id>/book.db or
    # the real data/settings.json shared with a live running app -- a prior
    # version of this fixture only isolated LIBRARY_DIR and ended up overwriting
    # the real app's active_book/settings on disk when tests ran concurrently
    # with `main.py`.
    tmp_library = tmp_path_factory.mktemp("library")
    config.LIBRARY_DIR = tmp_library
    tmp_data_dir = tmp_path_factory.mktemp("data")
    config.DATA_DIR = tmp_data_dir
    config.SETTINGS_PATH = tmp_data_dir / "settings.json"
    book_dir = os.path.join(str(tmp_library), "test_book")
    os.makedirs(book_dir, exist_ok=True)
    create_fake_book_db(os.path.join(book_dir, "book.db"), book_id="test_book")
    settings = config.load_settings()
    settings["active_book"] = "test_book"
    settings["llm_model"] = None
    config.save_settings(settings)

    # ensure a clean import of app.server (module-level state depends on config)
    for mod in list(sys.modules):
        if mod == "app.server":
            del sys.modules[mod]
    from app.server import app, state
    from fastapi.testclient import TestClient

    state.load_book("test_book")
    return TestClient(app)


def test_status_endpoint(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert data["db_loaded"] is True


def test_search_fts_finds_english(client):
    r = client.get("/api/search", params={"q": "Jesus", "mode": "fts", "k": 5})
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) >= 1
    assert any(p["id"] == "147.2.2" for p in data["results"])


def test_search_fts_finds_russian(client):
    r = client.get("/api/search", params={"q": "Иерусалиме", "mode": "fts", "k": 5})
    assert r.status_code == 200
    assert any(p["id"] == "147.2.4" for p in r.json()["results"])


def test_paragraph_get(client):
    r = client.get("/api/paragraph/147.2.1")
    assert r.status_code == 200
    assert r.json()["paper"] == 147


def test_paragraph_not_found(client):
    r = client.get("/api/paragraph/does-not-exist")
    assert r.status_code == 404


def test_paragraph_context(client):
    r = client.get("/api/paragraph/147.2.2/context", params={"before": 1, "after": 1})
    assert r.status_code == 200
    ctx = r.json()["context"]
    assert len(ctx) == 3


def test_papers_and_sections(client):
    r = client.get("/api/papers")
    papers = [p["paper"] for p in r.json()["papers"]]
    assert 147 in papers
    r2 = client.get("/api/papers/147/sections")
    sections = [s["section"] for s in r2.json()["sections"]]
    assert 2 in sections


def test_settings_roundtrip(client):
    r = client.get("/api/settings")
    assert r.status_code == 200
    r2 = client.post("/api/settings", json={"temperature": 0.5})
    assert r2.status_code == 200
    assert r2.json()["temperature"] == 0.5


def test_source_files_listing(client):
    r = client.get("/api/source-files")
    assert r.status_code == 200
    assert "files" in r.json()


def test_books_listing(client):
    r = client.get("/api/books")
    assert r.status_code == 200
    ids = [b["book_id"] for b in r.json()["books"]]
    assert "test_book" in ids


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "SmartSearch" in r.text


def test_translation_job_bad_reference(client):
    r = client.post("/api/jobs/translation-analysis", json={"reference": "gibberish text"})
    assert r.status_code == 400


def test_settings_rejects_bad_fts_values(client):
    assert client.post("/api/settings", json={"fts_mode_hybrid": "xor"}).status_code == 400
    assert client.post("/api/settings", json={"fts_weights": {"nope": 1}}).status_code == 400
    assert client.post("/api/settings", json={"fts_weights": {"en": "abc"}}).status_code == 400
    assert client.post("/api/settings", json={"fts_weights": {"en": -1}}).status_code == 400


def test_settings_fts_change_recreates_searcher(client):
    from app.server import state
    before = state.searcher
    assert before is not None
    r = client.post("/api/settings", json={"fts_weights": {"title_en": 3.0}, "fts_mode_standalone": "and"})
    assert r.status_code == 200
    body = r.json()
    assert body["fts_weights"]["title_en"] == 3.0 and body["fts_weights"]["en"] == 1.0  # merged, not replaced
    assert body["fts_mode_standalone"] == "and"
    assert state.searcher is not None and state.searcher is not before
    assert state.searcher.fts_mode_standalone == "and"
    client.post("/api/settings", json={"fts_mode_standalone": "or"})


def _wait_job(client, job_id, timeout=60):
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_rebuild_fts_job_runs_and_search_still_works(client):
    from app.server import state
    before = state.searcher
    r = client.post("/api/jobs/rebuild-fts")
    assert r.status_code == 200
    job = _wait_job(client, r.json()["job_id"])
    assert job["status"] == "done", job
    assert job["result"] == {"cancelled": False, "rows": 5}
    assert [s["name"] for s in job["stages"]] == ["fts"]
    assert state.searcher is not None and state.searcher is not before  # recreated
    res = client.get("/api/search", params={"q": "Jesus", "mode": "fts", "k": 5}).json()["results"]
    assert any(p["id"] == "147.2.2" for p in res)
    # FTS results carry highlight snippets (HTML-escaped, only <mark> tags) or None
    hit = next(p for p in res if p["id"] == "147.2.2")
    assert "snippet_en" in hit and "snippet_ru" in hit
    if hit["snippet_en"]:
        assert "<mark>" in hit["snippet_en"]


def test_index_job_cancel_keeps_old_db_loaded(client, monkeypatch):
    """build_index raises RuntimeError('cancelled') on cancel; the app must reopen the
    untouched old book.db instead of staying without a database."""
    from app import server
    from app.server import state

    def fake_build(*a, cancel=None, **kw):
        cancel.set()
        raise RuntimeError("cancelled")

    monkeypatch.setattr(server.core_indexer, "build_index", fake_build)
    r = client.post("/api/jobs/index", json={
        "json_path": __file__, "model_name": "bge-m3", "book_id": "test_book", "book_name": "T"})
    job = _wait_job(client, r.json()["job_id"])
    assert job["status"] == "cancelled", job
    assert job["result"]["cancelled"] is True
    assert state.conn is not None and state.searcher is not None
    assert client.get("/api/search", params={"q": "Jesus", "mode": "fts"}).status_code == 200


def test_index_job_failure_is_reported_and_db_stays_loaded(client, monkeypatch):
    from app import server
    from app.server import state

    def boom(*a, **kw):
        raise ValueError("source file is garbage")

    monkeypatch.setattr(server.core_indexer, "build_index", boom)
    r = client.post("/api/jobs/index", json={
        "json_path": __file__, "model_name": "bge-m3", "book_id": "test_book", "book_name": "T"})
    job = _wait_job(client, r.json()["job_id"])
    assert job["status"] == "error" and "garbage" in job["error"]
    assert state.conn is not None and state.searcher is not None


def test_index_job_validates_source_and_book_id(client):
    r = client.post("/api/jobs/index", json={
        "json_path": "does_not_exist.json", "model_name": "bge-m3", "book_id": "ok_id", "book_name": "T"})
    assert r.status_code == 400 and "не найден" in r.json()["detail"]
    r = client.post("/api/jobs/index", json={
        "json_path": __file__, "model_name": "bge-m3", "book_id": "bad id/..", "book_name": "T"})
    assert r.status_code == 400


def test_source_files_hides_readme(client):
    assert "README.md" not in client.get("/api/source-files").json()["files"]


def test_endpoint_for_active_book(client):
    r = client.get("/api/db/status")
    assert r.status_code == 200
    body = r.json()
    assert body["exists"] is True and body["paragraphs"] == 5
    assert body["state"] in ("ready", "attention")
    assert {"rows", "warnings", "fts", "vectors", "search_settings"} <= set(body)


def test_ui_files_are_not_cached(client):
    assert client.get("/").headers["cache-control"] == "no-cache"
    assert "cache-control" not in client.get("/api/status").headers or \
        client.get("/api/status").headers["cache-control"] != "no-cache"


# ------------------------------------------------------------------ history ---
def _clear_history(client):
    client.delete("/api/history")


def test_search_is_saved_and_reopened_with_text(client):
    _clear_history(client)
    r = client.get("/api/search", params={"q": "Jesus", "mode": "fts", "k": 5})
    assert r.status_code == 200
    lst = client.get("/api/history").json()
    assert lst["total"] == 1
    e = lst["items"][0]
    assert e["mode"] == "search" and e["query"] == "Jesus" and e["params"]["mode"] == "fts"
    assert e["summary"].split()[0].isdigit() and "результат" in e["summary"]
    full = client.get(f"/api/history/{e['id']}").json()
    items = full["result"]["results"]
    assert any(p["id"] == "147.2.2" for p in items)
    hit = next(p for p in items if p["id"] == "147.2.2")
    assert hit["en"] and hit["ru"]                      # paragraph text re-fetched by id
    assert full["missing_paragraphs"] == 0 and full["book_loaded"] is True


def test_search_can_skip_history_and_empty_result_is_labelled(client):
    _clear_history(client)
    client.get("/api/search", params={"q": "Jesus", "mode": "fts", "history": 0})
    assert client.get("/api/history").json()["total"] == 0
    client.get("/api/search", params={"q": "zzzqqq", "mode": "fts"})
    assert client.get("/api/history").json()["items"][0]["summary"] == "ничего не найдено"


def test_history_api_filter_pin_delete_clear(client):
    _clear_history(client)
    for q in ("Jesus", "Father"):
        client.get("/api/search", params={"q": q, "mode": "fts"})
    assert client.get("/api/history", params={"q": "jes"}).json()["total"] == 1
    assert client.get("/api/history", params={"mode": "rag"}).json()["total"] == 0
    ids = {e["query"]: e["id"] for e in client.get("/api/history").json()["items"]}
    assert client.post(f"/api/history/{ids['Jesus']}/pin").json() == {"ok": True, "pinned": True}
    assert client.post(f"/api/history/{ids['Jesus']}/pin", json={"pinned": False}).json()["pinned"] is False
    client.post(f"/api/history/{ids['Jesus']}/pin", json={"pinned": True})
    assert client.delete("/api/history").json() == {"deleted": 1}          # the pinned entry survives
    left = client.get("/api/history").json()
    assert [e["query"] for e in left["items"]] == ["Jesus"] and left["items"][0]["pinned"] is True
    assert client.delete(f"/api/history/{ids['Jesus']}").status_code == 200
    assert client.delete(f"/api/history/{ids['Jesus']}").status_code == 404
    assert client.get("/api/history/999999").status_code == 404
    assert client.post("/api/history/999999/pin").status_code == 404


class _FakeLLM:
    def resolve_model(self, configured):
        return {"model": "m", "context": 16000, "note": None}


def test_rag_and_translation_jobs_are_saved(client, monkeypatch):
    from app import server
    _clear_history(client)
    monkeypatch.setattr(server, "get_llm_client", lambda: _FakeLLM())

    class FakeRag:
        def __init__(self, *a, **kw):
            pass

        def answer(self, question, k=12, progress=None):
            return {"answer": "Иисус говорил [147.2.2].", "error": None, "cited_ids": ["147.2.2"],
                    "unknown_citations": [], "no_answer": False, "plan": {"summary": "Ищу: Иисус"},
                    "candidates": {"fts": 1, "semantic": 1}, "stats_summary": "m · 10 ток/с",
                    "sources": [{"id": "147.2.2", "score": 0.5, "en": "x", "ru": "y"},
                                {"id": "147.2.3", "score": 0.2, "en": "x", "ru": "y"}]}

    class FakeAnalyzer:
        def __init__(self, *a, **kw):
            pass

        def analyze(self, paper, section=None, para_from=None, para_to=None, progress=None):
            return {"issues": [{"id": "147.2.1", "severity": "major", "category": "omissions", "problem": "p",
                                "why_significant": "", "suggestion": "", "en": "E", "ru": "R"}],
                    "summary": "Проверено абзацев: 4. Найдено проблем: 1.", "paragraphs_analyzed": 4,
                    "stats_summary": "m · 5 ток/с"}

    monkeypatch.setattr(server, "RagPipeline", FakeRag)
    monkeypatch.setattr(server, "TranslationAnalyzer", FakeAnalyzer)
    j1 = client.post("/api/jobs/rag", json={"question": "Что говорил Иисус?", "k": 7}).json()["job_id"]
    j2 = client.post("/api/jobs/translation-analysis", json={"reference": "147:2"}).json()["job_id"]
    assert _wait_job(client, j1)["status"] == "done" and _wait_job(client, j2)["status"] == "done"

    items = {e["mode"]: e for e in client.get("/api/history").json()["items"]}
    assert items["rag"]["summary"] == "ответ, 1 источник" and items["rag"]["params"] == {"k": 7, "replay": None}
    assert items["translation"]["summary"] == "1 замечание" and items["translation"]["query"] == "147:2"
    rag = client.get(f"/api/history/{items['rag']['id']}").json()["result"]
    assert rag["answer"].startswith("Иисус") and rag["cited_ids"] == ["147.2.2"]
    assert [s["id"] for s in rag["sources"]] == ["147.2.2", "147.2.3"]      # original order kept
    assert rag["sources"][0]["en"].startswith("Jesus spoke")               # hydrated from the book
    tr = client.get(f"/api/history/{items['translation']['id']}").json()["result"]
    assert tr["issues"][0]["problem"] == "p"
    md = client.get(f"/api/history/{items['translation']['id']}/report.md")
    assert md.status_code == 200 and "147.2.1" in md.text
    assert client.get(f"/api/history/{items['rag']['id']}/report.md").status_code == 404


def test_history_failure_never_breaks_search(client, monkeypatch):
    from app import server

    def boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(server.hist, "add", boom)
    assert client.get("/api/search", params={"q": "Jesus", "mode": "fts"}).status_code == 200


def test_settings_validates_rag_strictness(client):
    assert client.post("/api/settings", json={"rag_strictness": "yolo"}).status_code == 400
    r = client.post("/api/settings", json={"rag_strictness": "facts_only"})
    assert r.status_code == 200 and r.json()["rag_strictness"] == "facts_only"
    client.post("/api/settings", json={"rag_strictness": "allow_inferences"})
