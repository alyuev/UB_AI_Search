"""Several books: the library list, adding a book (upload/preview), multi-book search and its merge."""
from __future__ import annotations

import numpy as np
import pytest

from app import config
from app.core import multi
from app.core.db import open_db
from tests.server_helpers import run_rag, wait_job
from tests.test_book_parsers import make_epub, RU1, RU2


# ------------------------------------------------------------------ merge (pure) --

def _hit(pid, score):
    return {"id": pid, "score": score}


def test_merge_tags_books_and_normalises_fts_per_book():
    per_book = [
        {"book_id": "a", "book_name": "Книга A", "model": "m", "results": [_hit("1:1.1", 20.0), _hit("1:1.2", 10.0)]},
        {"book_id": "b", "book_name": "Книга B", "model": "m", "results": [_hit("1:1.1", 2.0), _hit("1:1.2", 1.9)]},
    ]
    out = multi.merge_hits(per_book, "fts", 3)
    # raw bm25 of different books is not comparable: the best hit of each book becomes 1.0
    assert [(h["book_id"], h["id"]) for h in out] == [("a", "1:1.1"), ("b", "1:1.1"), ("b", "1:1.2")]
    assert [round(h["score"], 2) for h in out] == [1.0, 1.0, 0.95]
    assert out[0]["book_name"] == "Книга A" and out[0]["score_raw"] == 20.0


def test_merge_semantic_same_model_keeps_cosine_and_different_model_normalises():
    a = {"book_id": "a", "book_name": "A", "model": "m1", "results": [_hit("x", 0.8), _hit("y", 0.5)]}
    b = {"book_id": "b", "book_name": "B", "model": "m1", "results": [_hit("x", 0.6)]}
    out = multi.merge_hits([a, b], "semantic", 5)
    assert [(h["book_id"], h["score"]) for h in out] == [("a", 0.8), ("b", 0.6), ("a", 0.5)]
    b2 = {**b, "model": "m2"}
    out = multi.merge_hits([a, b2], "semantic", 5)
    assert out[0]["score"] == 1.0 and out[1]["score"] == 1.0   # each book's best hit is 1.0 (spaces differ)


def test_merge_single_book_keeps_scores_and_cuts_to_k():
    out = multi.merge_hits([{"book_id": "a", "book_name": "A", "model": "m",
                             "results": [_hit("1", 5.0), _hit("2", 3.0), _hit("3", 1.0)]}], "hybrid", 2)
    assert [h["score"] for h in out] == [5.0, 3.0]


class _FakeEmbedder:
    model_name, runtime, dim = "m", "onnx", 2

    def __init__(self):
        self.calls = 0

    def embed_query(self, text):
        self.calls += 1
        return np.array([1.0, 0.0], dtype=np.float32)


class _FakeSearcher:
    def __init__(self, embedder, score):
        self.embedder, self.score, self.last_timings = embedder, score, {}

    def semantic(self, q, k=20):
        v = self.embedder.embed_query(q)
        self.last_timings = {"semantic_ms": 1.0}
        return [{"id": "1:1.1", "score": float(v[0]) * self.score}]

    hybrid = fts = semantic


class _H:
    def __init__(self, book_id, searcher, model="m", problem=None):
        self.book_id, self.book_name, self.searcher, self.model, self.problem = book_id, book_id.upper(), searcher, model, problem


def test_query_is_embedded_once_for_books_sharing_an_embedder():
    emb = _FakeEmbedder()
    hs = [_H("a", _FakeSearcher(emb, 0.9)), _H("b", _FakeSearcher(emb, 0.7))]
    res, timings, warnings = multi.search_books(hs, "вопрос", "semantic", 5)
    assert emb.calls == 1 and timings["query_embeddings"] == 1
    assert [h["book_id"] for h in res] == ["a", "b"] and not warnings
    assert hs[0].searcher.embedder is emb      # the cached searcher keeps its own embedder, not the wrapper


def test_book_without_index_or_embedder_is_reported_not_fatal():
    emb = _FakeEmbedder()
    hs = [_H("a", _FakeSearcher(emb, 0.9)), _H("b", None, problem="нет базы"),
          _H("c", _FakeSearcher(None, 0.5), problem="модель не найдена")]
    res, _, warnings = multi.search_books(hs, "q", "semantic", 5)
    assert [h["book_id"] for h in res] == ["a"]
    assert {w["book_id"] for w in warnings} == {"b", "c"}


# ------------------------------------------------------------------ API --

def _make_mono_book(book_id="mono", name="Афоризмы"):
    """A one-language (Russian) book in the library, FTS only."""
    path = config.LIBRARY_DIR / book_id / "book.db"
    conn = open_db(str(path), readonly=False)
    conn.execute("INSERT OR REPLACE INTO books (book_id, book_name, language_source, language_translation) "
                 "VALUES (?, ?, 'ru', '')", (book_id, name))
    rows = [("1:0.1", 1, 0, 1, "Любовь Отца выше всякого разума."), ("1:0.2", 1, 0, 2, "Тишина говорит громче слов.")]
    for seq, (pid, paper, sec, par, ru) in enumerate(rows):
        conn.execute("INSERT INTO paragraphs (id, book_id, seq, paper, section, paragraph, en, ru) "
                     "VALUES (?, ?, ?, ?, ?, ?, '', ?)", (pid, book_id, seq, paper, sec, par, ru))
        conn.execute("INSERT INTO paragraphs_fts (id, en, ru, ru_lemma) VALUES (?, '', ?, ?)", (pid, ru, ru.lower()))
    conn.commit()
    conn.close()
    return book_id


@pytest.fixture
def two(api):
    _make_mono_book()
    return api


def test_books_listing_has_names_language_and_counts(two):
    books = {b["book_id"]: b for b in two.get("/api/books").json()["books"]}
    assert books["mono"]["book_name"] == "Афоризмы" and books["mono"]["language"] == "ru"
    assert books["mono"]["bilingual"] is False and books["mono"]["paragraphs"] == 2
    assert books["test_book"]["book_name"] == "Fake Book" and books["test_book"]["bilingual"] is True
    assert books["test_book"]["active"] and books["test_book"]["selected"] and not books["mono"]["selected"]


def test_search_selection_is_saved_and_used_by_default(two):
    r = two.post("/api/books/search-selection", json={"book_ids": ["test_book", "mono"]})
    assert r.status_code == 200
    assert config.load_settings()["search_books"] == ["test_book", "mono"]
    assert two.post("/api/books/search-selection", json={"book_ids": ["nope"]}).status_code == 400
    data = two.get("/api/search", params={"q": "Отца", "mode": "fts"}).json()
    assert data["book_ids"] == ["test_book", "mono"]
    assert {h["book_id"] for h in data["results"]} == {"test_book", "mono"}
    by = {h["book_id"]: h for h in data["results"]}
    assert by["mono"]["book_name"] == "Афоризмы" and by["test_book"]["book_name"] == "Fake Book"


def test_search_explicit_books_override_selection_and_unknown_is_400(two):
    only = two.get("/api/search", params={"q": "Отца", "mode": "fts", "book_ids": "mono"}).json()
    assert {h["book_id"] for h in only["results"]} == {"mono"}
    assert two.get("/api/search", params={"q": "x", "book_ids": "ghost"}).status_code == 400


def test_semantic_for_book_without_embedder_is_a_warning_hybrid_falls_back(two):
    data = two.get("/api/search", params={"q": "Отца", "mode": "hybrid", "book_ids": "test_book,mono"}).json()
    assert data["results"] and {w["book_id"] for w in data["warnings"]} == {"mono", "test_book"}   # no embedder in tests
    sem = two.get("/api/search", params={"q": "Отца", "mode": "semantic", "book_ids": "mono"})
    assert sem.status_code == 200 and sem.json()["results"] == [] and sem.json()["warnings"]   # explained, no crash


def test_monolingual_hit_has_empty_other_language(two):
    hit = two.get("/api/search", params={"q": "тишина", "mode": "fts", "book_ids": "mono"}).json()["results"][0]
    assert hit["en"] == "" and "Тишина" in hit["ru"]


def test_history_keeps_books_and_redraws_from_the_right_one(two):
    two.get("/api/search", params={"q": "Отца", "mode": "fts", "book_ids": "test_book,mono"})
    item = two.get("/api/history").json()["items"][0]
    assert item["params"]["book_ids"] == ["test_book", "mono"]
    e = two.get(f"/api/history/{item['id']}").json()
    texts = {h["book_id"]: h["ru"] for h in e["result"]["results"]}
    assert "Любовь Отца выше" in texts["mono"] and "любви Отца" in texts["test_book"]
    assert e["book_loaded"] and e["missing_paragraphs"] == 0


def test_paragraph_context_read_and_toc_take_book_id(two):
    assert "Любовь Отца" in two.get("/api/paragraph/1:0.1", params={"book_id": "mono"}).json()["ru"]
    ctx = two.get("/api/paragraph/1:0.1/context", params={"book_id": "mono", "before": 1, "after": 1}).json()["context"]
    assert [p["id"] for p in ctx] == ["1:0.1", "1:0.2"]
    assert two.get("/api/read/1", params={"book_id": "mono"}).status_code == 200
    assert two.get("/api/toc", params={"book_id": "mono"}).status_code == 200
    assert two.get("/api/paragraph/1:0.1", params={"book_id": "ghost"}).status_code == 404
    # without book_id the active book answers
    assert two.get("/api/paragraph/1.1.1").json()["en"] == "God is spirit."


def test_rag_with_several_books_answers_over_the_active_one(two):
    job = run_rag(two, book_ids=["mono", "test_book"])
    assert job["status"] == "done"
    assert any("по одной книге" in line for line in job["log"])
    item = two.get("/api/history").json()["items"][0]
    assert item["book_id"] == "test_book" and item["params"]["book_ids"] == ["mono", "test_book"]


def test_agent_search_accepts_book_ids_and_marks_books(two):
    r = two.get("/api/agent/search", params={"q": "Отца", "mode": "fts", "book_ids": "test_book,mono"}).json()
    assert {p["book_id"] for p in r["results"]} == {"test_book", "mono"}
    assert "«Афоризмы»" in r["text"]
    r1 = two.get("/api/agent/search", params={"q": "Отца", "mode": "fts"}).json()   # saved/default selection
    assert {p["book_id"] for p in r1["results"]} == {"test_book"}


# ------------------------------------------------------------------ add a book --

@pytest.fixture
def corpus(api, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CORPUS_DIR", tmp_path / "corpus")
    return tmp_path / "corpus"


def _epub_bytes(tmp_path, title="Основы Йоги", n=3):
    body = "".join(f"<p>{RU1} {i}.</p>" for i in range(n))
    p = make_epub(tmp_path / "src.epub", {"c": f"<h1>Глава</h1>{body}<h2>Раздел</h2><p>{RU2}</p>"}, ["c"], title=title)
    return p.read_bytes()


def test_upload_saves_into_corpus_and_suggests_id(api, corpus, tmp_path):
    r = api.post("/api/books/upload", params={"filename": "../evil/Основы Йоги.epub"}, content=_epub_bytes(tmp_path))
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["file"] == "corpus/Основы Йоги.epub" and (corpus / "Основы Йоги.epub").exists()
    assert d["title"] == "Основы Йоги" and d["author"] == "Автор" and d["language"] == "ru"
    assert d["book_id"] == "osnovy_yogi"
    # the very same file again: reused, no copy
    d1 = api.post("/api/books/upload", params={"filename": "copy.epub"}, content=_epub_bytes(tmp_path)).json()
    assert d1["file"] == "corpus/Основы Йоги.epub" and len(list(corpus.glob("*"))) == 1
    # same name, other content: not overwritten
    d2 = api.post("/api/books/upload", params={"filename": "Основы Йоги.epub"}, content=_epub_bytes(tmp_path, n=5)).json()
    assert d2["file"] == "corpus/Основы Йоги_2.epub"


def test_suggested_id_is_unique_against_library(two):
    from app import server
    assert server.suggest_book_id("Test Book") == "test_book_2"
    assert server.suggest_book_id("", "Афоризмы 1") == "aforizmy_1"
    assert server.suggest_book_id("###") == "book"


def test_upload_rejects_unsupported_and_broken_files(api, corpus):
    assert api.post("/api/books/upload", params={"filename": "x.txt"}, content=b"hi").status_code == 400
    assert api.post("/api/books/upload", params={"filename": "x.epub"}, content=b"").status_code == 400
    r = api.post("/api/books/upload", params={"filename": "bad.epub"}, content=b"not a zip")
    assert r.status_code == 400 and not (corpus / "bad.epub").exists()


def test_preview_counts_chapters_and_paragraphs(api, corpus, tmp_path):
    f = api.post("/api/books/upload", params={"filename": "b.epub"}, content=_epub_bytes(tmp_path, n=4)).json()["file"]
    p = api.get("/api/books/preview", params={"file": f})
    assert p.status_code == 200, p.text
    d = p.json()
    assert d["paragraphs"] == 5 and d["chapters"] >= 1 and d["language"] == "ru" and d["sample"]
    assert api.get("/api/books/preview", params={"file": "../secret.epub"}).status_code == 400


def test_index_job_rejects_unsupported_extension(api, tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("x", encoding="utf-8")
    r = api.post("/api/jobs/index", json={"json_path": str(f), "model_name": "bge-m3", "book_id": "x", "book_name": "X"})
    assert r.status_code == 400 and "не поддерживается" in r.json()["detail"]


def test_delete_book_not_active_only(two):
    assert two.delete("/api/books/test_book").status_code == 409
    two.post("/api/books/search-selection", json={"book_ids": ["test_book", "mono"]})
    assert two.delete("/api/books/mono").status_code == 200
    assert not (config.LIBRARY_DIR / "mono").exists()
    assert config.load_settings()["search_books"] == ["test_book"]
    assert two.delete("/api/books/mono").status_code == 404
    assert two.delete("/api/books/..%2Fx").status_code in (404, 405)


def test_book_info_rename_and_colour(two):
    i = two.get("/api/books/mono/info").json()
    assert i["book_name"] == "Афоризмы" and i["paragraphs"] > 0 and i["chapters"] >= 1
    assert i["language"] == "ru" and i["color"] is None and i["db_size_bytes"] > 0
    r = two.patch("/api/books/mono", json={"book_name": "  Мысли  ", "color": "#AA3300"}).json()
    assert r["book_name"] == "Мысли" and r["color"] == "#aa3300"
    b = {x["book_id"]: x for x in two.get("/api/books").json()["books"]}
    assert b["mono"]["book_name"] == "Мысли" and b["mono"]["color"] == "#aa3300"
    hits = two.get("/api/search", params={"q": "любовь", "mode": "fts", "book_ids": "mono"}).json()["results"]
    assert all(h["book_name"] == "Мысли" for h in hits)
    assert two.patch("/api/books/mono", json={"color": ""}).json()["color"] is None
    assert two.patch("/api/books/mono", json={"color": "red"}).status_code == 400
    assert two.patch("/api/books/mono", json={"book_name": " "}).status_code == 400
    assert two.get("/api/books/nope/info").status_code == 404


def test_upload_of_a_book_already_in_library_is_reported_not_saved(two, corpus, tmp_path):
    r = two.post("/api/books/upload", params={"filename": "a.epub"}, content=_epub_bytes(tmp_path, title="АФОРИЗМЫ!")).json()
    assert r["duplicate_of"]["book_id"] == "mono" and r["duplicate_of"]["reason"] == "то же название"
    assert r["file"] is None and not list(corpus.glob("*")) if corpus.exists() else True
    forced = two.post("/api/books/upload", params={"filename": "a.epub", "force": "true"},
                      content=_epub_bytes(tmp_path, title="Афоризмы")).json()
    assert forced["file"] == "corpus/a.epub" and "duplicate_of" not in forced
    other = two.post("/api/books/upload", params={"filename": "b.epub"}, content=_epub_bytes(tmp_path, title="Другая")).json()
    assert "duplicate_of" not in other


def test_api_returns_display_markup_fields(two):
    from app.core.db import encode_fmt
    conn = open_db(str(config.LIBRARY_DIR / "mono" / "book.db"), readonly=False)
    conn.execute("UPDATE paragraphs SET fmt = ? WHERE id = '1:0.1'", (encode_fmt("Любовь <i>Отца</i> выше всякого разума."),))
    conn.commit()
    conn.close()
    ctx = two.get("/api/paragraph/1:0.1/context", params={"book_id": "mono", "before": 0, "after": 1}).json()["context"]
    assert ctx[0]["ru_html"] == "Любовь <i>Отца</i> выше всякого разума." and ctx[0]["en_html"] is None
    assert ctx[1]["ru_html"] is None
    paras = two.get("/api/read/1", params={"book_id": "mono"}).json()["paragraphs"]
    assert paras[0]["ru_html"] and paras[1]["ru_html"] is None and paras[0]["ru"] == "Любовь Отца выше всякого разума."
    hits = two.get("/api/search", params={"q": "Любовь", "mode": "fts", "book_ids": "mono"}).json()["results"]
    hit = next(h for h in hits if h["id"] == "1:0.1")
    assert hit["ru_html"] and hit["ru"] == "Любовь Отца выше всякого разума." and "<i>" not in hit["ru"]
    assert "<i>" not in (hit.get("snippet_ru") or "")
