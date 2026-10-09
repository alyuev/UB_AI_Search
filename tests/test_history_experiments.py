"""History as an experiment log: snapshots, ratings, filter/sort, compare, exact replay, old-DB migration."""
import datetime
import os
import sqlite3

import pytest

from app import config
from app import history as hist
from app import prompts as P
from app.core.search import Searcher
from app.llm import translation as T
from app.version import __version__
from tests.server_helpers import ONE_ISSUE, ServerFakeLLM, last_history_id, run_rag, run_translation


def setopt(api, **kw):
    r = api.post("/api/settings", json=kw)
    assert r.status_code == 200, r.text


# ------------------------------------------------------------------------------- snapshots ---

def test_rag_snapshot_contents(api):
    setopt(api, temperature=0.3, max_tokens=900, rag_top_k=4, rag_rrf_k=25, rag_candidates_fts=15,
           rag_answer_lang="ru", llm_model="model-A")
    job = run_rag(api)
    assert job["status"] == "done", job
    exp = job["result"]["experiment"]
    call = api.llm.of("answer")[0]
    sys_msg, user_msg = call["messages"]
    # the exact messages the model received
    assert exp["prompts"]["system"] == sys_msg["content"] and exp["prompts"]["user"] == user_msg["content"]
    assert "Что говорил Иисус о любви?" in exp["prompts"]["user"] and "[147.2.2]" in exp["prompts"]["user"]
    assert exp["prompts"]["rewrite_user"] == "Что говорил Иисус о любви?"
    assert exp["prompts"]["rewrite_system"] == api.llm.of("plan")[0]["messages"][0]["content"]
    assert "ALWAYS answer in Russian" in exp["prompts"]["system"]  # «язык ответа» rule was added
    # model & generation parameters
    assert (exp["kind"], exp["model"], exp["temperature"], exp["max_tokens"]) == ("rag", "model-A", 0.3, 900)
    assert call["model"] == "model-A" and call["temperature"] == 0.3 and call["max_tokens"] == 900
    assert exp["context_length"] == 16000 and exp["reasoning"] == {"setting": "auto", "effort": "none"}
    # versions
    v = exp["prompt_versions"]
    assert set(v) == {"rag_system", "rag_formatting", "rag_user_template", "rag_inference_rules",
                      "rag_inference_reminder", "rag_answer_lang_ru", "query_rewrite"}
    assert all(x["source"] == "default" and x["label"].startswith("стандартный · ") for x in v.values())
    assert exp["prompt_texts"]["rag_system"] == P.specs()["rag_system"]["default"]
    # retrieval parameters and counters
    r = exp["retrieval"]
    assert r["k"] == 4 and r["rrf_k"] == 25 and r["candidates_fts_asked"] == 15 and r["candidates_semantic_asked"] == 30
    assert r["fallback_plain_hybrid"] is False and r["sources_lang"] == "both"
    assert r["retrieved"] >= 1 and r["sources_sent"] >= 1 and r["sources_sent"] <= r["with_neighbors"]
    assert exp["plan"]["keywords_en"] == ["love", "Jesus"] and exp["plan"]["summary"].startswith("Ищу:")
    assert exp["options"]["rag_top_k"] == 4
    # timings, stats, app version/date
    assert {"plan_s", "retrieval_s", "context_s", "generation_s", "total_s"} <= set(exp["timings"])
    assert exp["stats"]["prompt_tokens"] == 100 and exp["stats"]["completion_tokens"] == 20
    assert exp["app"] == {"version": __version__, "date": datetime.date.today().isoformat()}
    assert set(exp["search_settings"]) >= {"fts_weights", "fts_mode_hybrid", "fts_mode_standalone", "fts_stopwords"}
    # the very same snapshot is stored with the history entry
    e = api.get(f"/api/history/{last_history_id(api)}").json()
    assert e["has_experiment"] and e["experiment"] == exp and e["mode"] == "rag"
    assert e["result"]["answer"] and e["result"]["sources"]


def test_rag_snapshot_without_plan_marks_plain_fallback(api):
    setopt(api, rag_query_rewrite=False)
    exp = run_rag(api)["result"]["experiment"]
    assert exp["retrieval"]["fallback_plain_hybrid"] is True and exp["plan"] is None
    assert exp["prompts"]["rewrite_system"] is None and "query_rewrite" not in exp["prompt_versions"]


def test_translation_snapshot_contents(api):
    api.llm.review = ONE_ISSUE
    setopt(api, translation_batch_size=2, llm_model="model-T")
    job = run_translation(api)
    assert job["status"] == "done", job
    exp = job["result"]["experiment"]
    review_call = api.llm.of("review")[0]
    assert exp["kind"] == "translation" and exp["model"] == "model-T"
    assert exp["prompts"]["system"] == review_call["messages"][0]["content"]
    assert exp["prompts"]["user"] == review_call["messages"][1]["content"] and "### ID 147.2.1" in exp["prompts"]["user"]
    assert exp["prompts"]["verify_system"] and exp["prompts"]["verify_user"]  # a major issue -> second pass ran
    assert set(exp["prompt_versions"]) == {"translation_review", "translation_verify", "translation_verify_user"}
    assert exp["retrieval"]["paragraphs"] == 4 and exp["retrieval"]["batches"] == 2 and exp["retrieval"]["batch_size"] == 2
    assert exp["stats"]["total_prompt_tokens"] > 0 and exp["timings"]["total_s"] >= 0
    assert exp["app"]["version"] == __version__ and exp["app"]["date"] == datetime.date.today().isoformat()
    # stored with the entry, but not duplicated inside its result
    e = api.get(f"/api/history/{last_history_id(api)}").json()
    assert e["experiment"] == exp and "experiment" not in e["result"] and e["result"]["issues"]


def test_translation_snapshot_keeps_first_prompt_and_short_per_batch_hashes(api, monkeypatch):
    setopt(api, translation_batch_size=1)
    exp = run_translation(api)["result"]["experiment"]
    b = exp["prompts"]["batches"]
    assert exp["prompts"]["batches_total"] == 4 and [x["batch"] for x in b] == [1, 2, 3, 4]
    assert all(set(x) == {"batch", "paragraphs", "chars", "hash"} and x["paragraphs"] == 1 for x in b)
    assert len({x["hash"] for x in b}) == 4  # different paragraphs -> different prompts
    first = exp["prompts"]["user"]
    assert b[0]["chars"] == len(first) and b[0]["hash"] == P.text_hash(first)
    assert "full prompts" not in str(b) and "### ID" not in str(b)  # hashes/sizes only, no texts
    # capped: a long run keeps the first N entries and the real total
    monkeypatch.setattr(T, "MAX_BATCH_PROMPTS", 2)
    exp = run_translation(api)["result"]["experiment"]
    assert len(exp["prompts"]["batches"]) == 2 and exp["prompts"]["batches_total"] == 4


def test_search_snapshot(api):
    r = api.get("/api/search", params={"q": "Jesus", "mode": "fts", "k": 5})
    assert r.status_code == 200
    e = api.get(f"/api/history/{last_history_id(api)}").json()
    exp = e["experiment"]
    assert e["mode"] == "search" and e["has_experiment"]
    assert exp["kind"] == "search" and exp["retrieval"]["mode"] == "fts" and exp["retrieval"]["k"] == 5
    assert exp["app"]["version"] == __version__
    assert set(exp["search_settings"]) >= {"fts_weights", "fts_mode_hybrid", "fts_stopwords"}
    assert exp["prompts"] == {} and exp["prompt_texts"] == {}
    assert "fts_ms" in exp["timings"]
    # an exact replay of a search entry is not offered (no prompts)
    assert api.post("/api/jobs/rag", json={"question": "x", "replay": e["id"]}).status_code == 404


def test_search_without_history_flag_saves_nothing(api):
    api.get("/api/search", params={"q": "Jesus", "mode": "fts", "history": 0})
    assert api.get("/api/history").json()["total"] == 0


# --------------------------------------------------------------------------------- rating ---

def _three_searches(api):
    ids = []
    for word in ("Jesus", "God", "disciples"):
        api.get("/api/search", params={"q": word, "mode": "fts"})
        ids.append(last_history_id(api))
    return ids


def test_rating_and_note_roundtrip(api):
    a, = _three_searches(api)[:1]
    r = api.post(f"/api/history/{a}/rating", json={"rating": 1, "note": "хороший ответ"})
    assert r.status_code == 200 and r.json() == {"ok": True, "rating": 1, "note": "хороший ответ"}
    # each field is optional and only the given one changes
    assert api.post(f"/api/history/{a}/rating", json={"note": "иначе"}).json()["rating"] == 1
    assert api.post(f"/api/history/{a}/rating", json={"rating": -1}).json()["note"] == "иначе"
    e = api.get(f"/api/history/{a}").json()
    assert e["rating"] == -1 and e["note"] == "иначе"
    item = next(i for i in api.get("/api/history").json()["items"] if i["id"] == a)
    assert item["rating"] == -1 and item["note"] == "иначе"


@pytest.mark.parametrize("body,code", [
    ({"rating": 2}, 400), ({"rating": -5}, 400), ({"note": "я" * 4001}, 400), ({"rating": "up"}, 422),
])
def test_rating_validation(api, body, code):
    a = _three_searches(api)[0]
    r = api.post(f"/api/history/{a}/rating", json=body)
    assert r.status_code == code
    assert api.get(f"/api/history/{a}").json()["rating"] == 0


def test_rating_unknown_entry_is_404(api):
    assert api.post("/api/history/999/rating", json={"rating": 1}).status_code == 404


def test_filter_and_sort_by_rating(api):
    a, b, c = _three_searches(api)  # a oldest ... c newest
    api.post(f"/api/history/{a}/rating", json={"rating": 1})
    api.post(f"/api/history/{b}/rating", json={"rating": -1})

    def ids(**params):
        return [i["id"] for i in api.get("/api/history", params=params).json()["items"]]

    assert ids() == [c, b, a]                                   # default: newest first
    assert ids(rating="up") == [a] and ids(rating="down") == [b]
    assert ids(rating="rated") == [b, a]
    assert ids(sort="rating") == [a, c, b]                      # +1, 0, -1
    assert ids(sort="rating", rating="rated") == [a, b]
    assert api.get("/api/history", params={"rating": "up"}).json()["total"] == 1
    assert ids(rating="bogus") == [c, b, a]                     # unknown filter value = no filter


# --------------------------------------------------------------------------------- compare ---

def test_compare_param_prompt_and_metric_differences(api):
    run_rag(api)
    a = last_history_id(api)
    setopt(api, rag_top_k=3, rag_rrf_k=20, temperature=0.9)
    api.put("/api/prompts/rag_formatting", json={"text": "Пиши простым текстом.\nБез списков."})
    api.llm.answer = "Совсем другой, более длинный ответ про любовь Отца [147.2.2][147.2.3]."
    run_rag(api)
    b = last_history_id(api)
    d = api.get("/api/history/compare", params={"a": a, "b": b}).json()
    assert d["a"]["id"] == a and d["b"]["id"] == b
    assert d["same_query"] is True and d["same_mode"] is True and d["has_experiment"] == {"a": True, "b": True}
    diffs = {x["key"]: x for x in d["param_diffs"]}
    assert diffs["options.rag_top_k"]["a"] == "12" and diffs["options.rag_top_k"]["b"] == "3"
    assert diffs["options.rag_top_k"]["label"] == "Сколько абзацев передавать модели"
    assert diffs["retrieval.k"]["b"] == "3" and diffs["temperature"]["b"] == "0.9"
    assert diffs["prompt_versions.rag_formatting"]["b"].startswith("свой · ")
    assert "options.rag_sources_lang" not in diffs and "model" not in diffs  # equal values are not listed
    pd = {x["id"]: x for x in d["prompt_diffs"]}
    assert pd["rag_system"]["changed"] is False
    fmt = pd["rag_formatting"]
    assert fmt["changed"] is True
    ops = {(x["op"], x["text"]) for x in fmt["diff"]}
    assert ("+", "Пиши простым текстом.") in ops and ("+", "Без списков.") in ops
    assert any(op == "-" for op, _ in ops)
    m = {x["label"]: x for x in d["metrics"]}
    assert "Сослался на абзацев" in m  # (the fake book's ids are not in the «N:N.N» form, so 0 citations)
    assert m["Длина ответа, знаков"]["b"] > m["Длина ответа, знаков"]["a"]
    assert m["Токенов в запросе"]["a"] == m["Токенов в запросе"]["b"] == 100
    assert "Время всего, с" in m


def test_compare_identical_runs_has_no_differences(api):
    run_rag(api)
    a = last_history_id(api)
    d = api.get("/api/history/compare", params={"a": a, "b": a}).json()
    assert d["param_diffs"] == [] and all(not x["changed"] for x in d["prompt_diffs"])


def test_compare_entries_without_snapshot_and_other_modes(api):
    api.get("/api/search", params={"q": "Jesus", "mode": "fts"})
    s = last_history_id(api)
    run_rag(api)
    r = last_history_id(api)
    hist.add("test_book", "rag", "старый вопрос", {}, {"answer": "x"}, "ответ")  # no snapshot
    old = last_history_id(api)
    d = api.get("/api/history/compare", params={"a": r, "b": old}).json()
    assert d["has_experiment"] == {"a": True, "b": False} and d["param_diffs"] == [] and d["prompt_diffs"] == []
    assert d["same_query"] is False
    d = api.get("/api/history/compare", params={"a": s, "b": r}).json()
    assert d["same_mode"] is False


@pytest.mark.parametrize("params,codes", [
    ({"a": 1, "b": 999}, (404,)), ({"a": 999, "b": 1}, (404,)), ({"a": "x", "b": 1}, (400, 422)),
    ({"a": 1}, (400, 422)), ({}, (400, 422)),
])
def test_compare_missing_or_invalid_ids(api, params, codes):
    api.get("/api/search", params={"q": "Jesus", "mode": "fts"})
    assert api.get("/api/history/compare", params=params).status_code in codes


# ---------------------------------------------------------------------------------- replay ---

def _fts_spy(monkeypatch):
    calls = []
    orig = Searcher.fts

    def spy(self, query, k=20, lang="both", mode=None):
        cols = self._fts_cols()  # the weights really applied (instance override, else global settings)
        calls.append({"mode": mode, "weights": dict(zip(cols, self._weights(cols))), "stopwords": self.fts_stopwords})
        return orig(self, query, k, lang, mode)

    orig_terms = Searcher.fts_terms

    def spy_terms(self, st, k=20):
        cols = self._fts_cols()
        calls.append({"mode": "or", "weights": dict(zip(cols, self._weights(cols))), "stopwords": self.fts_stopwords})
        return orig_terms(self, st, k)

    monkeypatch.setattr(Searcher, "fts", spy)
    monkeypatch.setattr(Searcher, "fts_terms", spy_terms)  # the word search with verified terms (new path)
    return calls


def test_exact_replay_uses_stored_prompts_model_params_and_survives_deleted_preset(api, monkeypatch):
    spy = _fts_spy(monkeypatch)
    setopt(api, llm_model="model-A", temperature=0.55, max_tokens=777, rag_top_k=3, rag_rrf_k=20, rag_neighbors=1,
           rag_sources_lang="ru", rag_answer_lang="ru", fts_mode_hybrid="or", fts_stopwords=False,
           fts_weights={"en": 1.5, "ru_lemma": 0.25})
    api.put("/api/prompts/rag_system", json={"text": "Свой system-промт."})
    api.put("/api/prompts/rag_user_template", json={"text": "ВОПРОС {question}\n{sources}{inference_reminder}"})
    assert api.post("/api/prompt-presets", json={"name": "mine"}).status_code == 200
    first = run_rag(api, k=5)  # the UI used to send k explicitly: it differs from the setting (3)
    assert first["status"] == "done"
    e1 = last_history_id(api)
    orig_plan, orig_answer = api.llm.of("plan")[0], api.llm.of("answer")[0]
    orig_exp = first["result"]["experiment"]
    assert orig_exp["retrieval"]["k"] == 5 and spy[-1]["mode"] == "or" and spy[-1]["stopwords"] is False
    assert spy[-1]["weights"]["en"] == 1.5

    # ---- everything changes afterwards, the preset is deleted ----
    assert api.delete("/api/prompt-presets/mine").status_code == 200
    api.post("/api/prompts/reset-all")
    setopt(api, llm_model="model-B", temperature=0.9, max_tokens=100, rag_top_k=9, rag_rrf_k=70, rag_neighbors=0,
           rag_sources_lang="en", rag_answer_lang="question", rag_query_rewrite=False, fts_mode_hybrid="and",
           fts_stopwords=True, fts_weights={"en": 4.0})
    changed_settings = config.load_settings()
    live_searcher = api.server.state.searcher
    api.llm.calls.clear()
    spy.clear()

    job = run_rag(api, replay=e1)
    assert job["status"] == "done", job
    plan, answer = api.llm.of("plan")[0], api.llm.of("answer")[0]
    assert answer["messages"] == orig_answer["messages"] and plan["messages"] == orig_plan["messages"]
    assert (answer["model"], answer["temperature"], answer["max_tokens"]) == ("model-A", 0.55, 777)
    exp = job["result"]["experiment"]
    assert exp["retrieval"]["k"] == 5 and exp["preset"] == "mine"
    assert exp["prompt_versions"] == orig_exp["prompt_versions"]  # «свой»/«стандартный» labels preserved
    assert exp["prompt_texts"] == orig_exp["prompt_texts"]
    assert {k: exp["options"][k] for k in ("rag_rrf_k", "rag_neighbors", "rag_sources_lang", "rag_answer_lang")} == \
        {"rag_rrf_k": 20, "rag_neighbors": 1, "rag_sources_lang": "ru", "rag_answer_lang": "ru"}
    assert "Точный повтор записи" in job["result"]["replay_note"]
    # FTS weights / mode / stop words of the ORIGINAL run, for this run only
    assert spy[-1]["mode"] == "or" and spy[-1]["stopwords"] is False
    used = {k: v for k, v in spy[-1]["weights"].items() if k != "id"}
    assert used["en"] == 1.5 and used["ru_lemma"] == 0.25 and used["ru"] == 1.0
    assert exp["search_settings"] == orig_exp["search_settings"]
    # ... and nothing global was touched
    after = config.load_settings()
    assert after == changed_settings
    assert api.server.state.searcher is live_searcher
    assert live_searcher.fts_mode_hybrid == "and" and live_searcher.fts_stopwords is True
    assert live_searcher.fts_weights is None
    # the replay is a history entry of its own, pointing at the original
    e2 = api.get(f"/api/history/{last_history_id(api)}").json()
    assert e2["id"] != e1 and e2["params"] == {"k": 5, "replay": e1}
    # a normal run afterwards is unaffected by the replay
    spy.clear()
    run_rag(api)
    assert spy == [] or spy[-1]["mode"] == "and"  # query preparation is off now -> plain hybrid; no leak either way


def test_replay_translation_uses_stored_prompts_and_batch_size(api):
    setopt(api, translation_batch_size=2, llm_model="model-T")
    api.put("/api/prompts/translation_review", json={"text": "Свой промт проверки."})
    run_translation(api)
    e1 = last_history_id(api)
    first = api.llm.of("review")
    assert len(first) == 2 and first[0]["messages"][0]["content"] == "Свой промт проверки."
    api.post("/api/prompts/reset-all")
    setopt(api, translation_batch_size=30, llm_model="model-other")
    api.llm.calls.clear()
    job = run_translation(api, replay=e1)
    assert job["status"] == "done", job
    again = api.llm.of("review")
    assert len(again) == 2 and [c["messages"] for c in again] == [c["messages"] for c in first]
    assert again[0]["model"] == "model-T"


def test_replay_errors(api):
    run_rag(api)
    rag_id = last_history_id(api)
    run_translation(api)
    tr_id = last_history_id(api)
    hist.add("test_book", "rag", "без снимка", {}, {"answer": "x"}, "ответ")
    nosnap = last_history_id(api)
    assert api.post("/api/jobs/rag", json={"question": "q", "replay": 99999}).status_code == 404
    assert api.post("/api/jobs/rag", json={"question": "q", "replay": tr_id}).status_code == 404  # wrong kind
    assert api.post("/api/jobs/translation-analysis", json={"reference": "147:2", "replay": rag_id}).status_code == 404
    r = api.post("/api/jobs/rag", json={"question": "q", "replay": nosnap})
    assert r.status_code == 400 and "параметры не сохранены" in r.json()["detail"]


def test_replay_of_old_snapshot_without_fts_settings_keeps_current_ones(api):
    """Snapshots written before FTS settings were stored: replay still works (current FTS settings used)."""
    run_rag(api)
    eid = last_history_id(api)
    e = hist.get(eid)
    exp = e["experiment"]
    exp["search_settings"] = {}
    exp["retrieval"].pop("k")
    new = hist.add("test_book", "rag", e["query"], e["params"], e["result"], "s", experiment=exp)
    job = run_rag(api, replay=new)
    assert job["status"] == "done", job
    assert job["result"]["experiment"]["retrieval"]["k"] == 12  # falls back to the saved option


# -------------------------------------------------------------------------------- migration ---

OLD_SCHEMA = """
CREATE TABLE history (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, book_id TEXT, mode TEXT NOT NULL,
    query TEXT NOT NULL, params TEXT, result TEXT, summary TEXT, duration_s REAL,
    status TEXT NOT NULL DEFAULT 'ok', pinned INTEGER NOT NULL DEFAULT 0);
"""


def _make_old_db():
    path = os.path.join(str(config.DATA_DIR), "history.db")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    c = sqlite3.connect(path)
    c.executescript(OLD_SCHEMA)
    c.execute("INSERT INTO history (created_at, book_id, mode, query, params, result, summary, duration_s) "
              "VALUES ('2026-01-01T10:00:00','test_book','rag','старый вопрос','{\"k\": 12}',"
              "'{\"answer\": \"старый ответ\", \"sources\": []}','ответ, 0 источников', 3.5)")
    c.commit()
    c.close()
    return path


def test_old_history_db_gets_new_columns_and_old_entries_open(api):
    path = _make_old_db()
    cols = {r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(history)")}
    assert not {"experiment", "rating", "note"} & cols
    items = api.get("/api/history").json()["items"]  # first access migrates
    cols = {r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(history)")}
    assert {"experiment", "rating", "note"} <= cols
    assert len(items) == 1 and items[0]["rating"] == 0 and items[0]["has_experiment"] is False
    e = api.get(f"/api/history/{items[0]['id']}").json()
    assert e["experiment"] is None and e["has_experiment"] is False and e["result"]["answer"] == "старый ответ"
    # the old entry can be rated, compared and is refused for exact replay with the «not saved» message
    assert api.post(f"/api/history/{e['id']}/rating", json={"rating": 1}).status_code == 200
    run_rag(api)
    new = last_history_id(api)
    d = api.get("/api/history/compare", params={"a": e["id"], "b": new}).json()
    assert d["has_experiment"] == {"a": False, "b": True}
    r = api.post("/api/jobs/rag", json={"question": "q", "replay": e["id"]})
    assert r.status_code == 400 and "параметры не сохранены" in r.json()["detail"]
    assert api.get("/api/history", params={"rating": "up"}).json()["total"] == 1


def test_migration_is_idempotent_and_keeps_data(api):
    _make_old_db()
    for _ in range(3):
        assert api.get("/api/history").json()["total"] == 1
    hist.add("test_book", "search", "q", {}, {"results": []}, "ничего", experiment={"kind": "search"})
    assert api.get("/api/history").json()["total"] == 2
