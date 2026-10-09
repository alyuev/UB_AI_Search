"""Server level: the glossary job, account state of OpenRouter, backup models in profiles, partial answers."""
import httpx

from app.llm import providers as P
from app.llm.providers import ProviderError
from tests.provider_fakes import FakeOpenAI
from tests.server_helpers import fake_stats, last_history_id, run_rag, wait_job
from tests.test_providers_api import add_openrouter, allow_cloud, use_real_client


# ------------------------------------------------------------------ glossary job ---

def test_build_glossary_job_and_status_card(api):
    st = api.get("/api/db/status").json()
    assert st["glossary"]["exists"] is False
    assert any(r["status"] == "warn" and "Словарь терминов" in r["text"] for r in st["rows"])
    assert any(w["action"] and w["action"]["id"] == "build_glossary" for w in st["warnings"])
    r = api.post("/api/jobs/build-glossary")
    assert r.status_code == 200
    job = wait_job(api, r.json()["job_id"])
    assert job["status"] == "done" and job["result"]["cancelled"] is False and "pairs" in job["result"]
    assert [s["name"] for s in job["stages"]] == ["glossary"] and job["stage"] == "glossary"
    st = api.get("/api/db/status").json()
    assert st["glossary"]["exists"] is True
    assert any(r["status"] == "ok" and "Словарь терминов (русские ↔ английские слова): есть" in r["text"] for r in st["rows"])
    assert not any(w["action"] and w["action"]["id"] == "build_glossary" for w in st["warnings"])
    # a second run replaces the table in place
    assert wait_job(api, api.post("/api/jobs/build-glossary").json()["job_id"])["status"] == "done"


def test_glossary_job_errors_are_plain(api, monkeypatch):
    api.server.state.conn  # app is loaded
    monkeypatch.setattr(api.server.config, "LIBRARY_DIR", api.server.config.LIBRARY_DIR / "elsewhere")
    r = api.post("/api/jobs/build-glossary")
    assert r.status_code == 404 and "База книги не найдена" in r.json()["detail"]


def test_search_works_before_and_after_glossary(api):
    q = {"q": "Jesus", "mode": "fts"}
    before = [x["id"] for x in api.get("/api/search", params=q).json()["results"]]
    wait_job(api, api.post("/api/jobs/build-glossary").json()["job_id"])
    after = [x["id"] for x in api.get("/api/search", params=q).json()["results"]]
    assert before == after and before


# --------------------------------------------------------------- account state ---

def test_account_endpoint_in_plain_words(api, monkeypatch):
    fake = use_real_client(api, monkeypatch, FakeOpenAI(native={"/api/v1/key": {"data": {
        "is_free_tier": True, "usage": 0.25, "limit": None}}}))
    pid = add_openrouter(api)
    r = api.get(f"/api/llm/providers/{pid}/account").json()
    assert r["available"] is False and "выключены" in r["reason"]          # cloud off: nobody calls the service
    assert not [x for x in fake.requests if x.url.path.endswith("/key")]
    allow_cloud(api)
    r = api.get(f"/api/llm/providers/{pid}/account").json()
    assert r["available"] is True and r["is_free_tier"] is True and "ни разу не пополнялся" in r["summary"]
    assert "не задан" in r["summary"] and "TOPSECRET" not in str(r)
    assert api.get("/api/llm/providers/nope/account").status_code == 404
    local = api.get(f"/api/llm/providers/{P.LOCAL_ID}/account").json()
    assert local["available"] is False


def test_account_endpoint_without_key_or_on_error(api, monkeypatch):
    use_real_client(api, monkeypatch, FakeOpenAI(native={"/api/v1/key": httpx.Response(401, json={"error": {"message": "bad"}})}))
    r = api.post("/api/llm/providers", json={"name": "OR", "kind": "openrouter", "base_url": "https://openrouter.ai/api/v1"})
    pid = r.json()["id"]
    allow_cloud(api)
    assert api.get(f"/api/llm/providers/{pid}/account").json() == {"available": False, "reason": "Ключ не введён."}
    api.put(f"/api/llm/providers/{pid}", json={"api_key": "sk-or-v1-abcdefghijklmnop1234"})
    r = api.get(f"/api/llm/providers/{pid}/account").json()
    assert r["available"] is False and "ключ" in r["reason"]


# --------------------------------------------------------------- backup models ---

def test_fallback_models_saved_validated_and_shown(api):
    pid = add_openrouter(api, fallback_models=["a/x:free", "openrouter/free"])
    p = next(x for x in api.get("/api/llm/providers").json()["providers"] if x["id"] == pid)
    assert p["fallback_models"] == ["a/x:free", "openrouter/free"]
    assert api.put(f"/api/llm/providers/{pid}", json={"fallback_models": ["a"] * 9}).status_code == 400
    r = api.put(f"/api/llm/providers/{pid}", json={"fallback_models": []})
    assert next(x for x in r.json()["providers"] if x["id"] == pid)["fallback_models"] == []


def test_backup_model_used_end_to_end_and_reported(api, monkeypatch):
    from tests.provider_fakes import sse

    fake = FakeOpenAI()
    use_real_client(api, monkeypatch, fake)
    pid = add_openrouter(api, default_model="main/model:free", fallback_models=["backup/model:free"])
    allow_cloud(api)
    api.post(f"/api/llm/providers/{pid}/activate")
    api.post("/api/settings", json={"llm_model": "main/model:free", "rag_query_rewrite": False})
    # OpenRouter answers the request with its own failover: the reply names the model that really answered
    fake.script = [httpx.Response(200, content=sse("Ответ из запасной [147.2.2].", {"prompt_tokens": 9, "completion_tokens": 8},
                                                  model="backup/model:free"), headers={"content-type": "text/event-stream"})]
    job = run_rag(api)
    assert job["status"] == "done", job["error"]
    res = job["result"]
    body = fake.chats[-1]["body"]
    assert body["model"] == "main/model:free" and body["models"] == ["main/model:free", "backup/model:free"]
    assert res["answered_model"] == "backup/model:free" and "переключил" in res["fallback_note"]
    assert "ответила модель «backup/model:free»" in res["stats_summary"]


# --------------------------------------------------------------- stage labels ---

def test_generation_status_names_the_real_provider(api, monkeypatch):
    fake = FakeOpenAI(reply="Ответ [147.2.2].")
    use_real_client(api, monkeypatch, fake)
    pid = add_openrouter(api, default_model="nvidia/nemotron:free")
    allow_cloud(api)
    api.post(f"/api/llm/providers/{pid}/activate")
    api.post("/api/settings", json={"llm_model": "nvidia/nemotron:free", "rag_query_rewrite": False})
    job = run_rag(api)
    assert job["status"] == "done"
    log = "\n".join(job["log"])
    assert "отправка в OpenRouter · nvidia/nemotron:free" in log and "LM Studio" not in log


# --------------------------------------------------------------- partial answer ---

def test_partial_answer_is_a_job_error_but_kept_and_recorded(api):
    def cut(messages, model, **kw):
        if messages[-1]["content"].startswith("Sources"):
            e = ProviderError("Провайдер модели (Nvidia) перегружен (502). Ответ оборвался на середине.")
            e.partial, e.partial_stats = "Иисус говорил о любви Отца [147.2.2], но потом", fake_stats()
            raise e
        return "{}", fake_stats()

    api.llm.chat_with_stats = cut
    job = run_rag(api)
    assert job["status"] == "error" and "Nvidia" in job["error"]
    res = job["result"]
    assert res["truncated"] is True and res["answer"].startswith("Иисус говорил")
    assert any("Ответ оборван" in w for w in res["warnings"])
    h = api.get(f"/api/history/{last_history_id(api)}").json()
    assert h["summary"] == "ответ оборван на середине" and h["result"]["truncated"] is True


def test_error_without_any_text_stays_a_plain_error_without_history(api):
    def boom(messages, model, **kw):
        raise ProviderError("Провайдер модели (Nvidia) перегружен (429).")

    api.llm.chat_with_stats = boom
    n_before = api.get("/api/history").json()["total"]
    job = run_rag(api)
    assert job["status"] == "error" and "Nvidia" in job["error"] and not job["result"]["answer"]
    assert api.get("/api/history").json()["total"] == n_before
