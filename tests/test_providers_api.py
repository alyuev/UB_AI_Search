"""Providers through the HTTP API: profiles, secrets, cloud gating, per-task override, history snapshots."""
import json
import logging

import pytest

from app import config
from app.llm import providers as P
from tests.provider_fakes import FakeOpenAI
from tests.server_helpers import ServerFakeLLM, last_history_id, run_rag, run_translation

KEY = "sk-or-v1-TOPSECRETTOPSECRET9999"


def use_real_client(api, monkeypatch, fake=None):
    fake = fake or FakeOpenAI(usage={"prompt_tokens": 500, "completion_tokens": 50, "total_tokens": 550, "cost": 0.0031})
    monkeypatch.setattr(api.server, "get_llm_client", api.server._real_get_llm_client)
    monkeypatch.setattr(P, "_TEST_TRANSPORT", fake.transport())
    return fake


def add_openrouter(api, **extra):
    body = {"name": "OpenRouter", "kind": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key": KEY,
            "default_model": "acme/think-1", **extra}
    r = api.post("/api/llm/providers", json=body)
    assert r.status_code == 200, r.text
    return r.json()["id"]


def allow_cloud(api):
    r = api.post("/api/llm/cloud-access", json={"allow": True, "confirm": True})
    assert r.status_code == 200, r.text


def all_files_text(api) -> str:
    out = []
    for p in config.DATA_DIR.rglob("*"):
        if p.is_file() and p.name != "secrets.json" and p.suffix in (".json", ".db", ".tmp"):
            out.append(p.read_bytes().decode("utf-8", "ignore"))
    return "\n".join(out)


# ------------------------------------------------------------------ basics ---

def test_default_profile_is_migrated_lmstudio(api):
    j = api.get("/api/llm/providers").json()
    assert [p["id"] for p in j["providers"]] == [P.LOCAL_ID] and j["active"] == P.LOCAL_ID
    p = j["providers"][0]
    assert p["name"] == "LM Studio (локально)" and p["is_cloud"] is False and p["icon"] == "🖥" and p["active"]
    assert j["allow_cloud_llm"] is False
    assert {k["id"] for k in j["kinds"]} == set(P.KINDS)
    assert api.get("/api/settings").json()["allow_cloud_llm"] is False


def test_old_lmstudio_url_setting_keeps_working(api):
    api.post("/api/settings", json={"lmstudio_url": "http://localhost:4000", "llm_model": "m1"})
    p = api.get("/api/llm/providers").json()["providers"][0]
    assert p["base_url"] == "http://localhost:4000"
    pid = api.post("/api/llm/providers", json={"name": "Ollama", "kind": "ollama", "base_url": "http://localhost:11434"}).json()["id"]
    api.post("/api/settings", json={"lmstudio_url": "http://localhost:5000", "llm_model": "m2"})
    st = api.get("/api/llm/providers").json()
    loc = next(x for x in st["providers"] if x["id"] == P.LOCAL_ID)
    assert loc["base_url"] == "http://localhost:5000" and loc["default_model"] == "m2" and pid


def test_create_update_delete_validation(api):
    bad = api.post("/api/llm/providers", json={"name": "x", "kind": "ollama", "base_url": "localhost"})
    assert bad.status_code == 400 and "Адрес" in bad.json()["detail"]
    assert api.post("/api/llm/providers", json={"name": "", "kind": "ollama", "base_url": "http://x"}).status_code == 400
    pid = api.post("/api/llm/providers", json={"name": "Мой Ollama", "kind": "ollama", "base_url": "http://localhost:11434"}).json()["id"]
    assert pid == "moj-ollama"
    r = api.put(f"/api/llm/providers/{pid}", json={"name": "Ollama 2", "timeout": 99, "default_model": "llama3"})
    assert r.status_code == 200
    p = next(x for x in r.json()["providers"] if x["id"] == pid)
    assert p["name"] == "Ollama 2" and p["timeout"] == 99 and p["default_model"] == "llama3" and p["kind"] == "ollama"
    assert api.put(f"/api/llm/providers/{pid}", json={"timeout": -1}).status_code == 400
    assert api.put("/api/llm/providers/nope", json={}).status_code == 404
    assert api.delete("/api/llm/providers/nope").status_code == 404
    assert api.delete(f"/api/llm/providers/{pid}").status_code == 200
    assert api.delete(f"/api/llm/providers/{P.LOCAL_ID}").status_code == 400  # the last one stays


def test_generic_settings_cannot_touch_providers(api):
    for payload in ({"llm_providers": []}, {"active_llm_provider": "x"}, {"allow_cloud_llm": True},
                    {"llm_provider_rag": "nope"}, {"llm_model_rag": 5}, {"openrouter_title": "a\nb"}):
        assert api.post("/api/settings", json=payload).status_code == 400, payload
    assert api.post("/api/settings", json={"allow_cloud_llm": False, "openrouter_title": "SmartSearch"}).status_code == 200


# ----------------------------------------------------------------- secrets ---

def test_key_never_returned_or_stored_in_settings(api):
    pid = add_openrouter(api)
    st = api.get("/api/llm/providers")
    p = next(x for x in st.json()["providers"] if x["id"] == pid)
    assert p["has_key"] is True and p["last4"] == KEY[-4:] and "api_key" not in p
    for url in ("/api/llm/providers", "/api/settings", "/api/status", f"/api/llm/check?provider={pid}"):
        assert KEY not in api.get(url).text and "TOPSECRET" not in api.get(url).text, url
    assert KEY not in all_files_text(api) and "TOPSECRET" not in all_files_text(api)
    assert "api_key" not in config.SETTINGS_PATH.read_text(encoding="utf-8")
    assert KEY not in (config.DATA_DIR / "secrets.json").read_text(encoding="utf-8")  # encrypted (or at least not raw)
    # replace and clear
    api.put(f"/api/llm/providers/{pid}", json={"api_key": "sk-new-KEY-0000-ABCD"})
    assert next(x for x in api.get("/api/llm/providers").json()["providers"] if x["id"] == pid)["last4"] == "ABCD"
    api.put(f"/api/llm/providers/{pid}", json={"clear_key": True})
    assert next(x for x in api.get("/api/llm/providers").json()["providers"] if x["id"] == pid)["has_key"] is False
    assert api.post("/api/llm/providers", json={"name": "bad", "kind": "openrouter", "base_url": "https://x.io/v1",
                                                "api_key": "has space"}).status_code == 400


def test_delete_removes_key(api):
    pid = add_openrouter(api)
    api.delete(f"/api/llm/providers/{pid}")
    assert KEY not in (config.DATA_DIR / "secrets.json").read_text(encoding="utf-8")
    from app import secrets_store
    assert secrets_store.get_key(pid) is None


def test_prompt_preset_export_has_no_key(api):
    add_openrouter(api)
    api.post("/api/prompt-presets", json={"name": "mine"})
    r = api.get("/api/prompt-presets-export")
    assert r.status_code == 200 and KEY not in r.text and "api_key" not in r.text and "openrouter" not in r.text.lower()


# ------------------------------------------------------------------ gating ---

def test_cloud_blocked_until_allowed_with_confirmation(api, monkeypatch):
    use_real_client(api, monkeypatch)
    pid = add_openrouter(api)
    r = api.post(f"/api/llm/providers/{pid}/activate")
    assert r.status_code == 400 and "выключен" in r.json()["detail"]
    assert api.post(f"/api/llm/providers/{pid}/check").json()["ok"] is False
    assert api.get(f"/api/llm/models?provider={pid}").status_code == 400
    assert api.get("/api/llm/models?provider=" + pid).json()["detail"].startswith("Облачный провайдер")
    # a job through a cloud provider chosen for a task is refused up front
    api.post("/api/settings", json={"llm_provider_rag": pid})
    r = api.post("/api/jobs/rag", json={"question": "Что говорил Иисус о любви?"})
    assert r.status_code == 400 and "Облачный" in r.json()["detail"]
    api.post("/api/settings", json={"llm_provider_rag": ""})
    # explicit confirmation required
    assert api.post("/api/llm/cloud-access", json={"allow": True}).status_code == 400
    assert api.post("/api/llm/cloud-access", json={"allow": True, "confirm": "yes"}).status_code == 400
    assert api.get("/api/settings").json()["allow_cloud_llm"] is False
    assert "внешний сервер" in api.get("/api/llm/providers").json()["cloud_warning"]
    allow_cloud(api)
    assert api.post(f"/api/llm/providers/{pid}/activate").status_code == 200
    assert api.get("/api/llm/providers").json()["active"] == pid
    assert api.get("/api/settings").json()["llm_model"] == "acme/think-1"


def test_disabling_cloud_returns_to_local(api, monkeypatch):
    use_real_client(api, monkeypatch)
    pid = add_openrouter(api)
    allow_cloud(api)
    api.post(f"/api/llm/providers/{pid}/activate")
    j = api.post("/api/llm/cloud-access", json={"allow": False}).json()
    assert j["allow_cloud_llm"] is False and j["active"] == P.LOCAL_ID
    assert api.get("/api/settings").json()["llm_model"] == ""


def test_status_badge_info_and_no_cloud_ping(api, monkeypatch):
    fake = use_real_client(api, monkeypatch, FakeOpenAI(native={"/api/v0/models": {"data": []}}))
    s = api.get("/api/status").json()
    assert s["llm_active"]["is_cloud"] is False and s["llm_active"]["kind"] == "lmstudio" and "key" not in json.dumps(s).lower().replace("keywords", "")
    pid = add_openrouter(api)
    allow_cloud(api)
    api.post(f"/api/llm/providers/{pid}/activate")
    n = len(fake.requests)
    s = api.get("/api/status").json()
    la = s["llm_active"]
    assert la["is_cloud"] and la["name"] == "OpenRouter" and la["model"] == "acme/think-1" and la["host"] == "openrouter.ai"
    assert len(fake.requests) == n  # the status poll does not hit the cloud


# --------------------------------------------------------- models / check ---

def test_models_endpoint_filters_and_test_button(api, monkeypatch):
    fake = use_real_client(api, monkeypatch)
    pid = add_openrouter(api, extra_headers={"X-Custom": "1"})
    allow_cloud(api)
    j = api.get(f"/api/llm/models?provider={pid}").json()
    assert j["total"] == 2 and {m["id"] for m in j["models"]} == {"meta/llama-free:free", "acme/think-1"}
    assert api.get(f"/api/llm/models?provider={pid}&free_only=true").json()["models"][0]["id"] == "meta/llama-free:free"
    assert api.get(f"/api/llm/models?provider={pid}&min_context=100000").json()["models"][0]["id"] == "acme/think-1"
    assert api.get(f"/api/llm/models?provider={pid}&q=THINK").json()["models"][0]["id"] == "acme/think-1"
    assert api.get(f"/api/llm/models?provider={pid}&q=zzz").json()["models"] == []
    assert api.get("/api/llm/models?provider=nope").status_code == 404
    api.get(f"/api/llm/models?provider={pid}")
    assert fake.model_requests == 1  # server-side cache
    api.get(f"/api/llm/models?provider={pid}&refresh=true")
    assert fake.model_requests == 2
    r = api.post(f"/api/llm/providers/{pid}/test", json={"model": "acme/think-1"}).json()
    assert r["ok"] and r["cost"] == 0.0031 and r["latency_s"] >= 0 and fake.last_chat()["body"]["max_tokens"] == 10
    assert fake.last_chat()["headers"]["x-custom"] == "1"
    assert api.post(f"/api/llm/providers/{pid}/check").json()["ok"] is True
    fake.script = [__import__("httpx").Response(401, json={"error": {"message": f"nope {KEY}"}})]
    r = api.post(f"/api/llm/providers/{pid}/test", json={"model": "acme/think-1"}).json()
    assert r["ok"] is False and "ключ" in r["message"] and KEY not in json.dumps(r)


# ---------------------------------------------------- end to end (cloud) ---

def test_cloud_rag_records_cost_provider_and_never_the_key(api, monkeypatch):
    fake = use_real_client(api, monkeypatch)
    records = []

    class H(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())
    h = H()
    logging.getLogger("smartsearch.app").addHandler(h)
    try:
        pid = add_openrouter(api)
        allow_cloud(api)
        api.post(f"/api/llm/providers/{pid}/activate")
        job = run_rag(api)
        assert job["status"] == "done", job
        res = job["result"]
        assert "стоимость ≈ $0.0031" in res["stats_summary"] and res["stats"]["cost"] == 0.0031
        assert res["provider"] == {"id": pid, "name": "OpenRouter", "kind": "openrouter", "host": "openrouter.ai",
                                   "is_cloud": True, "model": "acme/think-1", "cost": 0.0031}
        assert res["experiment"]["provider"] == res["provider"]
        # every request used the cloud model and the key header; nothing leaked into the saved data
        assert all(c["body"]["model"] == "acme/think-1" for c in fake.chats)
        assert fake.last_chat()["headers"]["authorization"] == f"Bearer {KEY}"
        hid = last_history_id(api)
        entry = api.get(f"/api/history/{hid}")
        assert entry.json()["experiment"]["provider"]["name"] == "OpenRouter"
        row = api.get("/api/history").json()["items"][0]
        assert row["params"]["cloud"] is True and row["params"]["provider"] == "OpenRouter"
        assert api.get(f"/api/history/{hid}").json()["result"]["provider"]["is_cloud"] is True
        for text in (entry.text, api.get("/api/history").text, json.dumps(job), all_files_text(api)):
            assert KEY not in text and "TOPSECRET" not in text
    finally:
        logging.getLogger("smartsearch.app").removeHandler(h)
    assert records and not any(KEY in m or "TOPSECRET" in m for m in records)
    assert any("host=openrouter.ai" in m and "model=acme/think-1" in m for m in records)
    assert not any("Что говорил Иисус" in m for m in records)  # prompts are never logged


def test_cloud_error_job_is_russian_and_key_free(api, monkeypatch):
    fake = use_real_client(api, monkeypatch)
    pid = add_openrouter(api)
    allow_cloud(api)
    api.post(f"/api/llm/providers/{pid}/activate")
    import httpx
    fake.script = [httpx.Response(401, json={"error": {"message": f"bad {KEY}"}})] * 3
    job = run_rag(api)
    assert job["status"] == "error" and "ключ" in json.dumps(job, ensure_ascii=False)
    assert KEY not in json.dumps(job)


def test_cloud_translation_snapshot(api, monkeypatch):
    fake = use_real_client(api, monkeypatch, FakeOpenAI(reply="[]", usage={"prompt_tokens": 10, "completion_tokens": 2,
                                                                          "total_tokens": 12, "cost": 0.002}))
    pid = add_openrouter(api)
    allow_cloud(api)
    api.post(f"/api/llm/providers/{pid}/activate")
    job = run_translation(api)
    assert job["status"] == "done", job
    pr = job["result"]["provider"]
    assert pr["is_cloud"] and pr["model"] == "acme/think-1" and pr["cost"] and pr["cost"] > 0
    assert "стоимость ≈ $" in job["result"]["stats_summary"]
    exp = api.get(f"/api/history/{last_history_id(api)}").json()["experiment"]
    assert exp["provider"]["host"] == "openrouter.ai" and KEY not in json.dumps(exp)


# ------------------------------------------------- per-task provider override ---

def test_per_task_override_rewrite_local_answer_cloud(api, monkeypatch):
    local_llm, cloud_llm = ServerFakeLLM(), ServerFakeLLM()
    pid = add_openrouter(api, default_model="cloud-model")
    allow_cloud(api)
    fakes = {P.LOCAL_ID: local_llm, pid: cloud_llm}
    monkeypatch.setattr(api.server, "get_llm_client", lambda provider_id=None: fakes[provider_id or P.LOCAL_ID])
    assert api.post("/api/settings", json={"llm_provider_rag": pid, "rag_query_rewrite": True}).status_code == 200
    job = run_rag(api)
    assert job["status"] == "done", job
    assert len(local_llm.of("plan")) == 1 and not local_llm.of("answer")      # query preparation stays local
    assert len(cloud_llm.of("answer")) == 1 and not cloud_llm.of("plan")
    assert cloud_llm.of("answer")[0]["model"] == "cloud-model"
    assert job["result"]["provider"]["id"] == pid
    exp = job["result"]["experiment"]
    assert exp["rewrite_provider"]["id"] == P.LOCAL_ID and exp["provider"]["id"] == pid

    # and a model override for the cloud task; translation still on the active (local) one
    api.post("/api/settings", json={"llm_model_rag": "other-cloud-model"})
    local_llm.calls.clear(); cloud_llm.calls.clear()
    run_rag(api)
    assert cloud_llm.of("answer")[0]["model"] == "other-cloud-model"
    run_translation(api)
    assert not cloud_llm.of("review")


def test_rewrite_to_cloud_while_answer_local(api, monkeypatch):
    local_llm, cloud_llm = ServerFakeLLM(), ServerFakeLLM()
    pid = add_openrouter(api, default_model="tiny")
    allow_cloud(api)
    fakes = {P.LOCAL_ID: local_llm, pid: cloud_llm}
    monkeypatch.setattr(api.server, "get_llm_client", lambda provider_id=None: fakes[provider_id or P.LOCAL_ID])
    api.post("/api/settings", json={"llm_provider_rewrite": pid, "rag_query_rewrite": True})
    run_rag(api)
    assert len(cloud_llm.of("plan")) == 1 and cloud_llm.of("plan")[0]["model"] == "tiny"
    assert local_llm.of("answer") and not cloud_llm.of("answer")


def test_no_override_uses_active_for_everything(api):
    run_rag(api)
    assert api.llm.of("plan") and api.llm.of("answer")


def test_deleting_provider_clears_task_overrides(api):
    pid = add_openrouter(api)
    allow_cloud(api)
    api.post("/api/settings", json={"llm_provider_rag": pid})
    api.delete(f"/api/llm/providers/{pid}")
    assert api.get("/api/settings").json()["llm_provider_rag"] == ""


def test_provider_param_override_applies(api):
    api.put(f"/api/llm/providers/{P.LOCAL_ID}", json={"params": {"temperature": 0.7, "max_tokens": 321}})
    run_rag(api)
    a = api.llm.of("answer")[0]
    assert a["temperature"] == 0.7 and a["max_tokens"] == 321
    assert api.put(f"/api/llm/providers/{P.LOCAL_ID}", json={"params": {"temperature": 9}}).status_code == 400


def test_exact_replay_keeps_original_provider(api, monkeypatch):
    local_llm, cloud_llm = ServerFakeLLM(), ServerFakeLLM()
    pid = add_openrouter(api, default_model="cm")
    allow_cloud(api)
    fakes = {P.LOCAL_ID: local_llm, pid: cloud_llm}
    monkeypatch.setattr(api.server, "get_llm_client", lambda provider_id=None: fakes[provider_id or P.LOCAL_ID])
    api.post("/api/settings", json={"llm_provider_rag": pid})
    run_rag(api)
    e1 = last_history_id(api)
    api.post("/api/settings", json={"llm_provider_rag": ""})
    cloud_llm.calls.clear(); local_llm.calls.clear()
    job = run_rag(api, replay=e1)
    assert job["status"] == "done", job
    assert cloud_llm.of("answer") and not local_llm.of("answer")
