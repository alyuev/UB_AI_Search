"""Provider layer: generic client, errors, retries, model lists, reasoning mapping, profiles, secrets."""
import json
import sys

import httpx
import pytest

from app import secrets_store
from app.llm import providers as P
from tests.provider_fakes import FakeOpenAI, OPENROUTER_MODELS, sse

KEY = "sk-or-v1-SECRETSECRETSECRET1234"
MSG = [{"role": "user", "content": "hi"}]


def mk(kind="openrouter", url=None, key=KEY, fake=None, monkeypatch=None, **prof):
    fake = fake or FakeOpenAI()
    P._TEST_TRANSPORT = fake.transport()
    profile = P.normalize_profile({"id": "p1", "name": "P", "kind": kind,
                                   "base_url": url or P.KINDS[kind]["default_url"] or "https://api.example.com/v1", **prof})
    return P.ProviderClient(profile, api_key=key, timeout=5,
                            app_headers={"HTTP-Referer": "https://me.example", "X-Title": "SmartSearch"}), fake


# ------------------------------------------------------------- chat ---

def test_streaming_usage_and_reported_cost_openrouter():
    fake = FakeOpenAI(usage={"prompt_tokens": 812, "completion_tokens": 431, "total_tokens": 1243, "cost": 0.0031,
                             "completion_tokens_details": {"reasoning_tokens": 5}})
    c, _ = mk(fake=fake)
    deltas = []
    text, st = c.chat_with_stats(MSG, "acme/think-1", on_delta=deltas.append)
    assert text == fake.reply and "".join(deltas) == text and len(deltas) > 1
    assert (st.prompt_tokens, st.completion_tokens, st.reasoning_tokens) == (812, 431, 5)
    assert st.cost == 0.0031 and not st.cost_estimated
    assert "стоимость ≈ $0.0031" in st.summary_line()
    sent = fake.last_chat()
    assert sent["body"]["usage"] == {"include": True} and sent["body"]["stream"] is True
    assert sent["headers"]["authorization"] == f"Bearer {KEY}"
    assert sent["headers"]["http-referer"] == "https://me.example" and sent["headers"]["x-title"] == "SmartSearch"
    assert fake.last_chat()["body"]["messages"] == MSG


def test_cost_estimated_from_cached_prices():
    fake = FakeOpenAI(usage={"prompt_tokens": 1000, "completion_tokens": 1000, "total_tokens": 2000})
    c, _ = mk(fake=fake)
    c.list_models()  # fills the price cache: 3 / 15 USD per 1M tokens
    _, st = c.chat_with_stats(MSG, "acme/think-1")
    assert st.cost == pytest.approx(0.018) and st.cost_estimated


def test_local_kind_has_no_cost_and_think_is_stripped():
    fake = FakeOpenAI(reply="<think>секрет</think>Привет")
    c, _ = mk("ollama", fake=fake, key=None)
    text, st = c.chat_with_stats(MSG, "llama3")
    assert text == "Привет" and st.cost is None
    assert "стоимость" not in st.summary_line()
    assert "authorization" not in fake.last_chat()["headers"]
    assert fake.last_chat()["body"]["stream_options"] == {"include_usage": True} and "usage" not in fake.last_chat()["body"]


@pytest.mark.parametrize("code,word", [(401, "ключ"), (402, "средств"), (403, "запрещён"), (404, "не найден"),
                                       (500, "стороне провайдера"), (503, "стороне провайдера"), (400, "отклонил")])
def test_http_errors_in_plain_russian(code, word):
    fake = FakeOpenAI()
    fake.script = [httpx.Response(code, json={"error": {"message": "boom"}})]
    c, _ = mk(fake=fake)
    with pytest.raises(P.ProviderError) as ei:
        c.chat_with_stats(MSG, "acme/think-1")
    assert word in str(ei.value) and str(code) in str(ei.value)


def test_error_text_never_contains_key():
    fake = FakeOpenAI()
    fake.script = [httpx.Response(401, json={"error": {"message": f"bad key {KEY} given"}})]
    c, _ = mk(fake=fake)
    with pytest.raises(P.ProviderError) as ei:
        c.chat_with_stats(MSG, "m")
    assert KEY not in str(ei.value) and "SECRET" not in str(ei.value)


def test_429_retries_honouring_retry_after(monkeypatch):
    sleeps = []
    monkeypatch.setattr(P, "_sleep", sleeps.append)
    fake = FakeOpenAI()
    fake.script = [httpx.Response(429, headers={"retry-after": "3"}), httpx.Response(429, headers={"retry-after": "3"})]
    c, _ = mk(fake=fake)
    text, _ = c.chat_with_stats(MSG, "m")
    assert text == fake.reply and sleeps == [3.0, 3.0] and len(fake.chats) == 3


def test_429_gives_up_after_three_retries_with_backoff(monkeypatch):
    sleeps = []
    monkeypatch.setattr(P, "_sleep", sleeps.append)
    fake = FakeOpenAI()
    fake.script = [httpx.Response(429)] * 5
    c, _ = mk(fake=fake)
    with pytest.raises(P.ProviderError) as ei:
        c.chat_with_stats(MSG, "m")
    assert "429" in str(ei.value) and "лимит" in str(ei.value)
    assert len(fake.chats) == 4 and sleeps == [1.0, 2.0, 4.0]


def test_timeout_and_offline_messages():
    fake = FakeOpenAI()
    fake.script = [httpx.ReadTimeout("slow")]
    c, _ = mk(fake=fake)
    with pytest.raises(P.ProviderError) as ei:
        c.chat_with_stats(MSG, "m")
    assert "не ответил" in str(ei.value)

    fake.script = [httpx.ConnectError("dns")]
    with pytest.raises(P.ProviderError) as ei:
        c.chat_with_stats(MSG, "m")
    assert "нет интернета" in str(ei.value) and "Облачный провайдер недоступен" in str(ei.value)

    local, fake2 = mk("ollama", key=None)
    fake2.script = [httpx.ConnectError("refused")]
    with pytest.raises(P.ProviderError) as ei:
        local.chat_with_stats(MSG, "m")
    assert "нет интернета" not in str(ei.value) and "11434" in str(ei.value)


def test_midstream_error_chunk():
    fake = FakeOpenAI()
    body = b'data: {"error": {"message": "upstream died"}}\n\n'
    fake.script = [httpx.Response(200, content=body)]
    c, _ = mk(fake=fake)
    with pytest.raises(P.ProviderError, match="upstream died"):
        c.chat_with_stats(MSG, "m")


def test_tiny_test_completion():
    c, fake = mk()
    r = c.test_completion("acme/think-1")
    assert r["ok"] and fake.last_chat()["body"]["max_tokens"] == 10 and "latency_s" in r
    fake.script = [httpx.Response(402, json={})]
    r = c.test_completion("acme/think-1")
    assert not r["ok"] and "средств" in r["message"]


# ----------------------------------------------------------- reasoning ---

def test_reasoning_mapping_per_kind():
    assert P.reasoning_params("lmstudio", "none") == {"reasoning_effort": "none"}
    assert P.reasoning_params("lmstudio", None) == {}
    assert P.reasoning_params("openrouter", "none") == {"reasoning": {"enabled": False}}
    assert P.reasoning_params("openrouter", "high") == {"reasoning": {"effort": "high", "exclude": True}}
    assert P.reasoning_params("openrouter", "high", supports=False) == {}
    for kind in ("ollama", "llamacpp", "openai_compatible"):
        assert P.reasoning_params(kind, "none") == {}                                    # unknown params: never by default
        assert P.reasoning_params(kind, "low", send_generic=True) == {"reasoning_effort": "low"}


def test_reasoning_in_requests():
    c, fake = mk("openai_compatible", url="https://api.example.com/v1")
    c.chat_with_stats(MSG, "m", reasoning_effort="none")
    assert "reasoning_effort" not in fake.last_chat()["body"] and "reasoning" not in fake.last_chat()["body"]
    c2, fake2 = mk("openai_compatible", url="https://api.example.com/v1", send_reasoning_effort=True)
    c2.chat_with_stats(MSG, "m", reasoning_effort="low")
    assert fake2.last_chat()["body"]["reasoning_effort"] == "low"
    c3, fake3 = mk("lmstudio", key=None)
    c3.chat_with_stats(MSG, "m", reasoning_effort="none")
    assert fake3.last_chat()["body"]["reasoning_effort"] == "none"
    c4, fake4 = mk()
    c4.chat_with_stats(MSG, "acme/think-1", reasoning_effort="medium")
    assert fake4.last_chat()["body"]["reasoning"] == {"effort": "medium", "exclude": True}


# --------------------------------------------------------------- models ---

def test_openrouter_models_parsing_and_cache(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(P, "_now", lambda: now[0])
    c, fake = mk()
    ms = c.list_models()
    by = {m["id"]: m for m in ms}
    assert "acme/embed-3" not in by  # embeddings are not chat models
    assert by["acme/think-1"]["price_in"] == 3.0 and by["acme/think-1"]["price_out"] == 15.0
    assert by["acme/think-1"]["supports_reasoning"] is True and by["acme/think-1"]["context_length"] == 200000
    assert by["meta/llama-free:free"]["free"] is True and by["meta/llama-free:free"]["supports_reasoning"] is False
    assert by["acme/think-1"]["free"] is False and by["acme/think-1"]["kind"] == "openrouter"
    assert all(k in by["acme/think-1"] for k in ("id", "name", "context_length", "loaded", "price_in", "price_out",
                                                   "free", "supports_reasoning", "kind"))
    c.list_models()
    assert fake.model_requests == 1          # cached
    now[0] += 601
    c.list_models()
    assert fake.model_requests == 2          # 10 min TTL expired
    c.list_models(refresh=True)
    assert fake.model_requests == 3


def test_filters():
    c, _ = mk()
    ms = c.list_models()
    assert [m["id"] for m in P.filter_models(ms, q="think")] == ["acme/think-1"]
    assert [m["id"] for m in P.filter_models(ms, free_only=True)] == ["meta/llama-free:free"]
    assert [m["id"] for m in P.filter_models(ms, min_context=100000)] == ["acme/think-1"]


def test_ollama_models_and_context():
    fake = FakeOpenAI(native={"/api/tags": {"models": [{"name": "llama3:8b"}, {"name": "nomic-embed-text"}]},
                              "/api/show": {"model_info": {"llama.context_length": 131072}}})
    c, _ = mk("ollama", key=None, fake=fake)
    assert [m["id"] for m in c.list_models()] == ["llama3:8b"]
    r = c.resolve_model("llama3:8b")
    assert r == {"model": "llama3:8b", "context": 8192, "note": None}   # Ollama serves a small window by default
    c2, _ = mk("ollama", key=None, fake=fake, context_length=32768)
    assert c2.resolve_model("llama3:8b")["context"] == 32768
    assert c.check_connection()["ok"]
    assert fake.requests[-1].url.path == "/api/tags"


def test_lmstudio_models_state_and_fallback():
    fake = FakeOpenAI(native={"/api/v0/models": {"data": [
        {"id": "gemma", "state": "loaded", "max_context_length": 131072, "loaded_context_length": 33000, "type": "llm"},
        {"id": "big", "state": "not-loaded", "max_context_length": 8192, "type": "llm"},
        {"id": "text-embedding-x", "state": "loaded", "type": "embeddings"}]}})
    c, _ = mk("lmstudio", key=None, fake=fake)
    ms = c.list_models()
    assert [m["id"] for m in ms] == ["gemma", "big"]
    g = ms[0]
    assert g["loaded"] is True and g["effective_context"] == 33000 and g["state"] == "loaded"
    assert ms[1]["loaded"] is False
    assert c.resolve_model("gemma") == {"model": "gemma", "context": 33000, "note": None}
    r = c.resolve_model("big")
    assert r["model"] == "gemma" and "«big»" in r["note"]
    # old LM Studio without /api/v0
    c2, _ = mk("lmstudio", key=None, fake=FakeOpenAI(models={"data": [{"id": "only"}]}, native={"/api/v0/models": httpx.Response(404)}))
    ms = c2.list_models()
    assert ms[0]["id"] == "only" and ms[0]["loaded"] is None
    assert c2.resolve_model("only")["model"] == "only"
    assert c.check_connection()["message"] == "LM Studio доступен"


def test_lmstudio_none_loaded_error():
    fake = FakeOpenAI(native={"/api/v0/models": {"data": [{"id": "a", "state": "not-loaded"}]}})
    c, _ = mk("lmstudio", key=None, fake=fake)
    with pytest.raises(P.ProviderError, match="не загружена"):
        c.resolve_model("a")


def test_openai_compatible_and_llamacpp():
    fake = FakeOpenAI(models={"data": [{"id": "qwen", "max_model_len": 32768}, {"id": "bge-embed"}]},
                      native={"/props": {"default_generation_settings": {"n_ctx": 4096}}})
    c, _ = mk("openai_compatible", url="http://localhost:8000", key=None, fake=fake)
    assert c.v1 == "http://localhost:8000/v1"
    assert [m["id"] for m in c.list_models()] == ["qwen"]
    assert c.resolve_model("qwen")["context"] == 32768
    assert fake.requests[-1].url.path == "/v1/models"
    c2, _ = mk("llamacpp", key=None, fake=fake)
    assert c2.resolve_model("anything")["context"] == 4096
    with pytest.raises(P.ProviderError, match="не выбрана модель"):
        c2.resolve_model("")


def test_check_connection_openrouter_verifies_key_and_reports_401():
    c, fake = mk()
    assert c.check_connection()["ok"] and fake.requests[-1].url.path.endswith("/key")
    fake.native["/api/v1/key"] = httpx.Response(401, json={"error": {"message": "no"}})
    r = c.check_connection()
    assert not r["ok"] and "ключ" in r["message"] and KEY not in r["message"]


# -------------------------------------------------------------- profiles ---

def test_is_cloud_derivation():
    assert P.is_cloud("openrouter", "https://openrouter.ai/api/v1")
    assert not P.is_cloud("lmstudio", "http://localhost:1234")
    assert not P.is_cloud("ollama", "http://192.168.1.5:11434") and not P.is_cloud("openai_compatible", "http://nas:8000")
    assert P.is_cloud("openai_compatible", "https://api.openai.com/v1")
    assert P.is_cloud("lmstudio", "http://203.0.113.9:1234")


def test_migration_of_old_settings():
    s = {"lmstudio_url": "http://localhost:4321", "llm_model": "old-model"}
    profs = P.get_profiles(s)
    assert len(profs) == 1 and profs[0]["id"] == P.LOCAL_ID and profs[0]["name"] == "LM Studio (локально)"
    assert profs[0]["kind"] == "lmstudio" and profs[0]["base_url"] == "http://localhost:4321"
    assert P.active_id(s) == P.LOCAL_ID
    prof = P.profile_for_task(s, "rag")
    assert P.model_for_task(s, prof, "rag") == "old-model"


def test_task_override_resolution():
    s = {"llm_providers": [P.migrated_profile({}),
                           {"id": "or", "name": "OR", "kind": "openrouter", "base_url": "https://openrouter.ai/api/v1",
                            "default_model": "acme/think-1"}],
         "active_llm_provider": P.LOCAL_ID, "llm_model": "local-m", "llm_provider_rag": "or", "llm_model_rewrite": "tiny"}
    assert P.profile_for_task(s, "rag")["id"] == "or" and P.model_for_task(s, P.profile_for_task(s, "rag"), "rag") == "acme/think-1"
    assert P.profile_for_task(s, "translation")["id"] == P.LOCAL_ID
    assert P.model_for_task(s, P.profile_for_task(s, "rewrite"), "rewrite") == "tiny"
    s["llm_provider_rag"] = "gone"
    with pytest.raises(P.ProviderError, match="удалён"):
        P.profile_for_task(s, "rag")


@pytest.mark.parametrize("bad", [
    {"name": "", "kind": "ollama", "base_url": "http://x"}, {"name": "a", "kind": "zzz", "base_url": "http://x"},
    {"name": "a", "kind": "ollama", "base_url": "ftp://x"}, {"name": "a", "kind": "ollama", "base_url": "http://u:p@x"},
    {"name": "a", "kind": "ollama", "base_url": "http://x", "timeout": -3},
    {"name": "a", "kind": "ollama", "base_url": "http://x", "extra_headers": {"Authorization": "x"}},
    {"name": "a", "kind": "ollama", "base_url": "http://x", "extra_headers": {"X-A": "a\nb"}},
])
def test_profile_validation(bad):
    with pytest.raises(ValueError):
        P.validate_profile(bad)


# --------------------------------------------------------------- secrets ---

@pytest.fixture
def sec(tmp_path, monkeypatch):
    from app import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    return tmp_path / "secrets.json"


@pytest.mark.skipif(sys.platform != "win32", reason="DPAPI is Windows-only")
def test_dpapi_roundtrip_and_not_plaintext(sec):
    secrets_store.set_key("or", KEY)
    raw = sec.read_text(encoding="utf-8")
    assert KEY not in raw and "SECRETSECRET" not in raw and '"dpapi"' in raw
    assert secrets_store.get_key("or") == KEY
    assert secrets_store.protection()["method"] == "dpapi"
    secrets_store.delete_key("or")
    assert secrets_store.get_key("or") is None


def test_plain_fallback_with_notice(sec, monkeypatch):
    monkeypatch.setattr(secrets_store, "_force_plain", True)
    secrets_store.set_key("or", KEY)
    assert secrets_store.get_key("or") == KEY and '"plain"' in sec.read_text(encoding="utf-8")
    pr = secrets_store.protection()
    assert pr["method"] == "plain" and pr["notice"]


def test_key_info_never_reveals_key_and_env_fallback(sec, monkeypatch):
    assert secrets_store.key_info("or", "openrouter") == {"has_key": False, "last4": None, "key_source": "none"}
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key-0000-WXYZ")
    info = secrets_store.key_info("or", "openrouter", "https://openrouter.ai/api/v1")
    assert info == {"has_key": True, "last4": "WXYZ", "key_source": "env"}
    assert secrets_store.get_key("or", "openrouter", "https://openrouter.ai/api/v1") == "env-key-0000-WXYZ"
    secrets_store.set_key("or", KEY)
    assert secrets_store.key_info("or", "openrouter") == {"has_key": True, "last4": KEY[-4:], "key_source": "stored"}
    # an env key is never sent to some other server
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-0000-ABCD")
    assert secrets_store.get_key("x", "openai_compatible", "https://evil.example/v1") is None
    assert secrets_store.get_key("x", "openai_compatible", "https://api.openai.com/v1") == "sk-openai-0000-ABCD"
    assert secrets_store.get_key("x", "ollama", "http://localhost:11434") is None


def test_set_key_rejects_garbage(sec):
    for bad in ("", "  ", "a b", "ключ"):
        with pytest.raises(ValueError):
            secrets_store.set_key("or", bad)


def test_redact():
    assert KEY not in secrets_store.redact(f"x {KEY} y", KEY)
    assert "sk-" not in secrets_store.redact("token sk-abcdefghijk123 here")
    assert "Bearer" not in secrets_store.redact("Authorization: Bearer abcdefghijkl")


def test_build_dist_guard(tmp_path):
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "scripts"))
    import build_dist
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "titles").mkdir()
    (tmp_path / "data" / "titles" / "ok.json").write_text("{}")
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "secrets_store.py").write_text("# code is fine")
    assert build_dist.find_secret_files(tmp_path) == []
    (tmp_path / "data" / "secrets.json").write_text("{}")
    (tmp_path / "data" / "settings.json").write_text("{}")
    names = sorted(f.name for f in build_dist.find_secret_files(tmp_path))
    assert names == ["secrets.json", "settings.json"]
