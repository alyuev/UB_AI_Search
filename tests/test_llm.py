
import pytest

from app.llm.lmstudio import (
    LMStudioClient, ConnectionRefused, strip_reasoning,
)


class FakeResponse:
    def __init__(self, json_data, status=200):
        self._json = json_data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("http error")

    def json(self):
        return self._json


def test_strip_reasoning_basic():
    text = "<think>internal chatter</think>Hello world"
    assert strip_reasoning(text) == "Hello world"


def test_strip_reasoning_keeps_when_no_think():
    assert strip_reasoning("plain text") == "plain text"


def test_strip_reasoning_keep_tuple():
    visible, reasoning = strip_reasoning("<think>abc</think>visible", keep=True)
    assert visible == "visible"
    assert reasoning == "abc"


def test_list_models_prefers_api_v0(monkeypatch):
    client = LMStudioClient("http://localhost:1234")

    def fake_get(self, path, timeout=None):
        if path == "/api/v0/models":
            return {"data": [
                {"id": "llama-3-8b", "state": "loaded", "max_context_length": 8192, "type": "llm"},
                {"id": "text-embedding-e5", "state": "not-loaded", "type": "embeddings"},
            ]}
        raise AssertionError("should not fall back")

    monkeypatch.setattr(LMStudioClient, "_get", fake_get)
    models = client.list_models()
    assert len(models) == 1
    assert models[0]["id"] == "llama-3-8b"
    assert models[0]["state"] == "loaded"


def test_list_models_falls_back_to_v1(monkeypatch):
    client = LMStudioClient("http://localhost:1234")
    calls = []

    def fake_get(self, path, timeout=None):
        calls.append(path)
        if path == "/api/v0/models":
            from app.llm.lmstudio import LMStudioError
            raise LMStudioError("no v0")
        if path == "/v1/models":
            return {"data": [{"id": "some-model"}]}
        raise AssertionError(path)

    monkeypatch.setattr(LMStudioClient, "_get", fake_get)
    models = client.list_models()
    assert models[0]["id"] == "some-model"
    assert "/v1/models" in calls


def test_connection_refused_message():
    exc = ConnectionRefused("http://localhost:1234")
    assert "1234" in str(exc)


def test_chat_no_model_raises():
    client = LMStudioClient("http://localhost:1234")
    from app.llm.lmstudio import NoModelLoaded
    with pytest.raises(NoModelLoaded):
        client.chat([{"role": "user", "content": "hi"}], model=None)


def _fake_models(monkeypatch, models):
    monkeypatch.setattr(LMStudioClient, "list_models", lambda self, include_embeddings=False: models)


def test_resolve_model_configured_loaded(monkeypatch):
    _fake_models(monkeypatch, [
        {"id": "a", "state": "loaded", "effective_context": 33000},
        {"id": "b", "state": "not-loaded", "effective_context": 131072},
    ])
    r = LMStudioClient("http://x").resolve_model("a")
    assert r == {"model": "a", "context": 33000, "note": None}


def test_resolve_model_falls_back_to_loaded_with_note(monkeypatch):
    _fake_models(monkeypatch, [
        {"id": "a", "state": "not-loaded", "effective_context": 131072},
        {"id": "b", "state": "loaded", "effective_context": 33024},
    ])
    r = LMStudioClient("http://x").resolve_model("a")
    assert r["model"] == "b" and r["context"] == 33024
    assert "«a»" in r["note"] and "«b»" in r["note"]


def test_resolve_model_none_loaded_raises_with_hint(monkeypatch):
    from app.llm.lmstudio import NoModelLoaded
    _fake_models(monkeypatch, [{"id": "a", "state": "not-loaded", "effective_context": 1}])
    with pytest.raises(NoModelLoaded) as ei:
        LMStudioClient("http://x").resolve_model("a")
    assert "«a»" in str(ei.value)
