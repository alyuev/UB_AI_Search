"""«Параметры ответа»: GET /api/options, POST /api/settings validation, applying without restart."""
import json

import pytest

from app import config
from tests.server_helpers import run_rag


def test_options_endpoint_lists_every_option_with_value_and_spec(api):
    d = api.get("/api/options").json()
    keys = {o["key"] for o in d["options"]}
    assert {"rag_top_k", "translation_batch_size", "reasoning_rag", "rag_rrf_k"} <= keys
    assert {g["id"] for g in d["groups"]} >= {"rag_sources", "translation"}
    top_k = next(o for o in d["options"] if o["key"] == "rag_top_k")
    assert top_k["value"] == 12 and top_k["min"] == 1 and top_k["max"] == 40 and top_k["label"] and top_k["hint"]


def test_options_endpoint_reflects_saved_values(api):
    assert api.post("/api/settings", json={"rag_top_k": 5}).status_code == 200
    d = api.get("/api/options").json()
    assert next(o for o in d["options"] if o["key"] == "rag_top_k")["value"] == 5


@pytest.mark.parametrize("payload,fragment", [
    ({"rag_top_k": 0}, "допустимо от 1 до 40"),
    ({"rag_top_k": 41}, "допустимо от 1 до 40"),
    ({"rag_top_k": "много"}, "ожидается число"),
    ({"rag_top_k": 2.5}, "целое"),
    ({"rag_top_k": True}, "число"),
    ({"rag_context_share": 0.01}, "допустимо"),
    ({"rag_neighbors": 3}, "допустимо от 0 до 2"),
    ({"rag_query_rewrite": "возможно"}, "«вкл» или «выкл»"),
    ({"rag_sources_lang": "de"}, "допустимо одно из"),
    ({"reasoning_rag": "maybe"}, "допустимо одно из"),
    ({"translation_min_severity": "huge"}, "допустимо одно из"),
    ({"rag_min_relevance": float("nan")}, None),  # JSON null/NaN is never silently accepted
])
def test_invalid_option_values_are_rejected_with_a_russian_message(api, payload, fragment):
    before = config.load_settings()
    r = api.post("/api/settings", content=json.dumps(payload, allow_nan=True),
                 headers={"Content-Type": "application/json"})
    assert r.status_code == 400, r.text
    if fragment:
        assert fragment in r.json()["detail"]
    assert config.load_settings() == before  # nothing was saved


def test_one_bad_value_blocks_the_whole_save(api):
    r = api.post("/api/settings", json={"temperature": 0.5, "rag_top_k": 999})
    assert r.status_code == 400
    assert config.load_settings()["temperature"] != 0.5


def test_unknown_keys_are_rejected(api):
    r = api.post("/api/settings", json={"rag_top_kk": 5})
    assert r.status_code == 400 and "Неизвестная настройка" in r.json()["detail"] and "rag_top_kk" in r.json()["detail"]
    assert "rag_top_kk" not in config.load_settings()


@pytest.mark.parametrize("payload", [
    {"temperature": "тёплая"}, {"temperature": 5}, {"temperature": None}, {"max_tokens": 0}, {"max_tokens": 10.5},
    {"llm_timeout": -1}, {"lmstudio_url": 123}, {"fts_stopwords": "yes"},
    {"fts_mode_hybrid": "xor"}, {"fts_weights": [1]}, {"fts_weights": {"zz": 1}}, {"fts_weights": {"en": "a"}},
    {"fts_weights": {"en": -1}}, {"prompts": ["x"]},
])
def test_plain_settings_are_validated_too(api, payload):
    r = api.post("/api/settings", json=payload)
    assert r.status_code == 400 and r.json()["detail"]


def test_valid_save_is_returned_persisted_and_applied_without_restart(api):
    r = api.post("/api/settings", json={"rag_top_k": 3, "rag_neighbors": 1, "temperature": 0.7, "reasoning_rag": "on",
                                        "rag_query_rewrite": False})
    assert r.status_code == 200
    body = r.json()
    assert body["rag_top_k"] == 3 and body["temperature"] == 0.7
    assert json.load(open(config.SETTINGS_PATH, encoding="utf-8"))["rag_top_k"] == 3
    # the very next question uses the new values (no restart, no new client)
    job = run_rag(api)
    assert job["status"] == "done", job
    exp = job["result"]["experiment"]
    assert exp["options"]["rag_top_k"] == 3 and exp["options"]["rag_neighbors"] == 1
    assert exp["retrieval"]["k"] == 3 and exp["temperature"] == 0.7
    assert exp["reasoning"]["setting"] == "on" and exp["reasoning"]["effort"] is None
    assert api.llm.of("plan") == []  # query preparation switched off


def test_string_forms_of_numbers_and_bools_are_normalised(api):
    body = api.post("/api/settings", json={"rag_top_k": "7", "rag_query_rewrite": "false"}).json()
    assert body["rag_top_k"] == 7 and body["rag_query_rewrite"] is False


def test_partial_save_keeps_other_settings(api):
    api.post("/api/settings", json={"rag_top_k": 9})
    api.post("/api/settings", json={"rag_rrf_k": 30})
    s = api.get("/api/settings").json()
    assert s["rag_top_k"] == 9 and s["rag_rrf_k"] == 30


def test_fts_settings_apply_to_searcher_without_restart(api):
    old = api.server.state.searcher
    r = api.post("/api/settings", json={"fts_mode_hybrid": "and", "fts_weights": {"en": 3}})
    assert r.status_code == 200
    s = api.server.state.searcher
    assert s is not old and s.fts_mode_hybrid == "and"
    assert r.json()["fts_weights"]["en"] == 3 and r.json()["fts_weights"]["ru"] == 1.0  # merged, not replaced


def test_rag_without_k_uses_the_option_and_explicit_k_wins(api):
    """The UI will stop sending k: 12 -- the «сколько абзацев» setting must then be honoured."""
    api.post("/api/settings", json={"rag_top_k": 2, "rag_query_rewrite": False})
    assert run_rag(api)["result"]["experiment"]["retrieval"]["k"] == 2
    assert run_rag(api, k=4)["result"]["experiment"]["retrieval"]["k"] == 4
    api.post("/api/settings", json={"rag_top_k": 6})
    assert run_rag(api)["result"]["experiment"]["retrieval"]["k"] == 6
    h = api.get("/api/history", params={"mode": "rag"}).json()["items"]
    assert h[0]["params"]["k"] == 6 and h[1]["params"]["k"] == 4  # history records the k really used
