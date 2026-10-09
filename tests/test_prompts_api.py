"""Prompts and prompt presets over HTTP: edit, reset, placeholder validation, presets, import/export,
and what the RAG / translation jobs do with a broken override."""
import json
import urllib.parse

import pytest

from app import config
from app import prompts as P
from tests.server_helpers import run_rag, run_translation

SYS_DEFAULT = P.specs()["rag_system"]["default"]
TMPL = "ВОПРОС: {question}\nИСТОЧНИКИ:\n{sources}"


def put(api, pid, text):
    return api.put(f"/api/prompts/{pid}", json={"text": text})


def by_id(api, pid):
    return next(p for p in api.get("/api/prompts").json()["prompts"] if p["id"] == pid)


def q(name):
    return urllib.parse.quote(name, safe="")


# --------------------------------------------------------------------------- list / edit / reset ---

def test_list_has_all_prompts_with_defaults_and_versions(api):
    d = api.get("/api/prompts").json()
    ids = [p["id"] for p in d["prompts"]]
    assert set(ids) == set(P.specs()) and len(ids) == 10
    p = by_id(api, "rag_system")
    assert p["custom"] is False and p["text"] == p["default"] == SYS_DEFAULT
    assert p["info"]["label"].startswith("стандартный · ") and p["default_info"]["tokens"] > 0
    assert d["warnings"] == [] and d["preset"] is None


def test_edit_marks_prompt_custom_and_persists_only_the_difference(api):
    r = put(api, "rag_system", "Отвечай коротко.")
    assert r.status_code == 200 and r.json()["custom"] is True and r.json()["info"]["label"].startswith("свой · ")
    assert config.load_settings()["prompts"] == {"rag_system": "Отвечай коротко."}
    assert by_id(api, "rag_system")["text"] == "Отвечай коротко."


def test_saving_the_default_or_empty_text_removes_the_override(api):
    put(api, "rag_system", "Своё.")
    assert put(api, "rag_system", SYS_DEFAULT).json()["custom"] is False
    assert config.load_settings()["prompts"] == {}
    put(api, "rag_system", "Своё.")
    assert put(api, "rag_system", "   ").json()["custom"] is False
    assert config.load_settings()["prompts"] == {}


def test_unknown_prompt_is_404(api):
    assert put(api, "nope", "x").status_code == 404
    assert api.post("/api/prompts/nope/reset").status_code == 404


@pytest.mark.parametrize("pid,text,fragment", [
    ("rag_user_template", "Вопрос: {question}", "не хватает обязательных подстановок: {sources}"),
    ("rag_user_template", "{sources} {question} {mystery}", "неизвестные подстановки: {mystery}"),
    ("rag_user_template", "", None),  # empty = reset to default, not an error (checked below)
    ("rag_system", "Используй {sources}", "неизвестные подстановки: {sources}"),
    pytest.param("rag_system", "x" * 60001, "слишком длинный", id="too-long"),
    ("translation_verify_user", "{en} {ru}", "{problem}"),
])
def test_placeholder_validation_on_save(api, pid, text, fragment):
    r = put(api, pid, text)
    if fragment is None:
        assert r.status_code == 400 or r.json()["custom"] is False
        return
    assert r.status_code == 400
    assert fragment in r.json()["detail"] and "не сохранён" in r.json()["detail"]
    assert config.load_settings().get("prompts", {}) == {}


def test_json_braces_in_a_prompt_are_not_placeholders(api):
    assert put(api, "query_rewrite", 'Верни {"a": 1} и до {max_terms} слов').status_code == 200


def test_reset_one_and_reset_all(api):
    put(api, "rag_system", "A")
    put(api, "rag_formatting", "B")
    assert api.post("/api/prompts/rag_system/reset").json()["custom"] is False
    assert config.load_settings()["prompts"] == {"rag_formatting": "B"}
    api.post("/api/prompt-presets", json={"name": "мой"})  # activates a preset
    d = api.post("/api/prompts/reset-all").json()
    assert not any(p["custom"] for p in d["prompts"]) and d["preset"] is None
    s = config.load_settings()
    assert s["prompts"] == {} and s["prompt_preset"] is None


def test_settings_endpoint_validates_prompts_too(api):
    r = api.post("/api/settings", json={"prompts": {"rag_user_template": "нет подстановок"}})
    assert r.status_code == 400 and "rag_user_template" in r.json()["detail"]
    r = api.post("/api/settings", json={"prompts": {"ghost": "x"}})
    assert r.status_code == 400 and "неизвестный промт" in r.json()["detail"]


# ------------------------------------------------------------------ prompts in the RAG / translation flow ---

def test_custom_prompts_reach_the_model(api):
    put(api, "rag_user_template", TMPL)
    put(api, "rag_formatting", "Пиши простым текстом без разметки.")
    job = run_rag(api)
    assert job["status"] == "done"
    sys_msg, user_msg = api.llm.of("answer")[0]["messages"]
    assert sys_msg["content"].endswith("Пиши простым текстом без разметки.")
    assert user_msg["content"].startswith("ВОПРОС: Что говорил Иисус о любви?\nИСТОЧНИКИ:\n[")
    exp = job["result"]["experiment"]
    assert exp["prompt_versions"]["rag_user_template"]["source"] == "custom"
    assert exp["prompt_versions"]["rag_system"]["source"] == "default"


def test_invalid_override_in_rag_falls_back_to_default_and_warns_visibly(api):
    # bypass the API validation: a hand-edited settings.json (or a prompt from an older version)
    s = config.load_settings()
    config.save_settings({**s, "prompts": {"rag_user_template": "только вопрос {question}",
                                           "rag_system": "Рабочий свой вариант."}})
    job = run_rag(api)
    assert job["status"] == "done", job
    warns = job["result"]["warnings"]
    assert len(warns) == 1 and "не применён" in warns[0] and "{sources}" in warns[0] and "Использован стандартный" in warns[0]
    sys_msg, user_msg = api.llm.of("answer")[0]["messages"]
    assert sys_msg["content"].startswith("Рабочий свой вариант.")  # the valid override still applies
    assert "Sources / Источники:" in user_msg["content"]  # default template instead of the broken one
    assert job["result"]["prompt_warnings"] == warns
    assert any("не применён" in line for line in job["log"]) if "log" in job else True
    # and the warning is kept with the history entry
    h = api.get("/api/history", params={"mode": "rag"}).json()["items"][0]
    assert api.get(f"/api/history/{h['id']}").json()["result"]["warnings"] == warns


def test_invalid_override_in_translation_warns_visibly(api):
    config.save_settings({**config.load_settings(), "prompts": {"translation_verify_user": "{en} без остального"}})
    job = run_translation(api)
    assert job["status"] == "done", job
    assert len(job["result"]["warnings"]) == 1 and "не применён" in job["result"]["warnings"][0]


def test_clean_run_has_empty_warnings(api):
    assert run_rag(api)["result"]["warnings"] == []
    assert run_translation(api)["result"]["warnings"] == []


# ---------------------------------------------------------------------------------------- presets ---

def test_preset_save_list_load_dirty_flag(api):
    put(api, "rag_system", "Версия A")
    r = api.post("/api/prompt-presets", json={"name": "  Набор А  ", "note": "заметка"}).json()
    assert r["active"] == "Набор А"
    row = r["presets"][0]
    assert row["name"] == "Набор А" and row["count"] == 1 and row["note"] == "заметка" and row["active"] and not row["dirty"]
    put(api, "rag_system", "Версия B")
    assert api.get("/api/prompt-presets").json()["presets"][0]["dirty"] is True
    assert api.get("/api/prompts").json()["preset_dirty"] is True
    d = api.post(f"/api/prompt-presets/{q('Набор А')}/load").json()
    assert next(p for p in d["prompts"] if p["id"] == "rag_system")["text"] == "Версия A"
    assert d["preset"] == "Набор А" and d["preset_dirty"] is False and d["load_warnings"] == []


def test_save_over_existing_name_replaces_it(api):
    put(api, "rag_system", "1")
    api.post("/api/prompt-presets", json={"name": "x"})
    put(api, "rag_system", "2")
    rows = api.post("/api/prompt-presets", json={"name": "x"}).json()["presets"]
    assert len(rows) == 1
    api.post("/api/prompt-presets/x/load")
    assert by_id(api, "rag_system")["text"] == "2"


@pytest.mark.parametrize("name", ["", "   ", "a/b", "a\\b", "я" * 81])
def test_bad_preset_names_rejected(api, name):
    assert api.post("/api/prompt-presets", json={"name": name}).status_code == 400


def test_load_unknown_preset_is_404(api):
    assert api.post("/api/prompt-presets/нет/load").status_code == 404


def test_load_preset_with_a_prompt_that_became_invalid_reports_it(api):
    P.save_preset("старый", {"rag_user_template": "без подстановок", "rag_formatting": "ок"})
    d = api.post(f"/api/prompt-presets/{q('старый')}/load").json()
    assert d["load_warnings"] and "не применён" in d["load_warnings"][0]
    assert config.load_settings()["prompts"] == {"rag_formatting": "ок"}


def test_rename_preset_updates_active_name_and_rejects_collisions(api):
    put(api, "rag_system", "z")
    api.post("/api/prompt-presets", json={"name": "один"})
    api.post("/api/prompt-presets", json={"name": "два"})
    assert api.put("/api/prompt-presets/один", json={"new_name": "два"}).status_code == 400  # collision
    assert api.put("/api/prompt-presets/нет", json={"new_name": "x"}).status_code == 404
    assert api.put("/api/prompt-presets/один", json={"new_name": ""}).status_code == 400
    r = api.put("/api/prompt-presets/два", json={"new_name": "три"}).json()  # «два» is the active one
    assert r["active"] == "три" and [p["name"] for p in r["presets"]] == ["один", "три"]


def test_delete_preset_clears_active_marker(api):
    api.post("/api/prompt-presets", json={"name": "a"})
    api.post("/api/prompt-presets", json={"name": "b"})  # b is active
    r = api.delete("/api/prompt-presets/b").json()
    assert r["active"] is None and [p["name"] for p in r["presets"]] == ["a"]
    assert api.delete("/api/prompt-presets/b").status_code == 404


def test_export_all_and_one(api):
    put(api, "rag_system", "E")
    api.post("/api/prompt-presets", json={"name": "e1"})
    api.post("/api/prompt-presets", json={"name": "e2"})
    r = api.get("/api/prompt-presets-export")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    data = r.json()
    assert data["format"] == "smartsearch-prompt-presets" and set(data["presets"]) == {"e1", "e2"}
    one = api.get("/api/prompt-presets-export", params={"name": "e2"}).json()
    assert set(one["presets"]) == {"e2"} and one["presets"]["e2"]["prompts"] == {"rag_system": "E"}
    assert api.get("/api/prompt-presets-export", params={"name": "нет"}).status_code == 404


def test_import_roundtrip_and_name_collision_never_overwrites(api):
    put(api, "rag_system", "orig")
    api.post("/api/prompt-presets", json={"name": "p"})
    exported = api.get("/api/prompt-presets-export").json()
    exported["presets"]["p"]["prompts"]["rag_system"] = "от другого человека"
    r = api.post("/api/prompt-presets-import", json={"data": exported}).json()
    assert r["imported"] == ["p (2)"] and r["problems"] == []
    again = api.post("/api/prompt-presets-import", json={"data": exported}).json()
    assert again["imported"] == ["p (3)"]
    assert P.get_preset("p")["prompts"]["rag_system"] == "orig"
    assert P.get_preset("p (2)")["prompts"]["rag_system"] == "от другого человека"
    assert [row["name"] for row in r["presets"]] == ["p", "p (2)"]


@pytest.mark.parametrize("bad", [
    {}, {"format": "other", "presets": {}}, {"format": "smartsearch-prompt-presets"},
    {"format": "smartsearch-prompt-presets", "presets": []},
    {"format": "smartsearch-prompt-presets", "presets": {"a/b": {"prompts": {}}}},
])
def test_malformed_import_is_400_and_changes_nothing(api, bad):
    r = api.post("/api/prompt-presets-import", json={"data": bad})
    assert r.status_code == 400 and r.json()["detail"]
    assert P.load_presets() == {}


def test_import_drops_invalid_prompts_and_bad_entries_but_keeps_the_rest(api):
    data = {"format": "smartsearch-prompt-presets", "version": 1, "presets": {
        "ok": {"prompts": {"rag_formatting": "норм"}},
        "mixed": {"prompts": {"rag_user_template": "битый", "rag_system": "хороший", "ghost": "?"}},
        "junk": "строка вместо объекта",
        "noprompts": {"prompts": ["x"]},
    }}
    r = api.post("/api/prompt-presets-import", json={"data": data}).json()
    assert r["imported"] == ["ok", "mixed"]
    assert len(r["problems"]) == 4  # two bad prompts in «mixed» + two bad entries
    assert P.get_preset("mixed")["prompts"] == {"rag_system": "хороший"}


def test_import_endpoint_rejects_non_object_body(api):
    assert api.post("/api/prompt-presets-import", json={"data": "text"}).status_code == 422


def test_deleted_preset_does_not_break_prompts_listing(api):
    api.post("/api/prompt-presets", json={"name": "t"})
    P.delete_preset("t")  # removed behind the settings' back
    d = api.get("/api/prompts").json()
    assert d["preset"] == "t" and d["preset_dirty"] is None
