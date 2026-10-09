import json

import pytest

from app import options as O
from app import prompts as P


# ------------------------------------------------------------------ options ---
def test_option_defaults_keep_todays_behaviour():
    d = O.DEFAULTS
    assert d["rag_top_k"] == 12 and d["rag_rrf_k"] == 60 and d["rag_context_share"] == 1.0
    assert d["rag_neighbors"] == 0 and d["rag_min_relevance"] == 0.0
    assert d["rag_sources_lang"] == "both" and d["rag_answer_lang"] == "question"
    assert d["rag_query_rewrite"] is True and d["rag_strictness"] == "allow_inferences"
    assert d["translation_batch_size"] == 8 and d["translation_verify"] is True
    assert d["translation_min_severity"] == "minor" and d["rag_rewrite_max_terms"] == 6


@pytest.mark.parametrize("key,bad", [
    ("rag_top_k", 0), ("rag_top_k", 41), ("rag_top_k", "abc"), ("rag_top_k", 3.5), ("rag_context_share", 0.01),
    ("rag_neighbors", 3), ("rag_sources_lang", "de"), ("rag_query_rewrite", "maybe"),
    ("translation_min_severity", "huge"), ("rag_min_relevance", 1.5), ("nope", 1), ("rag_top_k", float("nan")),
])
def test_option_validation_rejects(key, bad):
    with pytest.raises(ValueError):
        O.coerce(key, bad)


def test_option_coercion_and_effective_fallback():
    assert O.coerce("rag_top_k", "7") == 7 and isinstance(O.coerce("rag_top_k", 7.0), int)
    assert O.coerce("rag_context_share", 1) == 1.0
    assert O.coerce("rag_query_rewrite", "false") is False
    eff = O.effective({"rag_top_k": 5, "rag_neighbors": 99, "rag_sources_lang": "ru"})
    assert eff["rag_top_k"] == 5 and eff["rag_sources_lang"] == "ru"
    assert eff["rag_neighbors"] == 0  # invalid stored value silently falls back to the default


def test_public_spec_is_complete():
    spec = O.public_spec({"rag_top_k": 9})
    keys = {o["key"] for o in spec["options"]}
    assert keys == set(O.DEFAULTS) and {g["id"] for g in spec["groups"]} >= {o["group"] for o in spec["options"]}
    top = next(o for o in spec["options"] if o["key"] == "rag_top_k")
    assert top["value"] == 9 and top["min"] == 1 and top["max"] == 40 and top["hint"] and top["label"]


# ------------------------------------------------------------------ prompts ---
def test_every_default_prompt_validates_and_is_described():
    assert len(P.specs()) == 10
    for pid, sp in P.specs().items():
        assert P.validate(pid, sp["default"]) is None, pid
        assert sp["title"] and sp["description"] and sp["group"] in ("rag", "translation")


def test_validation_messages():
    assert "неизвестные подстановки: {foo}" in P.validate("rag_system", "hello {foo}")
    msg = P.validate("rag_user_template", "only {question}")
    assert "не хватает" in msg and "{sources}" in msg
    assert P.validate("rag_user_template", "{sources} {question}") is None
    assert P.validate("rag_user_template", "{sources} {question} {inference_reminder}") is None
    assert "неизвестный промт" in P.validate("zzz", "x")
    assert P.validate("rag_system", "x" * 70000)
    # literal JSON braces are NOT placeholders
    assert P.validate("rag_system", 'answer as {"a": 1} please') is None
    assert P.validate("query_rewrite", "limit {max_terms}") is None


def test_render_only_replaces_known_placeholders():
    out = P.render('JSON {"k": 1} {question} {unknown}', question="Q")
    assert out == 'JSON {"k": 1} Q {unknown}'


def test_resolved_prompts_overrides_warnings_and_versions():
    rp = P.ResolvedPrompts({"rag_system": "my system prompt", "rag_user_template": "no placeholders at all",
                            "bogus": "x"}, preset="exp1")
    assert rp.text("rag_system") == "my system prompt" and rp.is_custom("rag_system")
    assert rp.text("rag_user_template") == P.specs()["rag_user_template"]["default"]  # invalid -> default
    assert len(rp.warnings) == 2 and any("Использован стандартный" in w for w in rp.warnings)
    i = rp.info("rag_system")
    assert i["source"] == "custom" and i["label"].startswith("свой · ") and i["chars"] == 16
    d = rp.info("rag_formatting")
    assert d["source"] == "default" and d["label"].startswith("стандартный · ")
    assert rp.versions(["rag_system"])["rag_system"]["hash"] == P.text_hash("my system prompt")
    # an override equal to the default is not an override
    assert not P.ResolvedPrompts({"rag_system": P.specs()["rag_system"]["default"]}).is_custom("rag_system")
    # replay mode keeps exact snapshot texts; the «стандартный/свой» label follows the saved version info
    dflt = P.specs()["rag_system"]["default"]
    rp = P.ResolvedPrompts({"rag_system": dflt}, replay=True)
    assert rp.text("rag_system") == dflt and not rp.is_custom("rag_system")
    rp = P.ResolvedPrompts({"rag_system": dflt}, replay=True, sources={"rag_system": {"source": "custom"}})
    assert rp.is_custom("rag_system") and rp.info("rag_system")["source"] == "custom"
    rp = P.ResolvedPrompts({"rag_system": "old default"}, replay=True, sources={"rag_system": {"source": "default"}})
    assert rp.text("rag_system") == "old default" and not rp.is_custom("rag_system")


# ------------------------------------------------------------------ presets ---
@pytest.fixture
def clean_presets():
    for n in list(P.load_presets()):
        P.delete_preset(n)
    yield
    for n in list(P.load_presets()):
        P.delete_preset(n)


def test_preset_save_load_rename_delete(clean_presets):
    P.save_preset("опыт 1", {"rag_system": "A"}, note="n")
    P.save_preset("опыт 2", {"rag_system": "B"})
    assert set(P.load_presets()) == {"опыт 1", "опыт 2"}
    assert P.get_preset("опыт 1")["prompts"] == {"rag_system": "A"}
    assert P.rename_preset("опыт 1", "первый") == "первый"
    with pytest.raises(ValueError):
        P.rename_preset("первый", "опыт 2")
    with pytest.raises(KeyError):
        P.rename_preset("нет", "x")
    with pytest.raises(ValueError):
        P.save_preset("   ", {})
    with pytest.raises(ValueError):
        P.save_preset("a/b", {})
    assert P.delete_preset("первый") is True and P.delete_preset("первый") is False


def test_preset_export_import_roundtrip_and_hardening(clean_presets):
    P.save_preset("exp", {"rag_system": "A", "rag_formatting": "plain"})
    exported = json.loads(json.dumps(P.export_json()))
    assert exported["format"] == "smartsearch-prompt-presets" and "exp" in exported["presets"]
    res = P.import_json(exported)
    assert res["imported"] == ["exp (2)"]  # never overwrites an existing preset
    bad = {"format": "smartsearch-prompt-presets", "version": 1,
           "presets": {"x": {"prompts": {"rag_user_template": "no placeholders", "rag_system": "ok"}}, "y": 5}}
    res = P.import_json(bad)
    assert res["imported"] == ["x"] and len(res["problems"]) == 2
    assert P.get_preset("x")["prompts"] == {"rag_system": "ok"}
    with pytest.raises(ValueError):
        P.import_json({"hello": 1})
    with pytest.raises(ValueError):
        P.import_json([])
