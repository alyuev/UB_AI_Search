import json

import pytest

from app.llm.lmstudio import ConnectionRefused
from app.llm.rag import (
    RagPipeline, extract_citations, fts_query_from_plan, parse_search_plan, plan_summary,
    rewrite_query, rrf_merge,
)


def P(id_, text="t"):
    return {"id": id_, "paper": 0, "section": 1, "paragraph": 1, "en": text, "ru": text}


class FakeStats:
    def to_dict(self):
        return {"model": "m", "elapsed_s": 1.0}

    def summary_line(self):
        return "m · 10 ток/с"


class FakeLLM:
    def __init__(self, plan_json=None, plan_exc=None, answer="Ответ [0:1.2].", answer_exc=None):
        self.plan_json, self.plan_exc, self.answer, self.answer_exc = plan_json, plan_exc, answer, answer_exc
        self.chat_calls, self.gen_calls = [], []

    def chat(self, messages, model, temperature, max_tokens, timeout, reasoning_effort=None, **kw):
        self.chat_calls.append(dict(messages=messages, temperature=temperature, max_tokens=max_tokens,
                                    reasoning_effort=reasoning_effort))
        if self.plan_exc:
            raise self.plan_exc
        return self.plan_json

    def chat_with_stats(self, messages, model, temperature, max_tokens, timeout, reasoning_effort=None,
                        on_delta=None):
        self.gen_calls.append(messages)
        if self.answer_exc:
            raise self.answer_exc
        return self.answer, FakeStats()


class FakeSearcher:
    fts_mode_hybrid = "or"

    def __init__(self):
        self.calls = []

    def fts(self, q, k=20, lang="both", mode=None):
        self.calls.append(("fts", q, mode))
        return [P("0:1.2"), P("0:1.3")]

    def semantic(self, q, k=20):
        self.calls.append(("semantic", q))
        return [P("0:2.10"), P("0:1.2")] if "plan" in q else [P("0:5.1"), P("0:1.2")]

    def hybrid(self, q, k=20, **kw):
        self.calls.append(("hybrid", q)); self.hybrid_kw = kw
        return [P("0:9.9")]


PLAN = json.dumps({"keywords_ru": ["божество"], "keywords_en": ["deity", "Deity"],
                   "semantic_query": "plan: определение божества"})


# ------------------------------------------------------------------ helpers ---
def test_parse_search_plan_variants():
    p = parse_search_plan('Sure!\n```json\n' + PLAN + '\n```')
    assert p["keywords_ru"] == ["божество"] and p["keywords_en"] == ["deity"]  # case-insensitive dedupe
    assert p["semantic_query"].startswith("plan")
    assert parse_search_plan("no json here") is None
    assert parse_search_plan('{"keywords_ru": [], "keywords_en": []}') is None
    assert parse_search_plan('{"keywords_ru": "oops"}') is None
    assert parse_search_plan("") is None
    assert parse_search_plan('{"keywords_ru":["а"],"semantic_query":null}')["semantic_query"] == ""


def test_fts_query_and_summary():
    plan = {"keywords_ru": ["агондонтеры", "высшее существо"], "keywords_en": ["agondonters"], "semantic_query": ""}
    assert fts_query_from_plan(plan) == 'агондонтеры "высшее существо" agondonters'
    assert plan_summary(plan) == "Ищу: агондонтеры, высшее существо / agondonters"
    assert plan_summary({"keywords_ru": ["а"], "keywords_en": [], "semantic_query": ""}) == "Ищу: а"


def test_rrf_merge_prefers_items_found_by_several_lists():
    merged = rrf_merge([[P("a"), P("b")], [P("c"), P("b")], [P("b")]], k=3)
    assert [m["id"] for m in merged][0] == "b"
    assert merged[0]["fts_rank"] == 2 and merged[0]["sem_rank"] == 1
    assert len(rrf_merge([[P("a"), P("b")]], k=1)) == 1


def test_extract_citations_order_groups_and_dedupe():
    t = "A [0:1.2][0:2.10]. B [0:1.2, 5:3.4]; C [нет] [12] [1:2.3–1:2.5]."
    assert extract_citations(t) == ["0:1.2", "0:2.10", "5:3.4"]
    assert extract_citations("") == []


def test_rewrite_query_failure_falls_back_to_none():
    assert rewrite_query(FakeLLM(plan_exc=ConnectionRefused("http://x")), "m", "q") is None
    assert rewrite_query(FakeLLM(plan_json="garbage"), "m", "q") is None
    llm = FakeLLM(plan_json=PLAN)
    assert rewrite_query(llm, "m", "Что такое божество?")["keywords_en"] == ["deity"]
    c = llm.chat_calls[0]
    assert c["temperature"] == 0.0 and c["reasoning_effort"] == "none" and c["max_tokens"] <= 250


# ------------------------------------------------------------------ pipeline ---
def run(llm, searcher=None, rewrite=True):
    searcher = searcher or FakeSearcher()
    progress = []
    pipe = RagPipeline(searcher, llm, "m", query_rewrite=rewrite, max_context_length=32000)
    res = pipe.answer("Что такое божество?", k=5, progress=lambda *a: progress.append(a))
    return res, searcher, progress


def test_pipeline_uses_plan_for_retrieval_and_reports_keywords():
    llm = FakeLLM(plan_json=PLAN, answer="Божество — это X [0:1.2]. Y [0:2.10][99:9.9].")
    res, s, progress = run(llm)
    kinds = [c[0] for c in s.calls]
    assert kinds == ["fts", "semantic", "semantic"] and "hybrid" not in kinds
    assert s.calls[0][1] == "божество deity" and s.calls[0][2] == "or"
    assert s.calls[1][1] == "plan: определение божества" and s.calls[2][1] == "Что такое божество?"
    msgs = [p[4] for p in progress if p[0] == "analysis"]
    assert "Ищу: божество / deity" in msgs
    assert res["plan"]["summary"] == "Ищу: божество / deity"
    ids = [p["id"] for p in res["sources"]]
    assert "0:1.2" in ids and "0:2.10" in ids and "0:5.1" in ids
    assert res["cited_ids"] == ["0:1.2", "0:2.10"] and res["unknown_citations"] == ["99:9.9"]
    assert res["no_answer"] is False and res["error"] is None
    assert res["candidates"] == {"fts": 2, "semantic": 3}
    assert "Question" in llm.gen_calls[0][1]["content"]


def test_pipeline_falls_back_to_hybrid_when_rewrite_fails_or_disabled():
    res, s, progress = run(FakeLLM(plan_exc=ConnectionRefused("http://x")))
    assert [c[0] for c in s.calls] == ["hybrid"] and res["plan"] is None
    assert any("по тексту вопроса" in str(p[4]) for p in progress if p[0] == "analysis")
    res, s, _ = run(FakeLLM(plan_json=PLAN), rewrite=False)
    assert [c[0] for c in s.calls] == ["hybrid"]


def test_pipeline_detects_no_answer_and_reports_llm_error():
    res, _, _ = run(FakeLLM(plan_json=PLAN, answer="В найденных абзацах нет достаточных данных для ответа на этот вопрос."))
    assert res["no_answer"] is True and res["cited_ids"] == []
    res, _, _ = run(FakeLLM(plan_json=PLAN, answer_exc=ConnectionRefused("http://x")))
    assert res["answer"] is None and "LM Studio" in res["error"] and res["sources"]


def test_system_prompt_demands_citations_and_admitting_gaps():
    from app.llm.rag import SYSTEM_PROMPT
    assert "[0:1.2]" in SYSTEM_PROMPT and "нет достаточных данных" in SYSTEM_PROMPT


# ------------------------------------------------------- labeled inferences ---
from app.llm.rag import SYSTEM_PROMPT, has_inference, system_prompt  # noqa: E402


def test_prompt_variants():
    strict, loose = system_prompt("facts_only"), system_prompt("allow_inferences")
    assert strict == SYSTEM_PROMPT and "Вывод:" not in strict
    assert loose.startswith(SYSTEM_PROMPT)
    for needle in ("Вывод:", "Допущение:", "EVERY fact", "do NOT infer", "not a statement of the book"):
        assert needle in loose
    assert system_prompt("garbage") == strict  # unknown value -> the strict prompt


def test_has_inference_detection():
    assert has_inference("Факт [1:1.1].\n\nВывод: 530 − 19 = 511 [1:1.1].")
    assert has_inference("**Вывод:** ...") and has_inference("Вывод :  x")
    assert not has_inference("В выводе книги сказано...") and not has_inference("")


def test_pipeline_uses_prompt_for_strictness_and_flags_inference():
    ans = "Адам прожил 530 лет [0:1.2].\n\nВывод: 530 − 19 = 511. Допущение: это про Еву [0:1.2][0:2.10]."
    for mode, expect_rules in (("allow_inferences", True), ("facts_only", False)):
        llm = FakeLLM(plan_json=PLAN, answer=ans)
        pipe = RagPipeline(FakeSearcher(), llm, "m", strictness=mode, max_context_length=32000)
        res = pipe.answer("Сколько прожила Ева?", k=5)
        system = llm.gen_calls[0][0]["content"]
        assert ("Вывод:" in system) is expect_rules
        assert res["strictness"] == mode and res["has_inference"] is True
        assert res["cited_ids"] == ["0:1.2", "0:2.10"]  # facts used inside the inference are cited too
    res = RagPipeline(FakeSearcher(), FakeLLM(plan_json=PLAN), "m", strictness="weird").answer("q")
    assert res["strictness"] == "allow_inferences" and res["has_inference"] is False


def test_rewrite_prompt_covers_quantitative_questions_but_stays_narrow():
    from app.llm.rag import QUERY_REWRITE_PROMPT as Q
    assert "how long" in Q and "directly related" in Q and "Ева" in Q
    assert "no unrelated concepts" in Q.lower() or "Still no unrelated concepts" in Q


# ------------------------------------------------------------------ knobs / prompts / snapshot ---
from app import prompts as prompts_mod  # noqa: E402
from app.llm.rag import build_context_block, compose_system_prompt, fit_context  # noqa: E402


def P2(id_, en="english text", ru="русский текст", **extra):
    return {"id": id_, "paper": 0, "section": 1, "paragraph": 1, "en": en, "ru": ru, **extra}


def pipe(llm=None, searcher=None, **kw):
    return RagPipeline(searcher or FakeSearcher(), llm or FakeLLM(plan_json=PLAN), "m",
                       max_context_length=kw.pop("max_context_length", 32000), **kw)


def test_defaults_match_previous_behaviour():
    llm, s = FakeLLM(plan_json=PLAN), FakeSearcher()
    res = pipe(llm, s).answer("Что такое божество?")  # k from the default option (12)
    assert [c[0] for c in s.calls] == ["fts", "semantic", "semantic"]
    assert res["experiment"]["retrieval"]["k"] == 12 and res["experiment"]["options"]["rag_top_k"] == 12
    system = llm.gen_calls[0][0]["content"]
    assert system.startswith(prompts_mod.specs()["rag_system"]["default"])
    assert "FORMATTING" in system  # the new default formatting instruction is appended last
    assert llm.gen_calls[0][1]["content"].startswith("Sources / Источники:\n[")


def test_top_k_candidates_rrf_are_applied():
    class S(FakeSearcher):
        def fts(self, q, k=20, lang="both", mode=None):
            self.calls.append(("fts_k", k))
            return super().fts(q, k, lang, mode)

        def semantic(self, q, k=20):
            self.calls.append(("sem_k", k))
            return super().semantic(q, k)

    s = S()
    res = pipe(searcher=s, options={"rag_top_k": 2, "rag_candidates_fts": 7, "rag_candidates_semantic": 9,
                                    "rag_rrf_k": 5}).answer("q")
    assert ("fts_k", 7) in s.calls and ("sem_k", 9) in s.calls
    assert len(res["sources"]) == 2 and res["experiment"]["retrieval"]["rrf_k"] == 5
    assert res["experiment"]["retrieval"]["candidates_fts_asked"] == 7


def test_min_relevance_drops_weak_sources_but_keeps_one():
    class S(FakeSearcher):
        def fts(self, q, k=20, lang="both", mode=None):
            return [P2("a"), P2("b"), P2("c")]

        def semantic(self, q, k=20):
            return [P2("a"), P2("d")]

    res = pipe(searcher=S(), options={"rag_min_relevance": 0.9}).answer("q")
    ids = [p["id"] for p in res["sources"]]
    assert ids == ["a"] and res["experiment"]["retrieval"]["retrieved"] > 1
    assert res["experiment"]["retrieval"]["after_relevance_filter"] == 1


def test_sources_language_changes_what_the_model_sees():
    p = P2("x", en="ENGLISH", ru="РУССКИЙ")
    assert "ENGLISH" in build_context_block([p]) and "РУССКИЙ" in build_context_block([p])
    assert "ENGLISH" not in build_context_block([p], "ru") and "РУССКИЙ" in build_context_block([p], "ru")
    assert "РУССКИЙ" not in build_context_block([p], "en") and "ENGLISH" in build_context_block([p], "en")
    long = [P2(str(i), en="e" * 300, ru="р" * 300) for i in range(20)]
    both = fit_context(long, 600, 0, "both")
    ru_only = fit_context(long, 600, 0, "ru")
    assert len(ru_only) > len(both)  # one language fits roughly twice as many paragraphs


def test_context_share_limits_sources():
    llm = FakeLLM(plan_json=PLAN)

    class S(FakeSearcher):
        def fts(self, q, k=20, lang="both", mode=None):
            return [P2(str(i), en="e" * 400, ru="р" * 400) for i in range(10)]

        def semantic(self, q, k=20):
            return []

    full = pipe(llm, S(), max_context_length=6000, max_tokens=500).answer("q")
    half = pipe(FakeLLM(plan_json=PLAN), S(), max_context_length=6000, max_tokens=500,
                options={"rag_context_share": 0.3}).answer("q")
    assert len(half["sources"]) < len(full["sources"])


def test_answer_language_rule_and_strictness_in_system_prompt():
    llm = FakeLLM(plan_json=PLAN)
    pipe(llm, options={"rag_answer_lang": "ru"}).answer("q")
    assert "ALWAYS answer in Russian" in llm.gen_calls[0][0]["content"]
    llm2 = FakeLLM(plan_json=PLAN)
    pipe(llm2).answer("q")
    assert "ALWAYS answer in Russian" not in llm2.gen_calls[0][0]["content"]
    rp = prompts_mod.ResolvedPrompts()
    assert compose_system_prompt(rp, "facts_only", "question").endswith(rp.text("rag_formatting"))
    assert rp.text("rag_inference_rules") not in compose_system_prompt(rp, "facts_only", "question")


def test_custom_prompts_are_used_and_recorded():
    rp = prompts_mod.ResolvedPrompts({
        "rag_system": "CUSTOM SYSTEM", "rag_formatting": " PLAIN TEXT ONLY",
        "rag_user_template": "Q={question}\nS={sources}", "query_rewrite": "REWRITE with {max_terms}"}, preset="мой набор")
    llm = FakeLLM(plan_json=PLAN)
    res = pipe(llm, prompts=rp, options={"rag_rewrite_max_terms": 3, "rag_strictness": "facts_only"}).answer("Вопрос?")
    system, user = llm.gen_calls[0][0]["content"], llm.gen_calls[0][1]["content"]
    assert system == "CUSTOM SYSTEM PLAIN TEXT ONLY"
    assert user.startswith("Q=Вопрос?\nS=[") and "Sources /" not in user
    assert llm.chat_calls[0]["messages"][0]["content"] == "REWRITE with 3"  # {max_terms} substituted
    exp = res["experiment"]
    assert exp["prompts"]["system"] == system and exp["prompts"]["user"] == user
    assert exp["preset"] == "мой набор"
    assert exp["prompt_versions"]["rag_system"]["source"] == "custom"
    assert exp["prompt_versions"]["rag_formatting"]["source"] == "custom"
    assert exp["prompt_texts"]["rag_system"] == "CUSTOM SYSTEM" and exp["prompt_texts"]["query_rewrite"].startswith("REWRITE")
    assert "rag_inference_rules" not in exp["prompt_texts"]  # facts_only: that prompt was not sent


def test_invalid_override_falls_back_with_visible_warning():
    rp = prompts_mod.ResolvedPrompts({"rag_user_template": "broken: {question} only"})
    llm = FakeLLM(plan_json=PLAN)
    res = pipe(llm, prompts=rp).answer("q")
    assert res["error"] is None and llm.gen_calls[0][1]["content"].startswith("Sources / Источники:")
    assert res["prompt_warnings"] and "не применён" in res["prompt_warnings"][0]


def test_rewrite_max_terms_caps_the_plan():
    many = json.dumps({"keywords_ru": list("абвгде"), "keywords_en": list("abcdef"), "semantic_query": "s"})
    res = pipe(FakeLLM(plan_json=many), options={"rag_rewrite_max_terms": 2}).answer("q")
    assert len(res["plan"]["keywords_ru"]) == 2 and len(res["plan"]["keywords_en"]) == 2


def test_experiment_snapshot_is_complete():
    res = pipe(FakeLLM(plan_json=PLAN), meta={"reasoning_setting": "off", "fts_weights": {"en": 1}}).answer("q", k=3)
    e = res["experiment"]
    for key in ("app", "model", "temperature", "max_tokens", "reasoning", "context_length", "prompt_versions",
                "prompt_texts", "prompts", "options", "search_settings", "retrieval", "plan", "timings", "stats"):
        assert key in e, key
    assert e["kind"] == "rag" and e["app"]["version"] and e["app"]["date"]
    assert e["reasoning"]["setting"] == "off" and e["search_settings"]["fts_weights"] == {"en": 1}
    assert set(e["timings"]) >= {"plan_s", "retrieval_s", "context_s", "generation_s", "total_s"}
    r = e["retrieval"]
    assert r["k"] == 3 and r["sources_sent"] == len(res["sources"]) and r["fallback_plain_hybrid"] is False
    assert e["prompts"]["rewrite_system"] and e["prompts"]["rewrite_user"] == "q"


def test_neighbours_are_added_and_citable(tmp_path):
    from app.core.db import open_db
    from tests.helpers import create_fake_book_db

    db = str(tmp_path / "b.db")
    create_fake_book_db(db)
    conn = open_db(db, readonly=True)

    class S(FakeSearcher):
        def fts(self, q, k=20, lang="both", mode=None):
            return [P2("147.2.2")]

        def semantic(self, q, k=20):
            return []

    res = pipe(searcher=S(), conn=conn, options={"rag_neighbors": 1}).answer("q")
    ids = [p["id"] for p in res["sources"]]
    assert ids == ["147.2.1", "147.2.2", "147.2.3"]  # reading order: before, source, after
    assert [p.get("neighbor_of") for p in res["sources"]] == ["147.2.2", None, "147.2.2"]
    assert res["experiment"]["retrieval"]["with_neighbors"] == 3
    res0 = pipe(searcher=S(), conn=conn, options={"rag_neighbors": 0}).answer("q")
    assert [p["id"] for p in res0["sources"]] == ["147.2.2"]


# ------------------------------------------------- automatic section / list expansion ---

def test_list_question_detection():
    from app.llm.rag import is_list_question
    assert is_list_question("перечисли участников совета двадцати четырех", None)
    assert is_list_question("Кто входит в совет?", None)
    assert is_list_question("Совет двадцати четырёх", {"answer_type": "list"})
    assert not is_list_question("Сколько прожила Ева?", {"answer_type": "other"})


def test_plan_answer_type_parsed():
    p = parse_search_plan('{"keywords_ru":["совет"],"keywords_en":[],"answer_type":"list"}')
    assert p["answer_type"] == "list"
    assert parse_search_plan('{"keywords_ru":["совет"]}')["answer_type"] == "other"


def test_section_title_match_needs_all_title_words():
    import sqlite3
    from app.llm.rag import match_section_titles
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE sections (paper INTEGER, section INTEGER, en TEXT, ru TEXT)")
    c.executemany("INSERT INTO sections VALUES (?,?,?,?)", [
        (45, 4, "The Four and Twenty Counselors", "Совет Двадцати Четырех"),
        (45, 3, "The System Government", "Правительство системы"),
        (74, 1, "Adam", "Адам"),
    ])
    hits = match_section_titles(c, "перечисли участников совета двадцати четырёх", None)
    assert [(h["paper"], h["section"]) for h in hits] == [(45, 4)]
    assert match_section_titles(c, "Кто такой Адам?", None) == []      # one-word titles never match
    assert match_section_titles(c, "что такое совет?", None) == []     # only part of the title
