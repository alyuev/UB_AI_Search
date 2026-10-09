"""Vocabulary check, «did you mean», automatic RU<->EN glossary and the word-search merge (query preparation)."""
import json
import os
import sqlite3
import threading

import pytest

from app.core import terms as T
from app.core.db import open_db
from app.core.indexer import _fts_rows, _lemmatize_ru
from app.core.search import Searcher
from app.llm import rag as R
from app import prompts as prompts_mod

PAIRS = [
    ("Edentia is the constellation capital where Eden was founded.", "Эдемия — столица созвездия, где был основан Эдем."),
    ("The rulers of Edentia govern from afar.", "Правители Эдемии управляют издалека."),
    ("Adam and Eve left Eden for the second garden.", "Адам и Ева покинули Эдем ради второго сада."),
    ("Eden was a paradise for the first Adam.", "Эдем был раем для первого Адама."),
    ("The Thought Adjuster dwells within the mind.", "Настройщик Мысли обитает в разуме."),
    ("Each Adjuster is a fragment of the Father.", "Каждый Настройщик — частица Отца."),
    ("The agondonters are will creatures who believe without seeing.", "Агондонтеры — волевые создания, верующие, не видя."),
    ("Edentia and Eden are connected by the Material Sons.", "Эдемия и Эдем связаны Материальными Сынами."),
    ("Deity is the source of all divinity.", "Божество — источник всего божественного."),
    ("The Deity of Paradise is one.", "Божество Рая едино."),
    ("A quiet evening in Jerusalem.", "Тихий вечер в Иерусалиме."),
    ("The disciples listened intently.", "Ученики слушали внимательно."),
]


def make_db(path, with_glossary=False):
    import pymorphy3

    morph = pymorphy3.MorphAnalyzer()
    conn = open_db(str(path), readonly=False)
    conn.execute("INSERT INTO books (book_id, book_name, language_source, language_translation) VALUES "
                 "('b', 'Mini', 'en', 'ru')")
    for i, (en, ru) in enumerate(PAIRS):
        conn.execute("INSERT INTO paragraphs (id, book_id, seq, paper, section, paragraph, en, ru, ru_lemma) "
                     "VALUES (?,?,?,?,?,?,?,?,?)",
                     (f"1:1.{i + 1}", "b", i, 1, 1, i + 1, en, ru, _lemmatize_ru(ru, morph)))
    conn.commit()
    conn.executemany("INSERT INTO paragraphs_fts (id, en, ru, ru_lemma, title_en, title_ru) VALUES (?,?,?,?,?,?)",
                     _fts_rows(conn, morph))
    conn.commit()
    conn.close()
    if with_glossary:
        T.build_glossary(str(path))
    return str(path)


@pytest.fixture
def db(tmp_path):
    return make_db(tmp_path / "book.db", with_glossary=True)


@pytest.fixture
def db_plain(tmp_path):
    return make_db(tmp_path / "plain.db", with_glossary=False)


def pair(db, ru):
    c = sqlite3.connect(db)
    try:
        return c.execute("SELECT en_term, mutual, df FROM term_glossary WHERE ru_lemma=? ORDER BY score DESC", (ru,)).fetchall()
    finally:
        c.close()


# ------------------------------------------------------------- similarity ---

def test_edit_distance_and_jaro_winkler():
    assert T.levenshtein("эдемия", "эдентия") == 2 and T.levenshtein("abc", "abc") == 0
    assert T.levenshtein("abcdef", "uvwxyz", 2) == 3          # gives up early: "more than the limit"
    assert T.jaro_winkler("martha", "marhta") > 0.95 and T.jaro_winkler("abc", "xyz") == 0
    assert T.translit("Агондонтер") == "agondonter"


# --------------------------------------------------------------- glossary ---

def test_build_glossary_pairs_names_in_both_directions(db):
    assert pair(db, "эдем")[0][0] == "eden" and pair(db, "эдем")[0][1] == 1
    assert pair(db, "эдемия")[0][0] == "edentia"
    assert pair(db, "настройщик")[0][0] == "adjuster"
    assert pair(db, "божество")[0][0] == "deity"
    # a name met only ONCE is accepted because both spellings are clearly the same name
    assert pair(db, "агондонтер")[0][0].startswith("agondonter")
    c = sqlite3.connect(db)
    assert c.execute("SELECT COUNT(*) FROM term_glossary_meta").fetchone()[0] == 1
    cols = [r[1] for r in c.execute("PRAGMA table_info(term_glossary)")]
    assert cols[:4] == ["ru_lemma", "en_term", "score", "df"]
    c.close()


def test_glossary_info_progress_and_rebuild_is_idempotent(tmp_path):
    p = make_db(tmp_path / "b.db")
    assert T.glossary_info(p) == {"exists": False, "pairs": 0, "mutual": 0, "built_at": None}
    seen = []
    r = T.build_glossary(p, progress=lambda st, cur, tot, cid, msg: seen.append((st, cur, tot)))
    assert seen and seen[0][0] == "glossary" and seen[-1][1] == seen[-1][2] and r["pairs"] > 0
    info = T.glossary_info(p)
    assert info["exists"] and info["pairs"] == r["pairs"] and info["built_at"]
    T.build_glossary(p)
    assert T.glossary_info(p)["pairs"] == r["pairs"]


def test_cancel_keeps_the_old_glossary(db):
    before = pair(db, "эдем")
    ev = threading.Event()
    ev.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        T.build_glossary(db, cancel=ev)
    assert pair(db, "эдем") == before


def test_glossary_never_touches_other_tables(db):
    c = sqlite3.connect(db)
    n = c.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0]
    f = c.execute("SELECT COUNT(*) FROM paragraphs_fts").fetchone()[0]
    c.close()
    T.build_glossary(db)
    c = sqlite3.connect(db)
    assert (c.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0],
            c.execute("SELECT COUNT(*) FROM paragraphs_fts").fetchone()[0]) == (n, f)
    c.close()


def test_empty_database_gives_clear_error(tmp_path):
    conn = open_db(str(tmp_path / "e.db"), readonly=False)
    conn.close()
    with pytest.raises(RuntimeError, match="нет абзацев"):
        T.build_glossary(str(tmp_path / "e.db"))


# -------------------------------------------------------------- vocabulary ---

def test_vocabulary_counts_and_did_you_mean(db):
    s = Searcher(db)
    v = s.terms.vocab
    assert v.ok and v.lemma["эдемия"] >= 3 and v.en["edentia"] == 3
    assert "эдемия" in v.nearest("эдентия", "ru")              # two typos away
    assert "edentia" in v.nearest("edentya", "en")
    assert v.nearest("эдемиа", "ru")[0] == "эдемия"
    assert v.nearest("zz", "en") == []                          # too short to guess
    # the vocabulary is a TEMP table: nothing was written into the database file
    c = sqlite3.connect(db)
    assert "vocab_col" not in [r[0] for r in c.execute("SELECT name FROM sqlite_master")]
    c.close()


def test_inflected_user_word_is_found_by_its_dictionary_form(db):
    tt = Searcher(db).terms
    t = tt.check_word("эдемом")                                # instrumental case of «Эдем»
    assert t["key"] == "эдем" and t["ok"] and t["hits"] >= 4
    assert t["equiv"][0]["key"] == "eden"
    assert tt.check_word("Eden")["equiv"][0]["key"] == "эдем"  # both directions


def test_unknown_word_gets_suggestions_with_translation(db):
    tt = Searcher(db).terms
    t = tt.check_word("эдемиар")
    assert not t["ok"] and t["suggest"][0]["key"] == "эдемия" and t["suggest"][0]["equiv"] == "Edentia"
    line = tt.status_lines([t])[0]
    assert "слова «эдемиар» нет в книге" in line and "эдемия ↔ Edentia" in line.lower().replace("edentia", "Edentia")


def test_hints_text_for_the_model(db):
    tt = Searcher(db).terms
    ut = tt.analyze_user("какая связь между эдемом и эдемиар?")
    h = tt.hints_text(ut)
    assert "эдемом (эдем) → Eden" in h and "связь" in h
    assert "эдемиар (нет в книге; близкие слова: эдемия ↔ Edentia)" in h
    assert "какая" not in h and "между" not in h               # stop words are not the question's content words


# ----------------------------------------------------------- merge with plan ---

def test_model_that_replaces_the_users_word_cannot_hide_it(db):
    """The incident: «эдемом» was turned into «отек / edema»; the user's own words are still searched."""
    tt = Searcher(db).terms
    ut = tt.analyze_user("какая связь между эдемом и эдемией?")
    plan = {"keywords_ru": ["отек", "эдемия"], "keywords_en": ["edema", "edemia"], "semantic_query": ""}
    m = tt.merge(ut, plan)
    st = m["search"]
    assert "эдем" in st.ru_exact and "эдемия" in st.ru_exact          # the user's words
    assert "эдем" in st.ru_prefix                                        # stem of 4+ letters
    assert "eden" in st.en_exact and "edentia" in st.en_exact          # glossary translations
    assert "отек" not in st.ru_exact and "edema" not in st.en_exact    # invented terms are dropped
    assert m["dropped"] == ["отек", "edema", "edemia"]
    lines = "\n".join(m["lines"])
    assert "✓ Эдем — " in lines and "✗ отек — в книге нет" in lines and "✗ edemia — в книге нет" in lines


def test_everything_the_model_proposed_is_missing_falls_back_to_user_words(db):
    tt = Searcher(db).terms
    m = tt.merge(tt.analyze_user("кто такие агондонтеры?"),
                 {"keywords_ru": ["мозг"], "keywords_en": ["brain"], "semantic_query": ""})
    assert m["model_all_missing"] and "агондонтер" in m["search"].ru_exact
    assert "agondonter" in "".join(m["search"].en_exact + ["agondonter"])


def test_works_without_the_glossary_table(db_plain):
    s = Searcher(db_plain)
    tt = s.terms
    assert tt.available() and not tt.glossary.exists
    ut = tt.analyze_user("что такое божество?")
    assert ut[0]["ok"] and ut[0]["equiv"] == []
    m = tt.merge(ut, None)
    res = s.fts_terms(m["search"], k=5)
    assert {r["id"] for r in res} == {"1:1.9", "1:1.10"}
    assert tt.hints_text(ut).startswith("божество → есть в книге")


def test_fts_terms_matches_prefixes_phrases_and_both_languages(db):
    s = Searcher(db)
    st = T.SearchTerms()
    st.add_ru(prefix="эдем")
    ids = {r["id"] for r in s.fts_terms(st, 20)}
    assert {"1:1.1", "1:1.2", "1:1.3"} <= ids                          # Эдем, Эдемия, Эдемии ...
    st = T.SearchTerms()
    st.add_en(phrase="thought adjuster")
    assert [r["id"] for r in s.fts_terms(st, 20)] == ["1:1.5"]
    assert s.fts_terms(T.SearchTerms(), 5) == []                       # nothing to search -> empty, no error


# ------------------------------------------------------------ prompt/pipeline ---

class RecordingLLM:
    profile = {"name": "OpenRouter", "kind": "openrouter"}

    def __init__(self, plan):
        self.plan, self.calls = plan, []

    def chat(self, messages, model, temperature=0.2, max_tokens=1024, timeout=None, stream=False, reasoning_effort=None):
        self.calls.append(messages)
        return self.plan

    def chat_with_stats(self, messages, model, temperature=0.2, max_tokens=1024, timeout=None, reasoning_effort=None,
                        on_delta=None, on_status=None):
        from tests.server_helpers import fake_stats
        return "Эдемия и Эдем связаны [1:1.8].", fake_stats()


def test_rewrite_prompt_is_valid_and_mentions_hints():
    assert prompts_mod.validate("query_rewrite", R.QUERY_REWRITE_PROMPT) is None
    assert "Слова из вопроса" in R.QUERY_REWRITE_PROMPT and "NEVER replace" in R.QUERY_REWRITE_PROMPT
    assert "{max_terms}" in R.QUERY_REWRITE_PROMPT


def test_pipeline_uses_hints_and_verified_terms(db):
    plan = json.dumps({"keywords_ru": ["отек", "эдемия"], "keywords_en": ["edema", "edemia"],
                       "semantic_query": "связь между Эдемом и Эдемией"})
    llm = RecordingLLM(plan)
    msgs = []
    pipe = R.RagPipeline(Searcher(db), llm, "m-1", query_rewrite=True, max_context_length=32000)
    res = pipe.answer("какая связь между эдемом и эдемией?", k=6,
                      progress=lambda st, cur, tot, cid, m: msgs.append((st, m)))
    user_msg = llm.calls[0][1]["content"]
    assert user_msg.startswith("какая связь между эдемом и эдемией?")
    assert "Слова из вопроса и их соответствия в книге: " in user_msg and "эдемом (эдем) → Eden" in user_msg
    analysis = [m for st, m in msgs if st == "analysis"][-1]
    assert "✓ Эдем — " in analysis and "✗ отек — в книге нет" in analysis
    gen = [m for st, m in msgs if st == "generation"][0]
    assert gen == "отправка в OpenRouter · m-1"                             # the real provider, not «LM Studio»
    ids = {s["id"] for s in res["sources"]}
    assert {"1:1.1", "1:1.2", "1:1.3", "1:1.8"} & ids
    assert res["plan"]["dropped"] == ["отек", "edema", "edemia"] and res["plan"]["terms"]
    exp = res["experiment"]["retrieval"]
    assert exp["terms_used"] is True and "эдем" in exp["search_terms"]["ru_exact"]
    assert exp["fallback_plain_hybrid"] is False


def test_pipeline_without_rewrite_keeps_plain_hybrid(db):
    llm = RecordingLLM("{}")
    pipe = R.RagPipeline(Searcher(db), llm, "m", query_rewrite=False, max_context_length=32000)
    res = pipe.answer("Эдем", k=3)
    assert llm.calls == [] and res["experiment"]["retrieval"]["fallback_plain_hybrid"] is True


def test_pipeline_when_the_model_fails_still_searches_user_words(db):
    class Broken(RecordingLLM):
        def chat(self, *a, **k):
            raise RuntimeError("boom")

    pipe = R.RagPipeline(Searcher(db), Broken(""), "m", query_rewrite=True, max_context_length=32000)
    res = pipe.answer("кто такие агондонтеры?", k=3)
    assert res["plan"]["model_failed"] is True and any(s["id"] == "1:1.7" for s in res["sources"])


def test_status_names_the_real_provider_for_a_generic_client(db):
    class Plain(RecordingLLM):
        profile = None            # no provider name known

    msgs = []
    R.RagPipeline(Searcher(db), Plain("{}"), "x", query_rewrite=False, max_context_length=32000).answer(
        "Эдем", k=2, progress=lambda st, cur, tot, cid, m: msgs.append((st, m)))
    assert [m for st, m in msgs if st == "generation"][0] == "отправка в модель · x"
    assert not any("LM Studio" in m for _, m in msgs)


def test_partial_answer_is_returned_with_a_warning(db):
    from app.llm.providers import ProviderError
    from tests.server_helpers import fake_stats

    class Cut(RecordingLLM):
        def chat_with_stats(self, *a, **k):
            e = ProviderError("Провайдер модели (Nvidia) перегружен. Ответ оборвался на середине.")
            e.partial, e.partial_stats = "Эдем — это сад [1:1.1].", fake_stats()
            raise e

    res = R.RagPipeline(Searcher(db), Cut("{}"), "m", query_rewrite=False, max_context_length=32000).answer("Эдем", k=5)
    assert res["answer"] == "Эдем — это сад [1:1.1]." and res["truncated"] is True and "Nvidia" in res["error"]
    assert any("Ответ оборван" in w for w in res["warnings"]) and res["cited_ids"] == ["1:1.1"]


# ---------------------------------------------------- Paper 0 scratch database ---

PAPER0 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench", "foreword", "paper0.db")


@pytest.mark.skipif(not os.path.exists(PAPER0), reason="Paper 0 scratch database is not built on this machine")
def test_glossary_on_the_paper0_slice(tmp_path):
    import shutil

    p = str(tmp_path / "paper0.db")
    shutil.copy(PAPER0, p)
    r = T.build_glossary(p)
    assert r["pairs"] > 100
    s = Searcher(p)
    tt = s.terms
    deity = tt.check_word("божество")
    assert deity["ok"] and any(e["key"] == "deity" for e in deity["equiv"])
    assert not tt.check_word("эдемиар")["ok"]
