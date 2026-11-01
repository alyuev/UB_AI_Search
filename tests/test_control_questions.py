"""Control questions (docs/control_questions.md): on the REAL book database, the paragraphs that must reach the
model for questions that once went wrong. The model is replaced by a stub returning a fixed search plan, so
the check runs in seconds; the embedder (llama-server) is the real one. Skipped when the book or the
embedding model is not available on this machine."""
import json
import os

import pytest

from app import config
from app import options as opts_mod
from app.core import db as core_db
from app.llm.rag import RagPipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOK = os.path.join(ROOT, "library", "urantia", "book.db")
CONTEXT = 32768   # a typical local model; the real one may be larger, which only helps
with open(os.path.join(os.path.dirname(__file__), "control_questions.json"), encoding="utf-8") as f:
    CASES = json.load(f)["questions"]


class _Stats:
    def to_dict(self):
        return {"model": "stub", "elapsed_s": 0.0}

    def summary_line(self):
        return "stub"


class PlanLLM:
    """Query preparation -> the case's plan; the answer itself is not needed."""

    def __init__(self, plan: dict):
        self.plan = json.dumps(plan, ensure_ascii=False)

    def chat(self, messages, model, temperature=0.0, max_tokens=200, timeout=None, reasoning_effort=None, **kw):
        return self.plan

    def chat_with_stats(self, messages, model, temperature=0.2, max_tokens=1024, timeout=None,
                        reasoning_effort=None, on_delta=None, **kw):
        return "-", _Stats()


@pytest.fixture(scope="module")
def book():
    if not os.path.exists(BOOK):
        pytest.skip("нет базы книги library/urantia/book.db")
    from app.core import embedder as core_embedder
    from app.core.search import Searcher
    conn = core_db.open_db(BOOK, readonly=True)
    info = core_db.get_book_info(conn, "urantia") or {}
    try:
        emb = core_embedder.create_embedder(info.get("embedding_model"), runtime=info.get("embedding_runtime") or "auto")
    except Exception as exc:  # noqa: BLE001
        conn.close()
        pytest.skip(f"модель эмбеддингов недоступна: {exc}")
    searcher = Searcher(BOOK, emb)
    yield conn, searcher
    conn.close()
    if hasattr(emb, "close"):
        emb.close()


def _run(book, case) -> dict:
    conn, searcher = book
    options = opts_mod.effective(config.load_settings())
    pipe = RagPipeline(searcher, PlanLLM(case["plan"]), model="stub", max_context_length=CONTEXT,
                       options=options, conn=conn, query_rewrite=True)
    return pipe.answer(case["question"])


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_control_question(book, case):
    if case.get("known_fail"):
        pytest.xfail(case["known_fail"])
    res = _run(book, case)
    if "overview" in case:
        assert bool(res.get("overview")) == case["overview"], \
            "обзор по всей книге " + ("не включился" if case["overview"] else "включился без нужды")
    got = [s["id"] for s in res.get("sources") or []]
    missing = [i for i in case.get("must", []) if i not in got]
    assert not missing, f"не попали к модели: {missing}; передано {len(got)}: {got}"
    if case.get("any"):
        assert any(i in got for i in case["any"]), f"ни одного из {case['any']} среди {got}"
    if case.get("terms_absent"):
        st = (res.get("plan") or {}).get("search_terms") or {}
        words = {w.lower() for lst in st.values() for w in lst}
        bad = [w for w in case["terms_absent"] if w in words]
        assert not bad, f"в поиск попали лишние слова: {bad}"


class _LengthStats(_Stats):
    stop_reason = "length"


class ThinkingOnlyLLM(PlanLLM):
    """Like recursivesi/flint on a long part: every token goes to reasoning, the answer stays empty."""

    def __init__(self, plan: dict):
        super().__init__(plan)
        self.calls = 0

    def chat_with_stats(self, messages, model, **kw):
        self.calls += 1
        return "", _LengthStats()


def test_overview_with_a_model_that_only_thinks(book):
    """An empty answer is not «nothing found»: the overview stops early and says why."""
    conn, searcher = book
    case = next(c for c in CASES if c["id"] == "pharisees")
    llm = ThinkingOnlyLLM(case["plan"])
    pipe = RagPipeline(searcher, llm, model="flint", max_context_length=CONTEXT,
                       options=opts_mod.effective(config.load_settings()), conn=conn, query_rewrite=True)
    res = pipe.answer(case["question"], overview=True)
    assert res["error"] and "рассуждения" in res["error"]
    assert "нет достаточных данных" not in (res.get("answer") or "").lower()
    assert llm.calls == 4   # two parts, each tried twice, then it stops
