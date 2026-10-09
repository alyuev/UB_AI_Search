from app.core.search import _is_stop
from app.core.terms import TermTools
from app.llm.rag import ensure_rare_terms


class FakeSearcher:
    def fts_terms(self, st, k=20):
        return [{"id": "80:9.6"}][:k]


def test_rare_term_forced_into_sources():
    retrieved = [{"id": f"1:{i}.1"} for i in range(12)] + [{"id": "80:9.6"}]  # rare one fused below k=12
    merged = {"terms": [{"ok": True, "lang": "ru", "key": "украина", "text": "Украина", "hits": 1,
                         "src": "user", "prefix": "укра"}]}
    out, info = ensure_rare_terms(retrieved, merged, FakeSearcher(), 12, 8, 5)
    assert len(out) == 12 and out[0]["id"] == "80:9.6"
    assert info[0]["id"] == "80:9.6" and info[0]["in_top_k_before"] is False
    out2, info2 = ensure_rare_terms(retrieved, merged, FakeSearcher(), 12, 0, 5)  # disabled
    assert out2 is retrieved and info2 == []


def test_meta_verbs_are_stop_words():
    for w in ("упоминаются", "mentioned", "говорится", "exists"):
        assert _is_stop(w)
    assert _is_stop("упоминаться", "упоминаться")
    assert not _is_stop("украина")


def test_suggestion_threshold():
    from app.core.terms import Vocabulary
    v = Vocabulary.__new__(Vocabulary)
    v.lemma, v.en, v._by_first = {}, {"urantians": 50, "ukrainian": 3}, {}
    assert "urantians" not in v.nearest("ukrainians", "en")
    assert "ukrainian" in v.nearest("ukrainians", "en")
