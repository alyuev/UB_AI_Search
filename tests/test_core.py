import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from app.core import db as dbmod
from app.core.parser import parse_json, parse_md
from app.core.search import _escape_fts_query, Searcher

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "urantia_bilingual_rag.json"


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

def test_parse_json_real_corpus_counts():
    records = list(parse_json(JSON_PATH))
    assert len(records) == 14585
    for r in records[:5]:
        assert set(r.keys()) == {"id", "paper", "section", "paragraph", "en", "ru"}
        assert isinstance(r["paper"], int)
        assert isinstance(r["en"], str) and isinstance(r["ru"], str)


def test_parse_json_skips_bad_records(tmp_path):
    data = [
        {"id": "1:1.1", "paper": 1, "section": 1, "paragraph": 1, "en": "hello", "ru": "привет"},
        {"id": "bad-id", "paper": 1, "section": 1, "paragraph": 2, "en": "x", "ru": "y"},
        {"id": "1:1.3", "paper": 1, "section": 1, "en": "missing paragraph field", "ru": "y"},
        {"id": "1:1.4", "paper": 1, "section": 1, "paragraph": 4, "en": "", "ru": ""},
    ]
    p = tmp_path / "sample.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    records = list(parse_json(p))
    assert len(records) == 1
    assert records[0]["id"] == "1:1.1"


def test_parse_md_basic(tmp_path):
    content = (
        "ID: 2:3.4\n"
        "EN: Hello world.\n"
        "RU: Привет мир.\n"
        "---\n"
        "ID: 2:3.5\n"
        "EN: Second paragraph\n"
        "line two.\n"
        "RU: Второй абзац.\n"
        "---\n"
    )
    p = tmp_path / "sample.md"
    p.write_text(content, encoding="utf-8")
    records = list(parse_md(p))
    assert len(records) == 2
    assert records[0]["id"] == "2:3.4"
    assert records[0]["paper"] == 2 and records[0]["section"] == 3 and records[0]["paragraph"] == 4
    assert "Second paragraph" in records[1]["en"]
    assert "line two." in records[1]["en"]


# --------------------------------------------------------------------------
# db
# --------------------------------------------------------------------------

def _make_test_db(tmp_path) -> Path:
    db_path = tmp_path / "test.db"
    conn = dbmod.open_db(db_path)
    conn.execute(
        "INSERT INTO books (book_id, book_name, language_source, language_translation, "
        "source_file, created_at, embedding_model, embedding_dim, index_scheme) "
        "VALUES ('b1','Book','en','ru','x.json','now','m',384,'D')"
    )
    rows = [
        ("1:1.1", "b1", 0, 1, 1, 1, "First EN", "Первый RU", "первый ru", None, None),
        ("1:1.2", "b1", 1, 1, 1, 2, "Second EN thought adjuster", "Второй RU", "второй ru", None, None),
        ("1:1.3", "b1", 2, 1, 1, 3, "Third EN", "Третий RU", "третий ru", None, None),
        ("2:1.1", "b1", 3, 2, 1, 1, "Other paper EN", "Другой RU", "другой ru", None, None),
    ]
    conn.executemany(
        "INSERT INTO paragraphs (id, book_id, seq, paper, section, paragraph, en, ru, "
        "ru_lemma, en_lemma, chunk_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.executemany(
        "INSERT INTO paragraphs_fts (id, en, ru, ru_lemma) VALUES (?,?,?,?)",
        [(r[0], r[6], r[7], r[8]) for r in rows],
    )
    conn.commit()
    return db_path


def test_db_get_paragraph_and_context(tmp_path):
    db_path = _make_test_db(tmp_path)
    conn = dbmod.open_db(db_path)

    para = dbmod.get_paragraph(conn, "1:1.2")
    assert para is not None
    assert para["en"] == "Second EN thought adjuster"
    assert "previous_id" not in para and "next_id" not in para

    assert dbmod.get_paragraph(conn, "nope") is None

    ctx = dbmod.get_context(conn, "1:1.2", before=1, after=1)
    assert [p["id"] for p in ctx] == ["1:1.1", "1:1.2", "1:1.3"]


def test_db_get_range_and_lists(tmp_path):
    db_path = _make_test_db(tmp_path)
    conn = dbmod.open_db(db_path)

    rng = dbmod.get_range(conn, paper=1, section=1, para_from=1, para_to=2)
    assert [p["id"] for p in rng] == ["1:1.1", "1:1.2"]

    assert [p["paper"] for p in dbmod.list_papers(conn)] == [1, 2]
    assert [s["section"] for s in dbmod.list_sections(conn, 1)] == [1]


def test_db_check_integrity(tmp_path):
    db_path = _make_test_db(tmp_path)
    conn = dbmod.open_db(db_path)
    ok, msg = dbmod.check_integrity(conn)
    assert ok, msg


def test_quantize_dequantize_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        vec = rng.normal(size=384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        blob, scale = dbmod.quantize_vector(vec)
        assert isinstance(blob, bytes)
        assert len(blob) == 384  # int8: 1 byte per dim
        recovered = dbmod.dequantize_vector(blob, scale)
        # cosine similarity between original and dequantized should be very high
        cos = float(vec @ (recovered / np.linalg.norm(recovered)))
        assert cos > 0.99, cos


def test_quantize_zero_vector():
    vec = np.zeros(16, dtype=np.float32)
    blob, scale = dbmod.quantize_vector(vec)
    assert scale == 1.0
    recovered = dbmod.dequantize_vector(blob, scale)
    assert np.allclose(recovered, 0.0)


def test_searcher_mixed_dtype_load(tmp_path):
    """A book.db may have some vectors stored as f32 and others as i8
    (e.g. mid-conversion, or an old db missing the dtype column entirely).
    Searcher must load both correctly."""
    db_path = tmp_path / "mixed.db"
    conn = dbmod.open_db(db_path)
    conn.execute(
        "INSERT INTO books (book_id, book_name, language_source, language_translation, "
        "source_file, created_at, embedding_model, embedding_dim, index_scheme) "
        "VALUES ('b1','Book','en','ru','x.json','now','m',4,'D')"
    )
    rows = [
        ("1:1.1", "b1", 0, 1, 1, 1, "First EN", "Первый RU", "первый ru", None, None),
        ("1:1.2", "b1", 1, 1, 1, 2, "Second EN", "Второй RU", "второй ru", None, None),
    ]
    conn.executemany(
        "INSERT INTO paragraphs (id, book_id, seq, paper, section, paragraph, en, ru, "
        "ru_lemma, en_lemma, chunk_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.executemany(
        "INSERT INTO paragraphs_fts (id, en, ru, ru_lemma) VALUES (?,?,?,?)",
        [(r[0], r[6], r[7], r[8]) for r in rows],
    )

    vec1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vec2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    import struct

    f32_blob = struct.pack("<4f", *vec1)
    conn.execute(
        "INSERT INTO embeddings (para_id, lang, model, vector, dtype, scale) VALUES (?,?,?,?,?,?)",
        ("1:1.1", "en", "m", f32_blob, "f32", None),
    )
    i8_blob, scale = dbmod.quantize_vector(vec2)
    conn.execute(
        "INSERT INTO embeddings (para_id, lang, model, vector, dtype, scale) VALUES (?,?,?,?,?,?)",
        ("1:1.2", "en", "m", i8_blob, "i8", scale),
    )
    conn.commit()

    class FakeEmbedder:
        model_name = "m"
        runtime = "onnx"
        dim = 4

        def embed_query(self, text):
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    searcher = Searcher(db_path, embedder=FakeEmbedder())
    assert searcher._en_mat.shape == (2, 4)
    # row for 1:1.1 (f32) should be close to [1,0,0,0]
    idx1 = searcher._id_to_idx["1:1.1"]
    assert np.allclose(searcher._en_mat[idx1], [1.0, 0.0, 0.0, 0.0], atol=1e-3)
    # row for 1:1.2 (dequantized i8) should be close to [0,1,0,0]
    idx2 = searcher._id_to_idx["1:1.2"]
    assert np.allclose(searcher._en_mat[idx2], [0.0, 1.0, 0.0, 0.0], atol=1e-2)


def test_db_titles(tmp_path):
    db_path = _make_test_db(tmp_path)
    conn = dbmod.open_db(db_path)
    conn.execute("INSERT INTO parts (part, en, ru, first_paper, last_paper) VALUES (1,'Part One','Часть первая',1,2)")
    conn.execute("INSERT INTO papers (paper, en, ru, part) VALUES (1,'Paper One','Документ первый',1)")
    conn.execute("INSERT INTO sections (paper, section, en, ru) VALUES (1,1,'Section One','Раздел первый')")
    conn.commit()

    titles = dbmod.get_titles(conn, 1, section=1)
    assert titles["paper_en"] == "Paper One"
    assert titles["paper_ru"] == "Документ первый"
    assert titles["section_en"] == "Section One"
    assert titles["part_en"] == "Part One"

    titles_no_section = dbmod.get_titles(conn, 2)
    assert titles_no_section["paper_en"] is None
    assert titles_no_section["section_en"] is None

    para = dbmod.get_paragraph(conn, "1:1.1")
    assert para["paper_title"] == {"en": "Paper One", "ru": "Документ первый"}
    assert para["section_title"] == {"en": "Section One", "ru": "Раздел первый"}

    papers = dbmod.list_papers(conn)
    p1 = next(p for p in papers if p["paper"] == 1)
    assert p1["title"]["en"] == "Paper One"
    p2 = next(p for p in papers if p["paper"] == 2)
    assert "title" not in p2

    sections = dbmod.list_sections(conn, 1)
    s1 = next(s for s in sections if s["section"] == 1)
    assert s1["title"]["en"] == "Section One"


def test_indexer_load_titles_helper(tmp_path):
    from app.core.indexer import _load_titles

    titles = {
        "parts": [{"part": 1, "en": "Part One", "ru": "Часть 1", "papers": [1, 1]}],
        "papers": {
            "1": {
                "en": "Paper One", "ru": "Документ 1", "part": 1,
                "sections": {"1": {"en": "Sec A", "ru": "Разд А"}, "2": {"en": "Sec B", "ru": "Разд Б"}},
            }
        },
    }
    titles_path = tmp_path / "titles.json"
    titles_path.write_text(json.dumps(titles, ensure_ascii=False), encoding="utf-8")

    db_path = tmp_path / "out.db"
    conn = dbmod.open_db(db_path)
    _load_titles(titles_path, conn)

    prow = conn.execute("SELECT * FROM papers WHERE paper = 1").fetchone()
    assert prow["en"] == "Paper One"
    srow = conn.execute("SELECT * FROM sections WHERE paper=1 AND section=2").fetchone()
    assert srow["en"] == "Sec B"
    partrow = conn.execute("SELECT * FROM parts WHERE part = 1").fetchone()
    assert partrow["first_paper"] == 1 and partrow["last_paper"] == 1

    # missing file is a no-op, does not raise
    _load_titles(tmp_path / "nope.json", conn)


def test_db_check_embedder_match(tmp_path):
    db_path = _make_test_db(tmp_path)
    conn = dbmod.open_db(db_path)

    ok, msg = dbmod.check_embedder_match(conn, "m", "onnx")
    assert ok, msg

    ok, msg = dbmod.check_embedder_match(conn, "different-model", "onnx")
    assert not ok
    assert "model mismatch" in msg


def test_db_check_integrity_detects_mismatch(tmp_path):
    db_path = _make_test_db(tmp_path)
    conn = dbmod.open_db(db_path)
    conn.execute("DELETE FROM paragraphs_fts WHERE id = '1:1.1'")
    conn.commit()
    ok, msg = dbmod.check_integrity(conn)
    assert not ok


# --------------------------------------------------------------------------
# search: FTS escaping
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    [
        '"',
        "*",
        "-",
        "OR",
        "AND NOT",
        "thought AND adjuster",
        "мысль AND настройщик",
        'say "hello world" now',
        "a'b\"c*d-e",
        "",
        "   ",
        "()[]{}",
        "col:value",
    ],
)
def test_escape_fts_query_never_raises_and_produces_string(raw):
    result = _escape_fts_query(raw)
    assert isinstance(result, str)


def test_escape_fts_query_quotes_bare_words():
    q = _escape_fts_query("thought OR adjuster")
    assert q == '"thought" "adjuster"'  # OR is a stop word, never an operator


def test_escape_fts_query_phrase():
    q = _escape_fts_query('"thought adjuster"')
    assert q == '"thought adjuster"'


def test_searcher_fts_on_weird_input(tmp_path):
    db_path = _make_test_db(tmp_path)
    searcher = Searcher(db_path, embedder=None)
    for bad in ['"', "*", "-", "OR", "col:val", "", "adjuster*", "мысль\"'"]:
        results = searcher.fts(bad, k=5)
        assert isinstance(results, list)


def test_searcher_fts_finds_expected_fields(tmp_path):
    db_path = _make_test_db(tmp_path)
    searcher = Searcher(db_path, embedder=None)
    results = searcher.fts("thought adjuster", k=5, lang="en")
    assert len(results) >= 1
    r = results[0]
    for key in ("id", "paper", "section", "paragraph", "en", "ru", "score", "fts_rank", "sem_rank", "match_lang"):
        assert key in r
    assert r["id"] == "1:1.2"


def test_searcher_fts_ru_lemma(tmp_path):
    db_path = _make_test_db(tmp_path)
    searcher = Searcher(db_path, embedder=None)
    results = searcher.fts("второй", k=5, lang="ru")
    ids = [r["id"] for r in results]
    assert "1:1.2" in ids


# --------------------------------------------------------------------------
# FTS weights / titles / rebuild / snippets
# --------------------------------------------------------------------------

def _make_titled_db(tmp_path, old_schema=False) -> Path:
    db_path = tmp_path / ("old.db" if old_schema else "new.db")
    conn = dbmod.open_db(db_path)
    conn.execute("INSERT INTO papers (paper, en, ru, part) VALUES (1, 'Trinities', 'Троицы', NULL)")
    conn.execute("INSERT INTO sections (paper, section, en, ru) VALUES (1, 1, 'Deity', 'Божество')")
    rows = [
        ("1:1.1", "b1", 0, 1, 1, 1, "plain words about nothing", "простые слова", "простой слово"),
        ("1:1.2", "b1", 1, 1, 1, 2, "a <b>bold</b> & claim about trinities", "жирный", "жирный"),
        ("2:1.1", "b1", 2, 2, 1, 1, "trinities trinities trinities in text", "текст", "текст"),
    ]
    conn.executemany(
        "INSERT INTO paragraphs (id, book_id, seq, paper, section, paragraph, en, ru, ru_lemma) "
        "VALUES (?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    if old_schema:
        conn.execute("DROP TABLE paragraphs_fts")
        conn.execute("CREATE VIRTUAL TABLE paragraphs_fts USING fts5(id UNINDEXED, en, ru, ru_lemma, "
                     "tokenize='unicode61 remove_diacritics 2')")
        conn.executemany("INSERT INTO paragraphs_fts (id,en,ru,ru_lemma) VALUES (?,?,?,?)",
                         [(r[0], r[6], r[7], r[8]) for r in rows])
        conn.commit()
    conn.close()
    if not old_schema:
        from app.core.indexer import rebuild_fts
        rebuild_fts(db_path)
    return db_path


def test_rebuild_fts_adds_title_columns_and_keeps_embeddings(tmp_path):
    from app.core.indexer import rebuild_fts
    db = _make_titled_db(tmp_path, old_schema=True)
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO embeddings (para_id, lang, model, vector) VALUES ('1:1.1','en','m',x'00')")
    conn.commit(); conn.close()
    assert "title_en" not in dbmod.fts_columns(sqlite3.connect(db))
    n = rebuild_fts(db)
    assert n == 3
    conn = sqlite3.connect(db)
    assert dbmod.fts_columns(conn) == dbmod.FTS_COLUMNS
    row = conn.execute("SELECT title_en, title_ru FROM paragraphs_fts WHERE id='1:1.1'").fetchone()
    assert row[0] == "Trinities Deity"
    assert "Троицы" in row[1] and "троица" in row[1] and "божество" in row[1]
    assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 1


def test_rebuild_fts_cancel_leaves_old_index(tmp_path):
    import threading
    from app.core.indexer import rebuild_fts
    db = _make_titled_db(tmp_path, old_schema=True)
    ev = threading.Event(); ev.set()
    with pytest.raises(RuntimeError):
        rebuild_fts(db, cancel=ev)
    conn = sqlite3.connect(db)
    assert "title_en" not in dbmod.fts_columns(conn)
    assert conn.execute("SELECT COUNT(*) FROM paragraphs_fts").fetchone()[0] == 3


def test_fts_matches_titles_and_weights(tmp_path):
    db = _make_titled_db(tmp_path)
    s = Searcher(db, embedder=None, fts_weights={"title_en": 0.0, "title_ru": 0.0, "en": 1.0})
    # title-only match: 1:1.1 has 'Trinities' only in its title
    ids = [r["id"] for r in s.fts("trinities", lang="en")]
    assert "1:1.1" in ids
    # with title weight 0 the body-heavy paragraph wins; with big weight title wins
    assert ids[0] == "2:1.1"
    s.fts_weights = {"title_en": 50.0, "en": 1.0}
    assert s.fts("trinities", lang="en")[0]["id"] in ("1:1.1", "1:1.2")
    # titles disabled -> title-only paragraph disappears
    s.fts_titles = False
    assert "1:1.1" not in [r["id"] for r in s.fts("trinities", lang="en")]
    # ru side: raw and lemmatized title
    s.fts_titles = True
    assert "1:1.1" in [r["id"] for r in s.fts("Троицы", lang="ru")]
    assert "1:1.1" in [r["id"] for r in s.fts("троица", lang="ru")]


def test_searcher_old_schema_compat(tmp_path):
    db = _make_titled_db(tmp_path, old_schema=True)
    s = Searcher(db, embedder=None)
    ids = [r["id"] for r in s.fts("trinities", lang="en")]
    assert set(ids) == {"1:1.2", "2:1.1"}  # no title columns -> no title hit
    assert [r["id"] for r in s.fts("слова", lang="ru")] == ["1:1.1"]


def test_fts_snippets_escaped(tmp_path):
    db = _make_titled_db(tmp_path)
    s = Searcher(db, embedder=None)
    r = [x for x in s.fts("bold", lang="en") if x["id"] == "1:1.2"][0]
    sn = r["snippet_en"]
    assert "<mark>bold</mark>" in sn
    assert "&lt;b&gt;" in sn and "&amp;" in sn
    assert "<b>" not in sn
    assert "snippet_ru" in r


def test_fts_mode_fallback_and_hybrid_or(tmp_path):
    db = _make_titled_db(tmp_path)
    s = Searcher(db, embedder=None, fts_mode_hybrid="or", fts_mode_standalone="and_fallback_or")
    assert s.fts("plain zzzunknown", lang="en", mode="and") == []
    r = s.fts("plain zzzunknown", lang="en")
    assert [x["id"] for x in r] == ["1:1.1"] and s.last_timings["fts_mode_used"] == "or"
    s.fts("plain words", lang="en")
    assert s.last_timings["fts_mode_used"] == "and"
    # phrase stays exact in OR mode
    assert s.fts('"words plain"', lang="en", mode="or") == []
    assert s.hybrid("plain zzzunknown")[0]["id"] == "1:1.1"
    assert s.last_timings["fts_mode_used"] == "or"


def test_stop_words_filtered(tmp_path):
    from app.core.search import _query_lemmas
    assert _escape_fts_query("кто такие агондонтеры?", " OR ") == '"агондонтеры"'
    assert _escape_fts_query("What is the Paradise Trinity", " OR ") == '"Paradise" OR "Trinity"'
    assert _escape_fts_query("кто что") == '"кто" "что"'          # all stop -> original terms
    assert _escape_fts_query('"кто такой" Бог') == '"кто такой" "Бог"'  # quoted untouched
    assert "агондонтер" in _query_lemmas("кто такие агондонтеры")[0] or len(_query_lemmas("кто такие агондонтеры")) == 1
    db = _make_titled_db(tmp_path)
    s = Searcher(db, embedder=None)
    # stop-word-only overlap must not produce a hit
    assert s.fts("what is zzzunknown", lang="en", mode="or") == []
    r = s.fts("what is plain", lang="en", mode="or")
    assert [x["id"] for x in r] == ["1:1.1"]


def test_fts_stopwords_setting(tmp_path):
    db = _make_titled_db(tmp_path)
    s = Searcher(db, embedder=None, fts_stopwords=False)
    assert [x["id"] for x in s.fts("about zzzunknown", lang="en", mode="or")] != []
    s2 = Searcher(db, embedder=None, fts_stopwords=True)
    assert s2.fts("about zzzunknown", lang="en", mode="or") == []
    assert _escape_fts_query("Я ЕСТЬ", " OR ") == '"Я" OR "ЕСТЬ"'
