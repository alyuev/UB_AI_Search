import os
import sqlite3

from app.db_status import build_db_status
from tests.helpers import create_fake_book_db


def _full_db(path):
    """A complete, healthy book: titles, FTS with titles, vectors for every paragraph (scheme D)."""
    create_fake_book_db(path, book_id="b")
    c = sqlite3.connect(path)
    c.execute("UPDATE books SET embedding_model='bge-m3', embedding_runtime='llamacpp-vulkan', "
              "embedding_dim=4, index_scheme='D', vector_dtype='i8', created_at='2026-09-27T00:18:21'")
    c.execute("INSERT INTO papers (paper, en, ru, part) VALUES (147, 'Paper', 'Глава', 4)")
    c.execute("INSERT INTO sections (paper, section, en, ru) VALUES (147, 2, 'Sec', 'Раздел')")
    c.execute("DELETE FROM paragraphs_fts")
    c.execute("INSERT INTO paragraphs_fts (id, en, ru, ru_lemma, title_en, title_ru) "
              "SELECT id, en, ru, ru, 'Paper', 'Глава' FROM paragraphs")
    ids = [r[0] for r in c.execute("SELECT id FROM paragraphs")]
    for pid in ids:
        for lang in ("en", "ru"):
            c.execute("INSERT INTO embeddings (para_id, lang, model, vector, dtype, scale) VALUES (?,?,?,?,?,?)",
                      (pid, lang, "bge-m3", b"\x01\x02\x03\x04", "i8", 0.1))
    c.commit()
    c.close()


def _actions(st, with_glossary=False):
    """Action ids of the warnings (the optional «build glossary» hint is left out unless asked for)."""
    return {w["action"]["id"] for w in st["warnings"] if w["action"]
            and (with_glossary or w["action"]["id"] != "build_glossary")}


def _texts(st):
    return " | ".join(r["text"] for r in st["rows"])


def test_missing_db(tmp_path):
    st = build_db_status(str(tmp_path / "nope" / "book.db"), "urantia")
    assert st["state"] == "missing" and st["state_label"] == "не построена"
    assert st["exists"] is False and "ещё не построена" in st["summary"]
    assert _actions(st) == {"rebuild_all"}


def test_missing_glossary_is_a_soft_hint_not_a_problem(tmp_path):
    p = str(tmp_path / "book.db")
    _full_db(p)
    st = build_db_status(p, "b", embedder_key=("bge-m3", "llamacpp-vulkan"))
    assert st["state"] == "ready" and st["glossary"]["exists"] is False
    assert _actions(st, with_glossary=True) == {"build_glossary"}
    assert any(r["status"] == "warn" and "Словарь терминов" in r["text"] for r in st["rows"])


def test_full_db_is_ready(tmp_path):
    from app.core.terms import build_glossary

    p = str(tmp_path / "book.db")
    _full_db(p)
    build_glossary(p)
    st = build_db_status(p, "b", embedder_key=("bge-m3", "llamacpp-vulkan"),
                         fts_weights={"en": 1.0, "title_en": 2.0})
    assert st["state"] == "ready", st["warnings"]
    assert st["paragraphs"] == 5 and st["papers"] == 2 and st["vectors"]["count"] == 10
    assert st["vectors"]["expected"] == 10 and st["vectors"]["coverage_pct"] == 100.0
    assert st["vectors"]["dtype"] == "int8" and st["fts"]["has_title_columns"] is True
    assert st["integrity"]["ok"] is True and st["built_at"] == "27.09.2026 00:18"
    t = _texts(st)
    assert "Абзацев: 5" in t and "Векторы для поиска по смыслу: 10 из 10 (bge-m3, int8)" in t
    assert "с названиями глав" in t and "Словарь терминов (русские ↔ английские слова): есть" in t
    assert st["warnings"] == [] and st["glossary"]["exists"] is True


def test_old_fts_schema_recommends_rebuild_fts(tmp_path):
    p = str(tmp_path / "book.db")
    _full_db(p)
    c = sqlite3.connect(p)
    c.execute("DROP TABLE paragraphs_fts")
    c.execute("CREATE VIRTUAL TABLE paragraphs_fts USING fts5(id UNINDEXED, en, ru, ru_lemma)")
    c.execute("INSERT INTO paragraphs_fts (id, en, ru, ru_lemma) SELECT id, en, ru, ru FROM paragraphs")
    c.commit()
    c.close()
    st = build_db_status(p, "b", embedder_key=("bge-m3", "llamacpp-vulkan"))
    assert st["state"] == "attention"
    assert st["fts"]["has_title_columns"] is False
    assert "без названий глав" in _texts(st)
    assert _actions(st) == {"rebuild_fts"}


def test_fts_with_empty_titles_while_titles_exist(tmp_path):
    p = str(tmp_path / "book.db")
    _full_db(p)
    c = sqlite3.connect(p)
    c.execute("DELETE FROM paragraphs_fts")
    c.execute("INSERT INTO paragraphs_fts (id, en, ru, ru_lemma, title_en, title_ru) "
              "SELECT id, en, ru, ru, '', '' FROM paragraphs")
    c.commit()
    c.close()
    st = build_db_status(p, "b", embedder_key=("bge-m3", "llamacpp-vulkan"))
    assert st["state"] == "attention" and _actions(st) == {"rebuild_fts"}
    assert "пустые" in _texts(st)


def test_partial_and_missing_vectors(tmp_path):
    p = str(tmp_path / "book.db")
    _full_db(p)
    c = sqlite3.connect(p)
    c.execute("DELETE FROM embeddings WHERE lang='ru'")
    c.commit()
    c.close()
    st = build_db_status(p, "b", embedder_key=("bge-m3", "llamacpp-vulkan"))
    assert st["vectors"]["count"] == 5 and st["vectors"]["coverage_pct"] == 50.0
    assert st["state"] == "attention" and "rebuild_all" in _actions(st)
    c = sqlite3.connect(p)
    c.execute("DELETE FROM embeddings")
    c.commit()
    c.close()
    st = build_db_status(p, "b")
    assert "нет — работает только поиск по словам" in _texts(st)


def test_fts_row_count_mismatch(tmp_path):
    p = str(tmp_path / "book.db")
    _full_db(p)
    c = sqlite3.connect(p)
    c.execute("DELETE FROM paragraphs_fts WHERE id = '147.2.1'")
    c.commit()
    c.close()
    st = build_db_status(p, "b", embedder_key=("bge-m3", "llamacpp-vulkan"))
    assert "неполный (4 из 5" in _texts(st) and "rebuild_fts" in _actions(st)


def test_embedder_failure_is_reported(tmp_path):
    p = str(tmp_path / "book.db")
    _full_db(p)
    st = build_db_status(p, "b", embedder_error="llama-server did not start")
    assert st["state"] == "attention" and "open_embeddings" in _actions(st)
    assert "не запустилась" in _texts(st)


def test_garbage_file_is_reported_not_raised(tmp_path):
    p = tmp_path / "book.db"
    p.write_bytes(os.urandom(50000))
    st = build_db_status(str(p), "b")
    assert st["state"] == "attention" and "повреждена" in _texts(st)
    assert {"rebuild_all", "select_book"} <= _actions(st)


def test_busy_job_shown_in_summary(tmp_path):
    p = str(tmp_path / "book.db")
    _full_db(p)
    st = build_db_status(p, "b", embedder_key=("bge-m3", "llamacpp-vulkan"),
                         busy_job={"job_id": "x", "type": "rebuild-fts"})
    assert "перестроение индекса" in st["summary"]
