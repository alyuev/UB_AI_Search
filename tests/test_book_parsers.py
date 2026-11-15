"""EPUB / FB2 parsers (synthetic books built in tmp_path) and monolingual handling downstream."""
import sqlite3
import zipfile

import pytest

from app.core import db as dbmod
from app.core.parser import (
    load_book, parse_book, parse_epub, parse_fb2, parse_json, read_book_meta,
)

# ------------------------------------------------------------------ helpers --

CONTAINER = (
    '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
    '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
    "</rootfiles></container>"
)


def _xhtml(body: str, title: str = "T") -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml">'
        f"<head><title>{title}</title></head><body>{body}</body></html>"
    )


def make_epub(path, docs: dict, spine: list, lang="ru", title="Тест", author="Автор", extra_manifest=""):
    items = "".join(f'<item id="{k}" href="{k}.xhtml" media-type="application/xhtml+xml"/>' for k in docs)
    refs = "".join(f'<itemref idref="{k}"/>' for k in spine)
    opf = (
        '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="2.0">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f"<dc:title>{title}</dc:title><dc:creator>{author}</dc:creator><dc:language>{lang}</dc:language>"
        f"</metadata><manifest>{items}{extra_manifest}</manifest><spine>{refs}</spine></package>"
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml", CONTAINER)
        z.writestr("OEBPS/content.opf", opf)
        for k, body in docs.items():
            z.writestr(f"OEBPS/{k}.xhtml", _xhtml(body))
    return path


RU1 = "Первый абзац главы с достаточным количеством слов для проверки."
RU2 = "Второй абзац главы, в нём тоже есть слова."


# --------------------------------------------------------------------- EPUB --

def test_epub_structure_order_and_skips(tmp_path):
    docs = {
        "cover": '<div><img src="c.jpg"/></div>',
        "toc": "<h1>Содержание</h1><p>Глава 1<a href='ch1.xhtml'>1</a></p>",
        "ch2": f"<h1>Вторая</h1><p>{RU2}</p>",          # listed AFTER ch1 in the spine
        "ch1": (
            "<h1>Первая</h1>"
            f"<p>Вступление<a href='#n1' epub:type='noteref'>[1]</a> {RU1}</p>"
            "<h2>Раздел А</h2>"
            f"<div class='paragraph'><p>{RU1} А1.</p><p>{RU2} А2.</p></div>"
            "<blockquote>Цитата внутри раздела А достаточно длинная.</blockquote>"
            "<ul><li>Пункт списка один.</li><li>Пункт списка два.</li></ul>"
            "<p>&#160;</p><p>* * *</p><p>12</p>"
            "<h3>Раздел Б</h3>"
            f"<p>{RU2} Б1.</p>"
            '<aside epub:type="footnote"><p>Текст сноски не должен попасть.</p></aside>'
        ),
        "notes": "<h1>Примечания</h1><p>Примечание 1. Текст примечания.</p>",
    }
    p = make_epub(tmp_path / "b.epub", docs, ["cover", "toc", "ch1", "ch2", "notes"])
    book = load_book(p)
    recs = book.records
    assert book.lang == "ru"
    assert all(r["en"] == "" and r["ru"] for r in recs)
    assert len({r["id"] for r in recs}) == len(recs)
    assert [r["id"] for r in recs] == [
        "1:0.1", "1:1.1", "1:1.2", "1:1.3", "1:1.4", "1:1.5", "1:2.1", "2:0.1"]
    texts = [r["ru"] for r in recs]
    assert not any("сноски" in t or "Примечание" in t or "Содержание" in t for t in texts)
    assert texts[0] == "Вступление " + RU1          # noteref link dropped
    assert "* * *" not in texts and "12" not in texts and "" not in texts
    assert book.titles["papers"]["1"]["ru"] == "Первая"
    assert book.titles["papers"]["2"]["ru"] == "Вторая"
    assert book.titles["papers"]["1"]["sections"] == {
        "1": {"en": None, "ru": "Раздел А"}, "2": {"en": None, "ru": "Раздел Б"}}
    # parse_epub / parse_book yield the same records
    assert list(parse_epub(p)) == recs == list(parse_book(p))


def test_epub_language_en_and_meta(tmp_path):
    docs = {"c1": "<h1>One</h1><p>The first paragraph of the book.</p><p>Second paragraph here.</p>"}
    p = make_epub(tmp_path / "e.epub", docs, ["c1"], lang="en-US", title="Book", author="Some Author")
    book = load_book(p)
    assert book.lang == "en"
    assert book.records[0]["ru"] == "" and book.records[0]["en"].startswith("The first")
    assert book.titles["papers"]["1"]["en"] == "One" and book.titles["papers"]["1"]["ru"] is None
    assert read_book_meta(p) == {"title": "Book", "author": "Some Author", "language": "en"}


def test_epub_split_documents_continue_chapter(tmp_path):
    docs = {
        "a": f"<h1>Глава</h1><p>{RU1}</p>",
        "b": f"<p>{RU2}</p>",          # size-split continuation, no heading, no TOC entry
        "c": f"<h1>Другая</h1><p>{RU1}</p>",
    }
    recs = list(parse_epub(make_epub(tmp_path / "s.epub", docs, ["a", "b", "c"])))
    assert [r["id"] for r in recs] == ["1:0.1", "1:0.2", "2:0.1"]


def test_epub_toc_by_content_and_link_only_blocks(tmp_path):
    docs = {
        "idx": "<h2>Что внутри</h2>" + "".join(f"<p>Глава номер {i}{i * 3}</p>" for i in range(1, 6)),
        "nav": "<p><a href='x.xhtml'>Ссылка на главу</a></p><p><a href='y.xhtml'>Другая ссылка</a></p>",
        "ch": f"<h1>Текст</h1><p>{RU1}</p><p><a href='x.xhtml'>Ссылка на главу</a></p>",
    }
    recs = list(parse_epub(make_epub(tmp_path / "t.epub", docs, ["idx", "nav", "ch"])))
    assert [r["ru"] for r in recs] == [RU1]


def test_epub_without_text_gives_no_records(tmp_path):
    p = make_epub(tmp_path / "x.epub", {"c": "<h1>Пусто</h1>"}, ["c"])
    assert list(parse_epub(p)) == []


def test_epub_heading_lines_without_markup(tmp_path):
    docs = {"c": "<div>Глава Первая</div><div>Начало пути</div>"
                 f"<div>{RU1}</div><div>Глава II</div><div>{RU2}</div>"}
    book = load_book(make_epub(tmp_path / "h.epub", docs, ["c"]))
    assert [r["id"] for r in book.records] == ["1:0.1", "2:0.1"]
    assert book.titles["papers"]["1"]["ru"] == "Глава Первая — Начало пути"


def test_unsupported_extension(tmp_path):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF")
    with pytest.raises(ValueError):
        list(parse_book(p))


# ---------------------------------------------------------------------- FB2 --

FB2 = """<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0" xmlns:l="http://www.w3.org/1999/xlink">
 <description><title-info>
  <author><first-name>Иван</first-name><last-name>Петров</last-name></author>
  <book-title>Малая книга</book-title><lang>ru</lang>
 </title-info></description>
 <body>
  <title><p>Малая книга</p></title>
  <section><title><p>Часть первая</p></title>
   <section><title><p>Глава 1</p></title>
    <epigraph><p>Эпиграф к первой главе.</p><text-author>Кто-то</text-author></epigraph>
    <p>Первый абзац<a l:href="#n1" type="note">[1]</a> первой главы.</p>
    <subtitle>Подраздел</subtitle>
    <p>Абзац <strong>с выделением</strong> и <emphasis>курсивом</emphasis>.</p>
    <empty-line/>
    <poem><stanza><v>Строка стиха раз,</v><v>строка стиха два.</v></stanza></poem>
   </section>
   <section><title><p>Глава 2</p></title><p>Единственный абзац второй главы.</p></section>
  </section>
  <section><title><p>Плоская глава</p></title><p>Абзац плоской главы.</p></section>
 </body>
 <body name="notes"><section><title><p>1</p></title><p>Текст сноски не нужен.</p></section></body>
 <binary id="c.jpg" content-type="image/jpeg">AAAA</binary>
</FictionBook>
"""


def test_fb2_structure(tmp_path):
    p = tmp_path / "b.fb2"
    p.write_text(FB2, encoding="utf-8")
    book = load_book(p)
    assert book.lang == "ru"
    assert read_book_meta(p) == {"title": "Малая книга", "author": "Иван Петров", "language": "ru"}
    ids = [r["id"] for r in book.records]
    assert ids == ["1:0.1", "1:0.2", "1:1.1", "1:1.2", "2:0.1", "3:0.1"]
    by = {r["id"]: r["ru"] for r in book.records}
    assert by["1:0.1"] == "Эпиграф к первой главе."          # text-author is not a paragraph
    assert by["1:0.2"] == "Первый абзац первой главы."        # footnote marker dropped
    assert by["1:1.1"] == "Абзац с выделением и курсивом."
    assert by["1:1.2"] == "Строка стиха раз, строка стиха два."
    assert all(r["en"] == "" for r in book.records)
    assert not any("сноски" in r["ru"] for r in book.records)
    t = book.titles
    assert [t["papers"][str(i)]["ru"] for i in (1, 2, 3)] == ["Глава 1", "Глава 2", "Плоская глава"]
    assert t["papers"]["1"]["sections"] == {"1": {"en": None, "ru": "Подраздел"}}
    assert t["parts"][0]["ru"] == "Часть первая" and t["parts"][0]["papers"] == [1, 2]
    assert t["papers"]["1"]["part"] == 1 and t["papers"]["3"]["part"] is None
    assert list(parse_fb2(p)) == book.records


def test_fb2_cp1251_english_and_zip(tmp_path):
    xml = ('<?xml version="1.0" encoding="windows-1251"?><FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">'
           "<description><title-info><lang>en</lang><book-title>Eng</book-title></title-info></description>"
           "<body><section><title><p>Chapter</p></title><p>Hello there, reader of this tale.</p></section></body></FictionBook>")
    z = tmp_path / "b.fb2.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("b.fb2", xml.encode("cp1251"))
    book = load_book(z)
    assert book.lang == "en"
    assert book.records == [{"id": "1:0.1", "paper": 1, "section": 0, "paragraph": 1,
                             "en": "Hello there, reader of this tale.", "ru": ""}]
    cp = tmp_path / "ru.fb2"
    cp.write_bytes(xml.replace("Hello there, reader of this tale.", "Привет, читатель этой сказки.")
                   .replace("Chapter", "Глава").replace("<lang>en", "<lang>ru").encode("cp1251"))
    rb = load_book(cp)
    assert rb.lang == "ru" and rb.records[0]["ru"].startswith("Привет")


# -------------------------------------------- monolingual downstream checks --

def test_json_dispatch_unchanged(tmp_path):
    import json
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"id": "1:1.1", "paper": 1, "section": 1, "paragraph": 1, "en": "hi", "ru": "привет"}]),
                 encoding="utf-8")
    assert list(parse_book(p)) == list(parse_json(p))
    assert load_book(p).lang is None and load_book(p).titles is None


def test_validator_accepts_one_empty_language(tmp_path):
    import json
    p = tmp_path / "m.json"
    p.write_text(json.dumps([
        {"id": "1:0.1", "paper": 1, "section": 0, "paragraph": 1, "en": "", "ru": "Только русский"},
        {"id": "1:0.2", "paper": 1, "section": 0, "paragraph": 2, "en": "", "ru": ""},
    ]), encoding="utf-8")
    assert [r["id"] for r in parse_json(p)] == ["1:0.1"]


def test_monolingual_build_and_search(tmp_path, monkeypatch):
    """build_index on an EPUB with a fake embedder: only non-empty text is embedded, books row carries
    the language, Searcher works with a missing language (no en vectors)."""
    import numpy as np
    from app.core import indexer
    from app.core.search import Searcher

    docs = {"c": f"<h1>Глава</h1><p>{RU1}</p><h2>Раздел</h2><p>{RU2}</p>"}
    src = make_epub(tmp_path / "m.epub", docs, ["c"])
    embedded: list[str] = []

    class FakeEmb:
        dim = 8
        model_name = "fake"
        runtime = "onnx"

        def _vec(self, t):
            v = np.zeros(8, dtype=np.float32)
            v[len(t) % 8] = 1.0
            return v

        def embed_passages(self, texts, batch_size=32):
            embedded.extend(texts)
            return np.stack([self._vec(t) for t in texts])

        def embed_query(self, t):
            return self._vec(t)

    monkeypatch.setattr(indexer, "create_embedder", lambda *a, **k: FakeEmb())
    monkeypatch.setattr(indexer, "DATA_DIR", tmp_path)
    db_path = tmp_path / "lib" / "book.db"
    indexer.build_index(src, db_path, "fake", book_id="mono", book_name="Mono", lower_priority=False)

    assert sorted(embedded) == sorted([RU1, RU2]) and "" not in embedded
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    info = dbmod.get_book_info(conn, "mono")
    assert info["language_source"] == "ru" and info["language_translation"] == ""
    assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 2
    assert {r[0] for r in conn.execute("SELECT lang FROM embeddings")} == {"ru"}
    assert conn.execute("SELECT ru FROM sections WHERE paper=1 AND section=1").fetchone()[0] == "Раздел"
    conn.close()

    s = Searcher(db_path, embedder=FakeEmb())
    res = s.semantic("первый абзац", k=5)
    assert len(res) == 2 and all(r["match_lang"] == "ru" for r in res)
    assert s.fts("абзац", k=5)
    assert s.hybrid("абзац", k=5)
