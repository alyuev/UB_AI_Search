"""Inline formatting of book paragraphs: parser markup, fmt column codec, old-database compatibility."""
import sqlite3
import zipfile

from app.core import db as dbmod
from app.core.parser import _rich_finish, load_book
from tests.test_book_parsers import make_epub

LONG = "слово " * 30


def _one(tmp_path, body, css=None, name="b.epub"):
    docs = {"ch": f"<h1>Глава</h1>{body}"}
    p = make_epub(tmp_path / name, docs, ["ch"],
                  extra_manifest='<item id="css" href="s.css" media-type="text/css"/>' if css else "")
    if css:
        with zipfile.ZipFile(p, "a") as z:
            z.writestr("OEBPS/s.css", css)
    return load_book(p).records


def test_epub_tags_mapped_and_others_stripped(tmp_path):
    recs = _one(tmp_path, f"<p>Это <em>важно</em> и <strong>очень</strong> <span class='x'>тут</span> "
                          f"<u>под</u> H<sub>2</sub>O x<sup>2</sup> &lt;tag&gt; &amp; {LONG}</p>")
    r = recs[0]
    assert r["ru"].startswith("Это важно и очень тут под H2O x2 <tag> &")
    assert r["fmt"].startswith("Это <i>важно</i> и <b>очень</b> тут под H<sub>2</sub>O x<sup>2</sup> &lt;tag&gt; &amp; ")
    assert "<span" not in r["fmt"] and "<u>" not in r["fmt"]


def test_epub_plain_paragraph_has_no_fmt(tmp_path):
    recs = _one(tmp_path, f"<p>Простой <span>текст</span> без форматирования {LONG}</p>")
    assert "fmt" not in recs[0]


def test_epub_footnote_marker_dropped_and_br_kept(tmp_path):
    recs = _one(tmp_path, f"<p>Строка один<sup><a href='#n1'>1</a></sup><br/>Строка <i>два</i> {LONG}</p>")
    r = recs[0]
    assert r["ru"].startswith("Строка один Строка два")
    assert r["fmt"].startswith("Строка один<br>Строка <i>два</i> ")
    assert "sup" not in r["fmt"]


def test_epub_class_based_italic_and_bold_from_css(tmp_path):
    css = ".italic {font-style: italic}\nspan.c2{font-weight:bold; color:red}\n.sm {font-size: 80%}\n"
    recs = _one(tmp_path, f"<p>Один <span class='italic'>курсив</span> и <span class='c2'>жирный</span> "
                          f"<span class='sm'>мелкий</span> {LONG}</p>"
                          f"<p class='italic'>Весь абзац курсивом {LONG}</p>", css)
    assert recs[0]["fmt"].startswith("Один <i>курсив</i> и <b>жирный</b> мелкий ")
    assert recs[1]["fmt"].startswith("<i>Весь абзац курсивом ") and recs[1]["fmt"].endswith("</i>")


def test_epub_nesting_across_blocks_is_balanced(tmp_path):
    recs = _one(tmp_path, f"<div><em>первый <b>абзац</b> {LONG}</em></div><p>обычный {LONG}</p>")
    assert recs[0]["fmt"].startswith("<i>первый <b>абзац</b> ")
    assert recs[0]["fmt"].count("<i>") == recs[0]["fmt"].count("</i>") == 1


def test_rich_finish_edge_cases():
    # unbalanced / crossed tags are repaired, markup never contains stray closers
    plain, fmt = _rich_finish([("c", "i"), "a ", ("o", "i"), "b ", ("o", "b"), "c", ("c", "i"), "d", ("c", "b")])
    assert plain == "a b cd"
    assert fmt == "a <i>b <b>c</b></i><b>d</b>"
    # nothing visible inside the tag: dropped; no formatting at all: None
    assert _rich_finish([("o", "i"), ("c", "i"), "просто текст"]) == ("просто текст", None)
    # text is escaped
    assert _rich_finish([("o", "b"), "a<b>&", ("c", "b")]) == ("a<b>&", "<b>a&lt;b&gt;&amp;</b>")


def test_fb2_emphasis_strong_and_poem(tmp_path):
    fb2 = (
        '<?xml version="1.0" encoding="utf-8"?><FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">'
        "<description><title-info><lang>ru</lang><book-title>T</book-title></title-info></description>"
        "<body><section><title><p>Глава</p></title>"
        f"<p>Это <emphasis>курсив</emphasis> и <strong>жирный</strong> текст {LONG}</p>"
        "<poem><stanza><v>Первая строка</v><v>Вторая <emphasis>строка</emphasis></v></stanza></poem>"
        f"<p>Обычный абзац {LONG}</p>"
        "</section></body></FictionBook>"
    )
    p = tmp_path / "b.fb2"
    p.write_text(fb2, encoding="utf-8")
    recs = load_book(p).records
    assert recs[0]["fmt"].startswith("Это <i>курсив</i> и <b>жирный</b> текст ")
    assert recs[1]["ru"] == "Первая строка Вторая строка"
    assert recs[1]["fmt"] == "Первая строка<br>Вторая <i>строка</i>"
    assert "fmt" not in recs[2]


# -------------------------------------------------------------------- storage --

def test_codec_roundtrip_and_prefix():
    assert dbmod.encode_fmt(None) is None and dbmod.encode_fmt("") is None
    short = "<i>a</i> б"
    assert dbmod.encode_fmt(short)[:1] == b"u" and dbmod.decode_fmt(dbmod.encode_fmt(short)) == short
    long = "Это <i>курсив</i> и <b>жирный</b> текст, повторяющийся. " * 40
    blob = dbmod.encode_fmt(long)
    assert blob[:1] == b"z" and len(blob) < len(long.encode("utf-8"))
    assert dbmod.decode_fmt(blob) == long
    assert dbmod.decode_fmt(b"?garbage") is None and dbmod.decode_fmt(b"z\x00bad") is None


def test_row_to_para_fields(tmp_path):
    conn = dbmod.open_db(tmp_path / "x.db")
    for pid, ru, en, fmt in [("1:0.1", "Текст курсив", "", "Текст <i>курсив</i>"),
                             ("1:0.2", "", "Some bold", "Some <b>bold</b>"), ("1:0.3", "Просто", "", None)]:
        conn.execute("INSERT INTO paragraphs (id, book_id, seq, paper, section, paragraph, en, ru, fmt) "
                     "VALUES (?, 'x', ?, 1, 0, 1, ?, ?, ?)", (pid, int(pid[-1]), en, ru, dbmod.encode_fmt(fmt)))
    conn.commit()
    a, b, c = (dbmod.get_paragraph(conn, i) for i in ("1:0.1", "1:0.2", "1:0.3"))
    assert (a["ru_html"], a["en_html"]) == ("Текст <i>курсив</i>", None)
    assert (b["ru_html"], b["en_html"]) == (None, "Some <b>bold</b>")
    assert (c["ru_html"], c["en_html"]) == (None, None)


def test_old_database_without_fmt_column_opens_readonly(tmp_path):
    path = tmp_path / "old.db"
    c = sqlite3.connect(path)
    c.executescript("""
        CREATE TABLE papers (paper INTEGER, en TEXT, ru TEXT, part INTEGER);
        CREATE TABLE sections (paper INTEGER, section INTEGER, en TEXT, ru TEXT);
        CREATE TABLE paragraphs (id TEXT PRIMARY KEY, book_id TEXT, seq INTEGER, paper INTEGER, section INTEGER,
            paragraph INTEGER, en TEXT, ru TEXT, ru_lemma TEXT, en_lemma TEXT, chunk_id TEXT);
        INSERT INTO paragraphs (id, book_id, seq, paper, section, paragraph, en, ru) VALUES ('1:0.1','o',0,1,0,1,'','Старый');
    """)
    c.commit()
    c.close()
    conn = dbmod.open_db(path, readonly=True)
    p = dbmod.get_paragraph(conn, "1:0.1")
    assert p["ru"] == "Старый" and p["ru_html"] is None and p["en_html"] is None
    assert dbmod.get_context(conn, "1:0.1")[0]["ru_html"] is None
    assert "fmt" not in [r[1] for r in conn.execute("PRAGMA table_info(paragraphs)")]  # no ALTER on open
