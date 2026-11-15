"""Parsers for the source bilingual corpus (JSON and MD variants).

Both parsers are generators: they never load the whole source into a single
in-memory structure beyond what the underlying `json.load` (JSON) or a
line-by-line scan (MD) requires. For the JSON case the file is one big JSON
array so a true streaming parse would require ijson; since the app's own
records must be validated one-by-one anyway, we use the stdlib json module but
immediately discard record fields we don't need and iterate one record at a
time.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator, Optional

from app.log import get_logger

logger = get_logger("index")

REQUIRED_FIELDS = ("id", "paper", "section", "paragraph", "en", "ru")

_ID_RE = re.compile(r"^(\d+):(\d+)\.(\d+)$")


def _validate_record(rec: dict, idx: int) -> Optional[dict]:
    missing = [f for f in REQUIRED_FIELDS if f not in rec]
    if missing:
        logger.warning("record #%d missing fields %s: %r", idx, missing, rec.get("id"))
        return None

    rid = rec.get("id")
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        logger.warning("record #%d has malformed id %r", idx, rid)
        return None

    try:
        paper = int(rec["paper"])
        section = int(rec["section"])
        paragraph = int(rec["paragraph"])
    except (TypeError, ValueError):
        logger.warning("record #%d has non-integer paper/section/paragraph: %r", idx, rid)
        return None

    en = rec.get("en")
    ru = rec.get("ru")
    if not isinstance(en, str) or not isinstance(ru, str):
        logger.warning("record #%d has non-string en/ru: %r", idx, rid)
        return None
    if not en.strip() and not ru.strip():
        logger.warning("record #%d has empty en and ru: %r", idx, rid)
        return None

    return {
        "id": rid,
        "paper": paper,
        "section": section,
        "paragraph": paragraph,
        "en": en,
        "ru": ru,
    }


def parse_json(path: str | Path) -> Iterator[dict]:
    """Yield validated {id,paper,section,paragraph,en,ru} dicts from the
    bilingual JSON corpus. Bad records are logged and skipped, not raised.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a top-level JSON array")

    n_ok, n_bad = 0, 0
    for idx, rec in enumerate(data):
        if not isinstance(rec, dict):
            n_bad += 1
            logger.warning("record #%d is not an object: %r", idx, type(rec))
            continue
        cleaned = _validate_record(rec, idx)
        if cleaned is None:
            n_bad += 1
            continue
        n_ok += 1
        yield cleaned

    logger.info("parse_json(%s): %d ok, %d bad out of %d", path, n_ok, n_bad, len(data))


# --- MD format -------------------------------------------------------------
# Blocks look like:
# ID: 147:2.15
# EN: <english text, possibly multi-line>
# RU: <russian text, possibly multi-line>
# ---

_ID_LINE = re.compile(r"^ID:\s*(\S+)\s*$")
_EN_LINE = re.compile(r"^EN:\s*(.*)$")
_RU_LINE = re.compile(r"^RU:\s*(.*)$")
_SEP_LINE = re.compile(r"^-{3,}\s*$")


def parse_md(path: str | Path) -> Iterator[dict]:
    """Yield validated dicts from the MD corpus format (ID:/EN:/RU:/--- blocks)."""
    path = Path(path)

    cur: dict | None = None
    field: str | None = None
    idx = 0
    n_ok, n_bad = 0, 0

    def flush():
        nonlocal n_ok, n_bad
        if cur is None:
            return None
        rec = {
            "id": cur.get("id"),
            "paper": None,
            "section": None,
            "paragraph": None,
            "en": "\n".join(cur.get("en", [])).strip(),
            "ru": "\n".join(cur.get("ru", [])).strip(),
        }
        rid = rec["id"]
        m = _ID_RE.match(rid) if isinstance(rid, str) else None
        if not m:
            n_bad += 1
            logger.warning("md record has malformed id %r", rid)
            return None
        rec["paper"] = int(m.group(1))
        rec["section"] = int(m.group(2))
        rec["paragraph"] = int(m.group(3))
        if not rec["en"] and not rec["ru"]:
            n_bad += 1
            logger.warning("md record %r has empty en and ru", rid)
            return None
        n_ok += 1
        return rec

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            idx += 1
            if _SEP_LINE.match(line):
                rec = flush()
                cur = None
                field = None
                if rec is not None:
                    yield rec
                continue

            m_id = _ID_LINE.match(line)
            if m_id:
                # start of a new block implicitly closes an unterminated one
                if cur is not None:
                    rec = flush()
                    if rec is not None:
                        yield rec
                cur = {"id": m_id.group(1), "en": [], "ru": []}
                field = None
                continue

            m_en = _EN_LINE.match(line)
            if m_en is not None and cur is not None:
                field = "en"
                cur["en"].append(m_en.group(1))
                continue

            m_ru = _RU_LINE.match(line)
            if m_ru is not None and cur is not None:
                field = "ru"
                cur["ru"].append(m_ru.group(1))
                continue

            if cur is not None and field in ("en", "ru") and line.strip():
                cur[field].append(line)

    rec = flush()
    if rec is not None:
        yield rec

    logger.info("parse_md(%s): %d ok, %d bad", path, n_ok, n_bad)


# =============================================================================
# Ordinary monolingual books: EPUB and FB2
# =============================================================================
#
# Both parsers produce the same record shape as parse_json ({id "P:S.N", paper, section, paragraph,
# en, ru}) with the text in ONE language field (ru for a Cyrillic book, otherwise en) and the other
# field "". Mapping: paper = chapter (a spine document / h1 / top-level FB2 section), section = a
# sub-heading inside the chapter (0 = paragraphs before the first sub-heading), paragraph = one
# block of text. Headings are not paragraphs; they become the book's titles (same layout as
# data/titles/<book_id>_titles.json), available through load_book().titles.

import html as _html_mod
import posixpath
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import unquote

BOOK_EXTENSIONS = (".json", ".md", ".epub", ".fb2", ".fb2.zip")

_WS_RE = re.compile(r"[\s  -   　]+")
_ZW_RE = re.compile(r"[­​-‍⁠﻿]")
_HAS_ALNUM_RE = re.compile(r"[^\W_]", re.UNICODE)
_PAGE_NUM_RE = re.compile(r"^[\[\(]?\s*\d{1,4}\s*[\]\)]?$")
_MARKER_RE = re.compile(r"^[\[\(\{]?\s*(\d{1,3}|[*†‡§#]+|[a-zA-Zа-яА-Я])\s*[\]\)\}]?\*?$")
_TOC_LINE_RE = re.compile(r"[^\W\d_][\s.…·_-]*\d{1,4}$")

# Documents whose LEADING heading (normalized) marks them as non-text: dropped as a whole ...
_DROP_DOC_TITLES = {
    "содержание", "оглавление", "contents", "table of contents", "toc", "примечания", "комментарии",
    "сноски", "notes", "footnotes", "endnotes", "bibliography", "библиография",
}
# ... or as front matter: only the heading and the text up to the next heading are dropped.
_FRONT_TITLES = {
    "annotation", "аннотация", "cover", "обложка", "титульный лист", "title page", "titlepage",
    "copyright", "от издателя", "об авторе", "about the author", "выходные данные",
}
# Short stand-alone line that is really a heading in a book converted without heading markup.
_HEADING_LINE_RE = re.compile(
    r"^(глава|часть|книга|раздел|chapter|part|book|section|пролог|эпилог|prologue|epilogue)(?![^\W\d_])", re.I)
_SKIP_NAME_RE = re.compile(
    r"^(?:(cover|title[-_ ]?page|toc|nav|ncx|copyright|annotation|notes?|footnotes?|"
    r"endnotes?|comments?|colophon)([\W\d_]|$)"
    # «contents.html» is a TOC, but «content0.html», «content1.html» hold the text itself (LitRes / fb2 converters)
    r"|contents?([\W_]|$))", re.I)
_SKIP_GUIDE_TYPES = {"cover", "title-page", "toc", "loi", "lot", "copyright-page", "notes"}

_BLOCK_TAGS = {"p", "div", "blockquote", "li", "pre", "dd", "dt", "td", "th", "figcaption", "caption",
               "center", "address"}
_FLUSH_TAGS = _BLOCK_TAGS | {"ul", "ol", "dl", "table", "tbody", "thead", "tfoot", "tr", "section",
                             "article", "aside", "header", "footer", "figure", "body", "hr", "html",
                             "h1", "h2", "h3", "h4", "h5", "h6", "nav", "main", "details", "summary"}
_VOID_TAGS = {"br", "hr", "img", "meta", "link", "input", "col", "area", "base", "wbr", "source", "param"}
_SKIP_TAGS = {"script", "style", "head", "svg", "math", "nav", "title", "noscript", "template"}


def _norm_text(s: str) -> str:
    s = _ZW_RE.sub("", s or "")
    return _WS_RE.sub(" ", s).strip()


# ---------------------------------------------------- inline formatting ----
# Display-only light markup of a paragraph: <i> <b> <sup> <sub> <br> and nothing else. Search keeps
# using the plain text; the markup is built from the very same text pieces (see _rich_finish).

_FMT_TAGS = ("i", "b", "sup", "sub")
_EMPTY_PAIR_RE = re.compile(r"<(i|b|sup|sub)></\1>|</(i|b|sup|sub)><\2>")
_TAG_RE = re.compile(r"</?(?:i|b|sup|sub)>")
_ITALIC_RE = re.compile(r"font-style\s*:\s*(?:italic|oblique)", re.I)
_BOLD_RE = re.compile(r"font-weight\s*:\s*(?:bold|bolder|[6-9]00)\b", re.I)
_SUP_RE = re.compile(r"vertical-align\s*:\s*super", re.I)
_SUB_RE = re.compile(r"vertical-align\s*:\s*sub", re.I)
_CSS_RULE_RE = re.compile(r"([^{}@]+)\{([^{}]*)\}")
_CSS_SIMPLE_SEL_RE = re.compile(r"^[a-z][a-z0-9]*\.([\w-]+)$|^\.([\w-]+)$", re.I)
_TAG_MARKS = {"em": "i", "i": "i", "strong": "b", "b": "b", "sup": "sup", "sub": "sub"}


def _style_marks(decl: str) -> list[str]:
    out = []
    if _ITALIC_RE.search(decl):
        out.append("i")
    if _BOLD_RE.search(decl):
        out.append("b")
    if _SUP_RE.search(decl):
        out.append("sup")
    elif _SUB_RE.search(decl):
        out.append("sub")
    return out


def _css_class_marks(css_texts) -> dict[str, tuple]:
    """{class name: marks} from the EPUB style sheets: rules like `.italic {font-style: italic}` or
    `span.c3 {font-weight: bold}` (Calibre writes formatting as classes, not as <em>/<strong>)."""
    marks: dict[str, set] = {}
    for css in css_texts:
        css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
        for m in _CSS_RULE_RE.finditer(css):
            found = _style_marks(m.group(2))
            if not found:
                continue
            for sel in m.group(1).split(","):
                sm = _CSS_SIMPLE_SEL_RE.match(sel.strip())
                if sm:
                    marks.setdefault(sm.group(1) or sm.group(2), set()).update(found)
    return {k: tuple(sorted(v)) for k, v in marks.items()}


def _rich_finish(tokens: list) -> tuple[str, Optional[str]]:
    """tokens: text strings and ("o"|"c", name) / ("br", None) markers -> (plain text, markup or None).
    The plain text is normalized exactly like _norm_text; the markup is balanced, properly nested,
    escaped, and None when it carries no formatting at all."""
    out: list = []          # text pieces (list [escaped]) and tag strings
    plain: list[str] = []
    cnt = {n: 0 for n in _FMT_TAGS}
    stack: list[str] = []
    last_space = True
    last_text = -1          # index in `out` of the last text piece
    have_text = False
    for t in tokens:
        if isinstance(t, str):
            s = _WS_RE.sub(" ", _ZW_RE.sub("", t))
            if last_space and s.startswith(" "):
                s = s[1:]
            if not s:
                continue
            plain.append(s)
            out.append([_html_mod.escape(s, quote=False)])
            last_text = len(out) - 1
            last_space = s.endswith(" ")
            have_text = True
            continue
        kind, name = t
        if kind == "br":
            if not have_text:
                continue
            if last_text >= 0 and out[last_text][0].endswith(" "):
                out[last_text][0] = out[last_text][0][:-1]
            out.append("<br>")
            if not last_space:
                plain.append(" ")
            last_space = True
        elif kind == "o":
            cnt[name] += 1
            if cnt[name] == 1:
                stack.append(name)
                out.append(f"<{name}>")
        elif kind == "c" and cnt[name] > 0:
            cnt[name] -= 1
            if cnt[name] == 0:
                above = stack[stack.index(name) + 1:]
                out.extend(f"</{n}>" for n in reversed(above))
                out.append(f"</{name}>")
                out.extend(f"<{n}>" for n in above)
                stack.remove(name)
    for n in reversed(stack):
        out.append(f"</{n}>")
    # no trailing space / line breaks before the closing tags at the very end
    i = len(out) - 1
    while i >= 0:
        o = out[i]
        if isinstance(o, str) and o.startswith("</"):
            i -= 1
        elif o == "<br>":
            out[i] = ""
            i -= 1
        elif isinstance(o, list):   # the last text piece
            o[0] = o[0].rstrip(" ")
            break
        else:                       # an opening tag or an emptied <br>: look further back
            i -= 1
    markup = "".join(o[0] if isinstance(o, list) else o for o in out)
    text = _norm_text("".join(plain))
    prev = None
    while prev != markup:
        prev, markup = markup, _EMPTY_PAIR_RE.sub("", markup)
    markup = re.sub(r"(?:<br>)+(?=(?:</(?:i|b|sup|sub)>)*$)", "", markup)
    # safety net: the markup must reduce to exactly the plain text, otherwise show plain
    back = _norm_text(_html_mod.unescape(_TAG_RE.sub("", markup.replace("<br>", " "))))
    if back != text:
        return text, None
    return text, (markup if markup != _html_mod.escape(text, quote=False) else None)


def _local(tag) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1].lower()


def _norm_lang(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    c = code.strip().lower().replace("_", "-").split("-")[0]
    return {"rus": "ru", "eng": "en", "ukr": "uk", "deu": "de", "ger": "de", "fra": "fr", "fre": "fr"}.get(c, c) or None


def _decode_bytes(raw: bytes) -> str:
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", "replace")
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", "replace")
    head = raw[:2048].decode("ascii", "ignore")
    m = re.search(r"""(?:encoding|charset)\s*=\s*["']?\s*([A-Za-z0-9_.:-]+)""", head)
    enc = (m.group(1) if m else "utf-8").lower()
    for e in (enc, "utf-8", "cp1251"):
        try:
            return raw.decode(e)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", "replace")


def _text_field(samples, declared: Optional[str]) -> str:
    """Field the book text goes to: "ru" if Cyrillic letters dominate the text sample, else "en"
    (the generic non-Russian field). Falls back to the declared language when there is no text."""
    cyr = lat = 0
    for s in samples:
        for ch in s:
            if "Ѐ" <= ch <= "ӿ":
                cyr += 1
            elif ch.isalpha():
                lat += 1
        if cyr + lat > 20000:
            break
    if cyr + lat == 0:
        return "ru" if declared == "ru" else "en"
    return "ru" if cyr >= lat else "en"


# ---------------------------------------------------------------- builder ----

class _Builder:
    """Collects chapters/sections/paragraphs and turns them into records + titles."""

    def __init__(self) -> None:
        self.chapters: list[dict] = []
        self.parts: list[dict] = []
        self._part_idx: Optional[int] = None

    def new_part(self, title: Optional[str]) -> None:
        self.parts.append({"title": title, "chapters": []})
        self._part_idx = len(self.parts) - 1

    def end_part(self) -> None:
        self._part_idx = None

    def new_chapter(self, title: Optional[str]) -> None:
        ch = {"title": title or None, "sections": [{"title": None, "implicit": True, "paras": []}],
              "part": self._part_idx}
        self.chapters.append(ch)
        if self._part_idx is not None:
            self.parts[self._part_idx]["chapters"].append(len(self.chapters) - 1)

    @property
    def has_chapter(self) -> bool:
        return bool(self.chapters)

    @property
    def chapter_has_paragraphs(self) -> bool:
        return bool(self.chapters) and any(s["paras"] for s in self.chapters[-1]["sections"])

    def extend_chapter_title(self, title: str) -> None:
        ch = self.chapters[-1]
        ch["title"] = f"{ch['title']} — {title}" if ch["title"] else title

    def new_section(self, title: Optional[str]) -> None:
        if not self.chapters:
            self.new_chapter(None)
        secs = self.chapters[-1]["sections"]
        last = secs[-1]
        if not last["paras"]:
            # no text since the previous heading: stacked headings -> one section, joined title
            if last["implicit"]:
                last["implicit"] = False
                last["title"] = title or None
            elif title:
                last["title"] = f"{last['title']} — {title}" if last["title"] else title
            return
        secs.append({"title": title or None, "implicit": False, "paras": []})

    def add_paragraph(self, text: str, fmt: Optional[str] = None) -> None:
        if not self.chapters:
            self.new_chapter(None)
        self.chapters[-1]["sections"][-1]["paras"].append((text, fmt))

    def finish(self, lang: str):
        """-> (records, titles dict or None). Chapters without text are dropped; papers are
        numbered 1..N, sections 0 (before the first sub-heading) then 1..M."""
        fld = lang if lang in ("en", "ru") else "en"
        records: list[dict] = []
        papers: dict = {}
        paper_of_chapter: dict[int, int] = {}
        n_paper = 0
        for ci, ch in enumerate(self.chapters):
            if not any(s["paras"] for s in ch["sections"]):
                continue
            n_paper += 1
            paper_of_chapter[ci] = n_paper
            pinfo: dict = {"en": None, "ru": None, "part": None, "sections": {}}
            if ch["title"]:
                pinfo[fld] = ch["title"]
            sec_no = 0
            for s in ch["sections"]:
                if not s["paras"]:
                    continue
                if s["implicit"]:
                    no = 0
                else:
                    sec_no += 1
                    no = sec_no
                    if s["title"]:
                        pinfo["sections"][str(no)] = {"en": None, "ru": None, fld: s["title"]}
                for k, (text, fmt) in enumerate(s["paras"], 1):
                    rec = {"id": f"{n_paper}:{no}.{k}", "paper": n_paper, "section": no,
                           "paragraph": k, "en": "", "ru": ""}
                    rec[fld] = text
                    if fmt:
                        rec["fmt"] = fmt    # display markup of the text field (see _rich_finish)
                    records.append(rec)
            papers[str(n_paper)] = pinfo
        parts_out = []
        for part in self.parts:
            nums = [paper_of_chapter[c] for c in part["chapters"] if c in paper_of_chapter]
            if not nums:
                continue
            idx = len(parts_out) + 1
            parts_out.append({"part": idx, "en": None, "ru": None, fld: part["title"],
                              "papers": [min(nums), max(nums)]})
            for n in nums:
                papers[str(n)]["part"] = idx
        has_titles = any(p.get(fld) or p["sections"] for p in papers.values())
        titles = {"parts": parts_out, "papers": papers} if has_titles else None
        return records, titles


# ------------------------------------------------------------------- EPUB ----

class _XhtmlBlocks(HTMLParser):
    """Flattens an (X)HTML chapter into ordered blocks: ("h", level, text) / ("p", text, markup).

    Every block-level element closes the running text, so nested div/p structures produce leaf
    blocks only. Dropped on the way: head/script/style/nav/svg, footnote bodies (epub:type
    footnote/endnote, <aside>), footnote-marker links ([1], *, superscript numbers) and blocks
    consisting only of links (navigation / table of contents). `markup` is the display-only light
    markup (<i> <b> <sup> <sub> <br>) of the paragraph, None when it has no formatting; `css_marks`
    maps class names to marks for books that write italics as classes (Calibre)."""

    def __init__(self, css_marks: Optional[dict] = None) -> None:
        super().__init__(convert_charrefs=True)
        self.items: list[tuple] = []
        self._buf = ""
        self._toks: list = []           # text pieces and format markers, see _rich_finish
        self._link_chars = 0
        self._heading: Optional[int] = None
        self._skip_tag: Optional[str] = None
        self._skip_depth = 0
        self._links: list[tuple] = []   # (buf offset, href, noteref attr, inside <sup>, token index)
        self._sup = 0
        self._css = css_marks or {}
        self._stack: list[tuple] = []   # (tag, marks opened by this element)
        self._active = {n: 0 for n in _FMT_TAGS}

    def _skip_start(self, tag: str, attrs: dict) -> bool:
        if tag in _SKIP_TAGS or tag == "aside":
            return True
        typ = ((attrs.get("epub:type") or "") + " " + (attrs.get("role") or "")).lower()
        return any(t in typ for t in ("footnote", "endnote", "rearnote")) and "noteref" not in typ

    def _flush(self) -> None:
        text = _norm_text(self._buf)
        toks = self._toks
        link_chars, self._buf, self._link_chars = self._link_chars, "", 0
        # formatting still open (the block sits inside <i>, class="italic" ...) continues in the next block
        self._toks = [("o", n) for n in _FMT_TAGS if self._active[n] > 0]
        if not text or not _HAS_ALNUM_RE.search(text):
            return
        if self._heading is not None:
            self.items.append(("h", self._heading, text))
            return
        if _PAGE_NUM_RE.match(text):
            return
        if link_chars >= 0.9 * len(text):
            return  # navigation / TOC line
        plain, fmt = _rich_finish(toks)
        self.items.append(("p", text, fmt if plain == text else None))

    def _marks_of(self, tag: str, a: dict) -> list[str]:
        found = []
        if tag in _TAG_MARKS:
            found.append(_TAG_MARKS[tag])
        for c in (a.get("class") or "").split():
            found.extend(self._css.get(c, ()))
        if a.get("style"):
            found.extend(_style_marks(a["style"]))
        return list(dict.fromkeys(found))

    def _open(self, tag: str, a: dict) -> None:
        marks = self._marks_of(tag, a)
        for n in marks:
            self._active[n] += 1
            self._toks.append(("o", n))
        self._stack.append((tag, marks))

    def _close(self, tag: str) -> None:
        for k in range(len(self._stack) - 1, -1, -1):
            if self._stack[k][0] == tag:
                break
        else:
            return
        while len(self._stack) > k:
            _, marks = self._stack.pop()
            for n in reversed(marks):
                self._active[n] -= 1
                self._toks.append(("c", n))

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        a = {k.lower(): (v or "") for k, v in attrs}
        if self._skip_tag is not None:
            if tag == self._skip_tag and tag not in _VOID_TAGS:
                self._skip_depth += 1
            return
        if self._skip_start(tag, a):
            self._flush()
            self._skip_tag, self._skip_depth = tag, 1
            return
        if tag == "br":
            self._buf += " "
            self._toks.append(("br", None))
            return
        if tag in _VOID_TAGS:
            if tag in _FLUSH_TAGS:
                self._flush()
            return
        if tag in _FLUSH_TAGS:
            self._flush()
            if len(tag) == 2 and tag[0] == "h" and tag[1] in "123456":
                self._heading = int(tag[1])
            self._open(tag, a)
            return
        self._open(tag, a)
        if tag == "sup":
            self._sup += 1
        elif tag == "a":
            typ = ((a.get("epub:type") or "") + " " + (a.get("role") or "")).lower()
            self._links.append((len(self._buf), a.get("href", ""), "noteref" in typ, self._sup > 0,
                                len(self._toks)))

    def handle_endtag(self, tag):
        tag = tag.lower()
        if self._skip_tag is not None:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if self._skip_depth <= 0:
                    self._skip_tag = None
            return
        if tag in _VOID_TAGS and tag not in _FLUSH_TAGS:
            return
        if tag in _FLUSH_TAGS:
            self._flush()
            self._close(tag)
            if self._heading is not None and tag == f"h{self._heading}":
                self._heading = None
            return
        if tag == "a" and self._links:
            start, href, noteref, in_sup, ntok = self._links.pop()
            if start <= len(self._buf):
                marker = _norm_text(self._buf[start:])
                if noteref or ("#" in href and (_MARKER_RE.match(marker) or (in_sup and len(marker) <= 6))):
                    self._buf = self._buf[:start]  # footnote marker: drop
                    del self._toks[ntok:]
                elif href and marker:
                    self._link_chars += len(marker)
        self._close(tag)
        if tag == "sup":
            self._sup = max(0, self._sup - 1)

    def handle_data(self, data):
        if self._skip_tag is None:
            self._buf += data
            self._toks.append(data)

    def close(self):
        super().close()
        self._flush()


def _xhtml_blocks(raw: bytes, css_marks: Optional[dict] = None) -> list[tuple]:
    p = _XhtmlBlocks(css_marks)
    p.feed(_decode_bytes(raw))
    p.close()
    return p.items


def _zip_lookup(z: zipfile.ZipFile) -> dict[str, str]:
    return {n.lower(): n for n in z.namelist()}


def _zread(z: zipfile.ZipFile, lookup: dict[str, str], name: str) -> Optional[bytes]:
    real = lookup.get(posixpath.normpath(unquote(name)).lower())
    return z.read(real) if real else None


def _opf_path(z: zipfile.ZipFile) -> str:
    try:
        root = ET.fromstring(z.read("META-INF/container.xml"))
        for el in root.iter():
            if _local(el.tag) == "rootfile" and el.get("full-path"):
                return el.get("full-path")
    except (KeyError, ET.ParseError):
        pass
    for n in z.namelist():
        if n.lower().endswith(".opf"):
            return n
    raise ValueError("EPUB without an OPF package document")


def _parse_opf(raw: bytes) -> dict:
    root = ET.fromstring(raw)
    meta = {"title": None, "author": None, "language": None}
    authors: list[str] = []
    manifest: dict[str, dict] = {}
    spine: list[tuple[str, bool]] = []
    guide: list[tuple[str, str]] = []
    for el in root.iter():
        t = _local(el.tag)
        txt = _norm_text("".join(el.itertext())) if t in ("title", "creator", "language") else ""
        if t == "title" and txt and not meta["title"]:
            meta["title"] = txt
        elif t == "creator" and txt:
            authors.append(txt)
        elif t == "language" and txt and not meta["language"]:
            meta["language"] = _norm_lang(txt)
        elif t == "item" and el.get("id"):
            manifest[el.get("id")] = {"href": el.get("href", ""), "type": el.get("media-type", ""),
                                      "props": el.get("properties", "")}
        elif t == "itemref" and el.get("idref"):
            spine.append((el.get("idref"), el.get("linear", "yes").lower() != "no"))
        elif t == "reference" and el.get("href"):
            guide.append(((el.get("type") or "").lower(), el.get("href")))
    meta["author"] = ", ".join(dict.fromkeys(authors)) or None
    return {"meta": meta, "manifest": manifest, "spine": spine, "guide": guide}


def _toc_labels(z, lookup, opf_dir: str, manifest: dict) -> dict[str, str]:
    """doc path (lowercase, no fragment) -> first TOC label, from NCX or the EPUB3 nav document."""
    out: dict[str, str] = {}

    def add(href: str, label: str, base: str) -> None:
        label = _norm_text(label)
        if not href or not label:
            return
        path = posixpath.normpath(posixpath.join(base, unquote(href.split("#")[0]))).lower()
        out.setdefault(path, label)

    for item in manifest.values():
        full = posixpath.join(opf_dir, item["href"])
        base = posixpath.dirname(full)
        if item["type"] == "application/x-dtbncx+xml":
            raw = _zread(z, lookup, full)
            if raw is None:
                continue
            try:
                root = ET.fromstring(raw)
            except ET.ParseError:
                continue
            for np_ in root.iter():
                if _local(np_.tag) == "navpoint":
                    label = src = None
                    for ch in np_:
                        if _local(ch.tag) == "navlabel":
                            label = "".join(ch.itertext())
                        elif _local(ch.tag) == "content":
                            src = ch.get("src")
                    add(src or "", label or "", base)
        elif "nav" in item["props"].split():
            raw = _zread(z, lookup, full)
            if raw is None:
                continue
            for m in re.finditer(r"""<a\b[^>]*href\s*=\s*["']([^"']+)["'][^>]*>(.*?)</a>""",
                                 _decode_bytes(raw), re.S | re.I):
                add(m.group(1), re.sub(r"<[^>]+>", "", _html_mod.unescape(m.group(2))), base)
    return out


def _looks_like_toc(items: list[tuple]) -> bool:
    paras = [i[1] for i in items if i[0] == "p"]
    if len(paras) < 3:
        return False
    hits = sum(1 for t in paras if len(t) < 100 and _TOC_LINE_RE.search(t))
    return hits >= 0.8 * len(paras)


def _drop_front_matter(items: list[tuple]) -> Optional[list[tuple]]:
    """Apply the leading-heading skip rules; None = drop the whole document."""
    first = next((k for k, i in enumerate(items) if i[0] == "h"), None)
    if first is None or any(i[0] == "p" for i in items[:first]):
        return items                     # no heading, or it is not the leading element
    title = _norm_text(items[first][2]).lower().rstrip(".:")
    if title in _DROP_DOC_TITLES:
        return None
    if title in _FRONT_TITLES:
        nxt = next((k for k in range(first + 1, len(items)) if items[k][0] == "h"), None)
        if nxt is not None:
            return items[nxt:]
        # no further heading: the heading and its first paragraph (the annotation) go
        rest = items[first + 1:]
        return rest[1:] if rest and rest[0][0] == "p" else rest
    return items


def _promote_heading_lines(items: list[tuple]) -> list[tuple]:
    """For a document with no heading tags: short lines such as «Глава Первая» become level-1
    headings, and the short title-like lines right after them join the heading text."""
    out: list[tuple] = []
    k = 0
    while k < len(items):
        it = items[k]
        if it[0] == "p" and len(it[1]) <= 60 and _HEADING_LINE_RE.match(it[1]):
            title = it[1]
            k += 1
            for _ in range(2):
                if (k < len(items) and items[k][0] == "p" and len(items[k][1]) <= 60
                        and not items[k][1].rstrip().endswith((".", "!", "?", ":", ";", ","))
                        and not _HEADING_LINE_RE.match(items[k][1])):
                    title += " — " + items[k][1]
                    k += 1
                else:
                    break
            out.append(("h", 1, title))
            continue
        out.append(it)
        k += 1
    return out


def _epub_feed_doc(b: _Builder, items: list[tuple], toc_label: Optional[str]) -> None:
    opened = False        # a chapter was started by this document
    lead_level = 1
    for it in items:
        if it[0] == "h":
            _, level, text = it
            if not opened:
                b.new_chapter(text)
                opened, lead_level = True, level
            elif level == 1 or level < lead_level:
                if b.chapter_has_paragraphs:
                    b.new_chapter(text)
                    lead_level = level
                else:
                    b.extend_chapter_title(text)   # stacked "Part / Chapter" headings
            else:
                b.new_section(text)
        else:
            if not opened:
                opened = True
                if toc_label or not b.has_chapter:
                    b.new_chapter(toc_label)       # untitled document that has its own TOC entry
                # else: a size-split continuation of the previous chapter
            b.add_paragraph(it[1], it[2] if len(it) > 2 else None)


def _epub_load(path: Path, lang: Optional[str]) -> "BookData":
    b = _Builder()
    samples: list[str] = []
    with zipfile.ZipFile(path) as z:
        lookup = _zip_lookup(z)
        opf_full = _opf_path(z)
        opf = _parse_opf(z.read(lookup.get(opf_full.lower(), opf_full)))
        opf_dir = posixpath.dirname(opf_full)
        manifest = opf["manifest"]
        guide_skip = set()
        for typ, href in opf["guide"]:
            if typ in _SKIP_GUIDE_TYPES:
                guide_skip.add(posixpath.normpath(posixpath.join(opf_dir, unquote(href.split("#")[0]))).lower())
        toc = _toc_labels(z, lookup, opf_dir, manifest)
        css_texts = []
        for item in manifest.values():
            if "css" in item["type"]:
                raw_css = _zread(z, lookup, posixpath.normpath(posixpath.join(opf_dir, unquote(item["href"]))))
                if raw_css is not None:
                    css_texts.append(_decode_bytes(raw_css))
        css_marks = _css_class_marks(css_texts)

        for idref, _linear in opf["spine"]:
            item = manifest.get(idref)
            if not item or ("html" not in item["type"] and "xml" not in item["type"]):
                continue
            if "nav" in item["props"].split():
                continue
            full = posixpath.normpath(posixpath.join(opf_dir, unquote(item["href"])))
            stem = posixpath.splitext(posixpath.basename(full))[0]
            if full.lower() in guide_skip or _SKIP_NAME_RE.match(stem) or _SKIP_NAME_RE.match(idref):
                continue
            raw = _zread(z, lookup, full)
            if raw is None:
                continue
            items = _xhtml_blocks(raw, css_marks)
            items = _drop_front_matter(items)
            if items is None:
                continue
            if not any(i[0] == "h" for i in items):
                items = _promote_heading_lines(items)
            if _looks_like_toc(items) or not any(i[0] == "p" for i in items):
                continue
            _epub_feed_doc(b, items, toc.get(full.lower()))
            samples.extend(i[2] if i[0] == "h" else i[1] for i in items[:200])
    meta = opf["meta"]
    book_lang = lang if lang in ("en", "ru") else _text_field(samples, meta.get("language"))
    records, titles = b.finish(book_lang)
    return BookData(records, titles, meta, book_lang)


# -------------------------------------------------------------------- FB2 ----

def _fb2_root(path: Path):
    if path.name.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".fb2")]
            if not names:
                raise ValueError(f"{path}: no .fb2 inside the archive")
            raw = z.read(names[0])
    else:
        raw = path.read_bytes()
    try:
        return ET.fromstring(raw)
    except ET.ParseError:
        # wrong/unknown declared encoding or stray bytes: decode ourselves and retry as UTF-8
        text = re.sub(r"^\s*<\?xml[^>]*\?>", "", _decode_bytes(raw))
        return ET.fromstring(text.encode("utf-8"))


def _fb2_text(el) -> str:
    parts: list[str] = []

    def rec(e, is_root: bool) -> None:
        tag = _local(e.tag)
        if tag == "image":
            pass
        elif tag == "a" and (e.get("type") == "note" or _MARKER_RE.match(_norm_text("".join(e.itertext())))):
            pass  # footnote marker
        else:
            if e.text:
                parts.append(e.text)
            for c in e:
                rec(c, False)
        if not is_root and e.tail:
            parts.append(e.tail)

    rec(el, True)
    return _norm_text("".join(parts))


def _fb2_tokens(el) -> list:
    toks: list = []

    def rec(e, is_root: bool) -> None:
        tag = _local(e.tag)
        mark = _TAG_MARKS.get(tag) if tag in ("emphasis", "strong", "sup", "sub") else None
        if tag == "emphasis":
            mark = "i"
        elif tag == "strong":
            mark = "b"
        if tag == "image":
            pass
        elif tag == "a" and (e.get("type") == "note" or _MARKER_RE.match(_norm_text("".join(e.itertext())))):
            pass  # footnote marker
        else:
            if mark:
                toks.append(("o", mark))
            if e.text:
                toks.append(e.text)
            for c in e:
                rec(c, False)
            if mark:
                toks.append(("c", mark))
        if not is_root and e.tail:
            toks.append(e.tail)

    rec(el, True)
    return toks


def _fb2_rich(el) -> tuple[str, Optional[str]]:
    """(plain text, display markup or None) of a <p>/<v>/<subtitle> element."""
    return _rich_finish(_fb2_tokens(el))


def _fb2_title(sec) -> Optional[str]:
    for c in sec:
        if _local(c.tag) == "title":
            ps = [_fb2_text(p) for p in c if _local(p.tag) == "p"] or [_fb2_text(c)]
            return " ".join(p for p in ps if p) or None
    return None


def _fb2_meta(root) -> dict:
    meta = {"title": None, "author": None, "language": None}
    ti = next((e for e in root.iter() if _local(e.tag) == "title-info"), None)
    if ti is None:
        return meta
    authors = []
    for c in ti:
        t = _local(c.tag)
        if t == "book-title":
            meta["title"] = _norm_text("".join(c.itertext())) or None
        elif t == "lang":
            meta["language"] = _norm_lang("".join(c.itertext()))
        elif t == "author":
            bits = {_local(x.tag): _norm_text("".join(x.itertext())) for x in c}
            name = " ".join(bits[k] for k in ("first-name", "middle-name", "last-name") if bits.get(k))
            authors.append(name or bits.get("nickname", ""))
    meta["author"] = ", ".join(a for a in dict.fromkeys(authors) if a) or None
    return meta


_FB2_DIRECT = {"p", "poem", "cite", "epigraph"}


def _fb2_section(b: _Builder, sec, in_chapter: bool, samples: list[str]) -> None:
    kids = list(sec)
    subs = [c for c in kids if _local(c.tag) == "section"]
    has_direct = any(_local(c.tag) in _FB2_DIRECT for c in kids)
    title = _fb2_title(sec)
    if not in_chapter and subs and not has_direct:
        b.new_part(title)                      # container section (part): its children are chapters
        for s in subs:
            _fb2_section(b, s, False, samples)
        b.end_part()
        return
    if in_chapter:
        b.new_section(title)
    else:
        b.new_chapter(title)

    def para(text: str, fmt: Optional[str] = None) -> None:
        if text and _HAS_ALNUM_RE.search(text) and not _PAGE_NUM_RE.match(text):
            b.add_paragraph(text, fmt)
            samples.append(text)

    def block(e) -> None:
        t = _local(e.tag)
        if t == "p":
            para(*_fb2_rich(e))
        elif t == "poem":
            for st in e.iter():
                if _local(st.tag) == "stanza":
                    toks: list = []
                    for v in st:
                        if _local(v.tag) == "v":
                            if toks:
                                toks.append(("br", None))
                            toks.extend(_fb2_tokens(v))
                    para(*_rich_finish(toks))
        elif t in ("cite", "epigraph"):
            for c in e:
                if _local(c.tag) in ("p", "poem", "cite"):
                    block(c)

    for c in kids:
        t = _local(c.tag)
        if t in _FB2_DIRECT:
            block(c)
        elif t == "subtitle":
            txt = _fb2_text(c)
            if txt:
                b.new_section(txt)
        elif t == "section":
            _fb2_section(b, c, True, samples)


def _fb2_load(path: Path, lang: Optional[str]) -> "BookData":
    root = _fb2_root(path)
    meta = _fb2_meta(root)
    b = _Builder()
    samples: list[str] = []
    for body in (e for e in root if _local(e.tag) == "body"):
        kind = ((body.get("name") or "") + " " + (body.get("type") or "")).lower()
        if "notes" in kind or "comments" in kind or "footnote" in kind:
            continue
        secs = [c for c in body if _local(c.tag) == "section"]
        if not secs:  # body with bare paragraphs
            b.new_chapter(None)
            for p in body:
                if _local(p.tag) == "p":
                    t, f = _fb2_rich(p)
                    if t:
                        b.add_paragraph(t, f)
                        samples.append(t)
        for s in secs:
            _fb2_section(b, s, False, samples)
    book_lang = lang if lang in ("en", "ru") else _text_field(samples, meta.get("language"))
    records, titles = b.finish(book_lang)
    return BookData(records, titles, meta, book_lang)


# --------------------------------------------------------------- dispatcher ----

@dataclass
class BookData:
    records: list
    titles: Optional[dict] = None     # same layout as data/titles/<book_id>_titles.json
    meta: dict = field(default_factory=dict)   # {title, author, language} as declared by the file
    lang: Optional[str] = None        # field the text lives in: "ru" | "en"; None = bilingual en+ru


def _kind(path: Path) -> str:
    n = path.name.lower()
    if n.endswith(".fb2.zip") or n.endswith(".fb2"):
        return "fb2"
    ext = path.suffix.lower()
    if ext in (".epub", ".json", ".md"):
        return ext[1:]
    raise ValueError(f"unsupported book format: {path.name} (supported: {', '.join(BOOK_EXTENSIONS)})")


def load_book(path: str | Path, lang: Optional[str] = None) -> BookData:
    """Parse any supported source into BookData (records + built-in titles + meta). `lang` forces the
    text field ("ru"/"en") for EPUB/FB2; by default it is detected from the text."""
    path = Path(path)
    kind = _kind(path)
    if kind == "epub":
        return _epub_load(path, lang)
    if kind == "fb2":
        return _fb2_load(path, lang)
    recs = list(parse_json(path) if kind == "json" else parse_md(path))
    return BookData(recs, None, {}, None)


def parse_epub(path: str | Path, lang: Optional[str] = None) -> Iterator[dict]:
    yield from _epub_load(Path(path), lang).records


def parse_fb2(path: str | Path, lang: Optional[str] = None) -> Iterator[dict]:
    yield from _fb2_load(Path(path), lang).records


def parse_book(path: str | Path) -> Iterator[dict]:
    """Yield records from .json / .md (bilingual corpus), .epub, .fb2 or .fb2.zip."""
    yield from load_book(path).records


def read_book_meta(path: str | Path) -> dict:
    """{title, author, language} from the OPF (EPUB) or FB2 description; Nones for json/md or when
    missing. Does not parse the book text."""
    path = Path(path)
    kind = _kind(path)
    if kind == "epub":
        with zipfile.ZipFile(path) as z:
            lookup = _zip_lookup(z)
            p = _opf_path(z)
            return _parse_opf(z.read(lookup.get(p.lower(), p)))["meta"]
    if kind == "fb2":
        return _fb2_meta(_fb2_root(path))
    return {"title": None, "author": None, "language": None}
