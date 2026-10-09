#!/usr/bin/env python3
"""Extract structural titles (parts / papers / sections) from a pair of
bilingual EPUB books that follow the Urantia Foundation EPUB layout:

    EPUB/docs/01-Parts.xhtml        -- part titles + which papers belong to them
    EPUB/docs/04-Paper000.xhtml     -- paper 0 (Foreword) + its h2 sections
    EPUB/docs/06-Paper001.xhtml ..  -- paper N: <h1> title, <h2> section titles

The layout is generic enough to work for any EPUB built the same way (single
<h1> = paper title, <h2> = section titles, a "Parts" page listing part number,
part title and the papers under each part via anchors "PaperNNN").

Usage:
    python extract_epub_titles.py <en_epub> <ru_epub> <out_json>

Only Python stdlib is used (zipfile + re / html.parser).
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
import zipfile
from html.parser import HTMLParser
from pathlib import Path

PAPER_FILE_RE = re.compile(r"Paper(\d{3})\.xhtml$")
PART_FILE_RE = re.compile(r"Part(\d+)\.xhtml$")
# Capture an optional id="U<paper>_<section>_0" alongside the heading text so
# section numbering can be taken from the id rather than positional order
# (a paper can have a combined heading like "9. and 10. ..." that covers two
# section numbers but is only one <h2> element).
H_TAG_RE = re.compile(
    r'<h([1-4])\b(?:[^>]*\bid="U\d+_(\d+)_0")?[^>]*>(.*?)</h\1>',
    re.IGNORECASE | re.DOTALL,
)
# Matches leading section number(s) in a heading, e.g. "1. Title",
# "9 and 10. Title", "9 и 10. Title" (Russian "and"). We don't match the
# connector word itself (locale-dependent) -- just pull out up to two leading
# integers separated by non-digit filler, then require a period/space before
# the real title text.
LEADING_SECTION_NUMS_RE = re.compile(
    r"^\s*(\d+)\s*[.\)]?\s*(?:[^\d.]{0,6}(\d+)\s*[.\)]?)?\s+",
)
TAG_RE = re.compile(r"<[^>]+>")
FOOTNOTE_RE = re.compile(r"\[\s*\d+\s*\]|\(\s*\d+\s*\)$")
LEADING_NUM_RE = re.compile(
    r"^\s*(?:\d+\.|\d+\)|[IVXLCDM]+\.)\s+", re.IGNORECASE
)


class _TextExtractor(HTMLParser):
    """Collects visible text from a small HTML fragment (a heading)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data):
        self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def strip_tags(fragment: str) -> str:
    parser = _TextExtractor()
    parser.feed(fragment)
    parser.close()
    return parser.text()


def clean_text(t: str) -> str:
    """Strip numbering ("1. ", "I. "), footnote markers, collapse whitespace.
    Expects already-detagged text (see strip_tags)."""
    t = unicodedata.normalize("NFC", t)
    t = t.replace(" ", " ")
    t = FOOTNOTE_RE.sub("", t)
    # Strip leading numbering: try the digit-combo form first ("9. and 10. ",
    # "1. ") since it correctly consumes combined section numbers; fall back
    # to the roman-numeral form ("I. ") used by Paper 0's sections.
    m = LEADING_SECTION_NUMS_RE.match(t)
    if m:
        t = t[m.end():]
    else:
        t = LEADING_NUM_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def clean_title(raw: str) -> str:
    return clean_text(strip_tags(raw))


def title_case_if_reliable(title: str) -> str | None:
    """Return a title-cased version only when the source looks ALL CAPS
    (i.e. the EPUB stores the heading in uppercase and we can reliably
    reconstruct a natural title case). Returns None otherwise."""
    letters = [c for c in title if c.isalpha()]
    if not letters:
        return None
    if not all(c.isupper() for c in letters):
        return None
    # crude but reliable-enough title case: capitalize each word, keep
    # short connector words lower-case unless they are the first word.
    small_words = {
        "a", "an", "and", "as", "at", "but", "by", "for", "in", "of",
        "on", "or", "the", "to", "with",
    }
    words = title.split(" ")
    out = []
    for i, w in enumerate(words):
        core = w.lower()
        if i > 0 and core in small_words:
            out.append(core)
        else:
            out.append(core[:1].upper() + core[1:] if core else core)
    return " ".join(out)


def read_zip_text(zf: zipfile.ZipFile, name: str) -> str:
    return zf.read(name).decode("utf-8")


def extract_headings(html: str) -> list[tuple[int, int | None, str, list[int]]]:
    """Return [(level, id_section_num_or_None, cleaned_text, leading_nums), ...]
    for all h1-h4 tags in document order. leading_nums are the section
    number(s) found at the start of the raw (pre-clean) heading text, e.g.
    [9, 10] for a combined "9. and 10. ..." / "9 и 10. ..." heading."""
    out = []
    for m in H_TAG_RE.finditer(html):
        level = int(m.group(1))
        id_sec = int(m.group(2)) if m.group(2) is not None else None
        raw_text = strip_tags(m.group(3))
        raw_text_n = unicodedata.normalize("NFC", raw_text).replace(" ", " ").strip()
        num_match = LEADING_SECTION_NUMS_RE.match(raw_text_n)
        nums = [int(g) for g in num_match.groups() if g] if num_match else []
        text = clean_text(raw_text)
        if text:
            out.append((level, id_sec, text, nums))
    return out


def find_paper_files(names: list[str]) -> dict[int, str]:
    result = {}
    for n in names:
        m = PAPER_FILE_RE.search(n)
        if m:
            result[int(m.group(1))] = n
    return result


def find_part_files(names: list[str]) -> dict[int, str]:
    result = {}
    for n in names:
        m = PART_FILE_RE.search(n)
        if m:
            result[int(m.group(1))] = n
    return result


def extract_paper_titles_and_sections(
    zf: zipfile.ZipFile, paper_files: dict[int, str]
) -> dict[int, dict]:
    """Returns {paper_num: {"title": str, "sections": {sec_num: str}}}"""
    out = {}
    for pnum, fname in paper_files.items():
        html = read_zip_text(zf, fname)
        headings = extract_headings(html)
        title = None
        sections: dict[int, str] = {}
        sec_counter = 0
        for level, id_sec, text, nums in headings:
            if level == 1 and title is None:
                title = text
                continue
            if level != 2:
                continue
            sec_counter += 1
            # A heading can cover more than one section number (e.g.
            # "9. and 10. James and Judas Alpheus" / "9 и 10. ...").
            # id_sec (from id="U<paper>_<section>_0") gives the *last*
            # section number the heading covers; nums (parsed from the raw,
            # pre-clean heading text) recovers any earlier numbers too.
            if id_sec is not None:
                target = id_sec
            elif nums:
                target = max(nums)
            else:
                target = sec_counter
            covered = sorted(set(nums)) if len(nums) > 1 else []
            if covered and covered[-1] == target:
                for n in covered:
                    sections[n] = text
            else:
                sections[target] = text
            sec_counter = target
        out[pnum] = {"title": title or "", "sections": sections}
    return out


def extract_parts(
    zf: zipfile.ZipFile, names: list[str]
) -> list[dict]:
    """Get part number, title and the paper-number range under each part.

    Each part has its own page (e.g. EPUB/docs/05-Part1.xhtml) that carries
    the part's <h2> title and links to every paper it contains, with
    zero-padded hrefs like "06-Paper001.xhtml" -- unlike the combined
    01-Parts.xhtml overview page, which only lists part-level blurbs.
    """
    part_files = find_part_files(names)
    parts = []
    for part_num, fname in sorted(part_files.items()):
        html = read_zip_text(zf, fname)
        h2m = re.search(r"<h2[^>]*>(.*?)</h2>", html, re.IGNORECASE | re.DOTALL)
        part_title = clean_title(h2m.group(1)) if h2m else ""
        # Only count papers listed in the part's own table of contents
        # (class="btn pti" rows), not nav-bar "prev/next" links at the top
        # or bottom of the page, which can reference the adjacent part's
        # boundary paper.
        pti_rows = re.findall(
            r'class="btn pti"[^>]*>\s*<a[^>]*href="[^"]*Paper(\d{3})\.xhtml"',
            html,
        )
        paper_nums = sorted({int(x) for x in pti_rows})
        low, high = (paper_nums[0], paper_nums[-1]) if paper_nums else (None, None)
        parts.append(
            {"part": part_num, "title": part_title, "papers": [low, high]}
        )
    return parts


def build_language_data(epub_path: str):
    zf = zipfile.ZipFile(epub_path)
    names = zf.namelist()
    paper_files = find_paper_files(names)
    papers = extract_paper_titles_and_sections(zf, paper_files)
    parts = extract_parts(zf, names)
    return parts, papers


def merge(en_parts, en_papers, ru_parts, ru_papers):
    # --- parts ---
    ru_parts_by_num = {p["part"]: p for p in ru_parts}
    parts_out = []
    for p in sorted(en_parts, key=lambda x: x["part"]):
        ru_p = ru_parts_by_num.get(p["part"], {})
        parts_out.append(
            {
                "part": p["part"],
                "en": p["title"],
                "ru": ru_p.get("title", ""),
                "papers": p["papers"],
            }
        )

    # --- papers + sections ---
    all_paper_nums = sorted(set(en_papers) | set(ru_papers))
    papers_out = {}
    for pnum in all_paper_nums:
        en_p = en_papers.get(pnum, {"title": "", "sections": {}})
        ru_p = ru_papers.get(pnum, {"title": "", "sections": {}})

        # figure out which part this paper belongs to
        part_num = None
        for p in parts_out:
            lo, hi = p["papers"]
            if lo is not None and lo <= pnum <= hi:
                part_num = p["part"]
                break

        en_title = en_p["title"]
        entry = {
            "en": en_title,
            "ru": ru_p["title"],
            "part": part_num,
            "sections": {},
        }
        tc = title_case_if_reliable(en_title)
        if tc:
            entry["en_titlecase"] = tc

        sec_nums = sorted(set(en_p["sections"]) | set(ru_p["sections"]))
        for snum in sec_nums:
            entry["sections"][str(snum)] = {
                "en": en_p["sections"].get(snum, ""),
                "ru": ru_p["sections"].get(snum, ""),
            }
        # section 0 = intro (untitled), included for completeness
        entry["sections"]["0"] = {"en": "", "ru": ""}
        papers_out[str(pnum)] = entry

    return {"parts": parts_out, "papers": papers_out}


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: extract_epub_titles.py <en_epub> <ru_epub> <out_json>",
            file=sys.stderr,
        )
        sys.exit(1)
    en_epub, ru_epub, out_path = sys.argv[1:4]

    en_parts, en_papers = build_language_data(en_epub)
    ru_parts, ru_papers = build_language_data(ru_epub)

    result = merge(en_parts, en_papers, ru_parts, ru_papers)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_file} with {len(result['papers'])} papers, "
          f"{len(result['parts'])} parts.")


if __name__ == "__main__":
    main()
