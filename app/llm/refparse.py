"""Parse bilingual RU/EN references like "Paper 147 Section 2", "Документ 147 раздел 2",
"147:2", "147:2.4-11", "147:2.4", "весь Paper 147", "Paper 147" into (paper, section, para_from, para_to).

Returns a tuple (paper: int, section: int|None, para_from: int|None, para_to: int|None)
or None if nothing recognizable was found.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Reference:
    paper: int
    section: Optional[int] = None
    para_from: Optional[int] = None
    para_to: Optional[int] = None

    def as_tuple(self):
        return (self.paper, self.section, self.para_from, self.para_to)


# word forms that mean "paper/document"
_PAPER_WORDS = r"(?:paper|документ|док-?т|раздел(?:а)?|part)"
_PAPER_ONLY_WORDS = r"(?:paper|документ|док)"
_SECTION_WORDS = r"(?:section|раздел|секция|разд\.?)"
_PARA_WORDS = r"(?:paragraph|абзац|параграф|para\.?)"


def parse_reference(text: str) -> Optional[Reference]:
    """Best-effort parse of a single reference from free text.

    Supported shapes (case-insensitive, RU/EN):
      - "147:2.4-11"  -> paper=147 section=2 para_from=4 para_to=11
      - "147:2.4"     -> paper=147 section=2 para_from=4 para_to=4
      - "147:2"       -> paper=147 section=2
      - "Paper 147 Section 2" / "Документ 147 раздел 2"
      - "весь Paper 147" / "Paper 147" -> paper=147 only
    """
    if not text:
        return None
    s = text.strip()

    # 1) compact colon notation: 147:2.4-11 / 147:2.4 / 147:2 / 147
    m = re.search(
        r"\b(?P<paper>\d{1,4})\s*:\s*(?P<section>\d{1,4})"
        r"(?:\.(?P<pfrom>\d{1,4})(?:\s*-\s*(?P<pto>\d{1,4}))?)?\b",
        s,
    )
    if m:
        paper = int(m.group("paper"))
        section = int(m.group("section"))
        pfrom = m.group("pfrom")
        pto = m.group("pto")
        if pfrom is not None:
            para_from = int(pfrom)
            para_to = int(pto) if pto is not None else para_from
            return Reference(paper, section, para_from, para_to)
        return Reference(paper, section)

    # 2) word forms: Paper 147 Section 2[.4[-11]]
    m = re.search(
        rf"{_PAPER_ONLY_WORDS}\.?\s*№?\s*(?P<paper>\d{{1,4}})"
        rf"(?:\s+{_SECTION_WORDS}\.?\s*№?\s*(?P<section>\d{{1,4}}))?"
        rf"(?:\s*[.,]\s*(?:{_PARA_WORDS}\.?\s*)?(?P<pfrom>\d{{1,4}})"
        rf"(?:\s*-\s*(?P<pto>\d{{1,4}}))?)?",
        s,
        re.IGNORECASE,
    )
    if m and m.group("paper"):
        paper = int(m.group("paper"))
        section = int(m.group("section")) if m.group("section") else None
        pfrom = m.group("pfrom")
        pto = m.group("pto")
        if pfrom is not None:
            para_from = int(pfrom)
            para_to = int(pto) if pto is not None else para_from
            return Reference(paper, section, para_from, para_to)
        return Reference(paper, section)

    # 3) "раздел 2" without an explicit paper number nearby -> not enough info
    return None


def parse_all_references(text: str) -> list[Reference]:
    """Find all references mentioned in a longer piece of text (best effort, non-overlapping)."""
    if not text:
        return []
    results: list[Reference] = []
    # try colon notation occurrences
    for m in re.finditer(
        r"\b(\d{1,4})\s*:\s*(\d{1,4})(?:\.(\d{1,4})(?:\s*-\s*(\d{1,4}))?)?\b", text
    ):
        paper = int(m.group(1))
        section = int(m.group(2))
        pfrom = m.group(3)
        pto = m.group(4)
        if pfrom is not None:
            para_from = int(pfrom)
            para_to = int(pto) if pto is not None else para_from
            results.append(Reference(paper, section, para_from, para_to))
        else:
            results.append(Reference(paper, section))
    if results:
        return results
    # fallback: try word notation, one shot
    r = parse_reference(text)
    return [r] if r else []
