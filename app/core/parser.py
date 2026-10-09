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
