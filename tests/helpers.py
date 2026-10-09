"""Test helper: build a tiny book.db using the REAL app.core.db schema (TEXT ids,
book_id/seq columns), so server/translation tests exercise the real core contract
instead of a schema mismatch."""
from __future__ import annotations

from app.core.db import open_db

_DATA = [
    # id, paper, section, paragraph, en, ru
    ("147.2.1", 147, 2, 1, "In the beginning was the Word.", "В начале было Слово."),
    ("147.2.2", 147, 2, 2, "Jesus spoke of the Father's love.", "Иисус говорил о любви Отца."),
    ("147.2.3", 147, 2, 3, "The disciples listened intently.", "Ученики слушали внимательно."),
    ("147.2.4", 147, 2, 4, "It was evening in Jerusalem.", "Был вечер в Иерусалиме."),
    ("1.1.1", 1, 1, 1, "God is spirit.", "Бог есть дух."),
]


def create_fake_book_db(path: str, book_id: str = "urantia", book_name: str = "Fake Book"):
    conn = open_db(path, readonly=False)
    conn.execute(
        "INSERT OR REPLACE INTO books (book_id, book_name, language_source, language_translation) "
        "VALUES (?, ?, 'en', 'ru')",
        (book_id, book_name),
    )
    for seq, (id_, paper, section, paragraph, en, ru) in enumerate(_DATA):
        conn.execute(
            "INSERT OR REPLACE INTO paragraphs "
            "(id, book_id, seq, paper, section, paragraph, en, ru) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (id_, book_id, seq, paper, section, paragraph, en, ru),
        )
        conn.execute(
            "INSERT INTO paragraphs_fts (id, en, ru, ru_lemma) VALUES (?, ?, ?, ?)",
            (id_, en, ru, ru.lower()),
        )
    conn.commit()
    conn.close()
