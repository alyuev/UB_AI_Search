"""Fake app.core.indexer for tests."""
from __future__ import annotations

import threading


def build_index(json_path, db_path, model_name, scheme="D", book_id=None, book_name=None,
                 progress=None, cancel: threading.Event = None):
    from . import db as fake_db
    stages = ["parse", "embed_en", "embed_ru", "write_db", "fts"]
    total = 5
    for i, stage in enumerate(stages, start=1):
        if cancel is not None and cancel.is_set():
            if progress:
                progress(stage, i, total, None, "cancelled")
            return {"cancelled": True}
        if progress:
            progress(stage, i, total, None, f"stage {stage}")
    fake_db.create_fake_db(db_path, book_id or "urantia", book_name or "Fake Book")
    return {"cancelled": False, "paragraphs": 5}
