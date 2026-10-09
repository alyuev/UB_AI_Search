import os

import pytest

from app import config
from app import history as hist


@pytest.fixture(autouse=True)
def fresh_history():
    hist.clear_unpinned()
    yield
    hist.clear_unpinned()


def test_add_list_get_roundtrip():
    i = hist.add("urantia", "search", "Божество", {"mode": "hybrid", "k": 10}, {"results": [{"id": "0:1.2"}]},
                 "1 результат", 0.5)
    lst = hist.list_entries()
    assert lst["total"] == 1 and lst["items"][0]["id"] == i
    row = lst["items"][0]
    assert row["query"] == "Божество" and row["params"] == {"mode": "hybrid", "k": 10}
    assert row["summary"] == "1 результат" and row["pinned"] is False
    assert row.get("result") is None  # list rows are light: no snapshot
    full = hist.get(i)
    assert full["result"] == {"results": [{"id": "0:1.2"}]} and full["duration_s"] == 0.5
    assert hist.get(99999) is None


def test_rejects_unknown_mode():
    with pytest.raises(ValueError):
        hist.add("b", "weird", "q", {}, {}, "s")


def test_filters_by_mode_text_book_and_cyrillic_case():
    hist.add("a", "search", "Божество и Бог", {}, {}, "s")
    hist.add("a", "rag", "кто такие агондонтеры?", {}, {}, "s")
    hist.add("b", "translation", "Paper 0 Section 1", {}, {}, "s")
    assert hist.list_entries(mode="rag")["total"] == 1
    assert hist.list_entries(book="b")["items"][0]["mode"] == "translation"
    # case-insensitive for Cyrillic too (plain SQLite LIKE is not)
    assert hist.list_entries(q="БОЖЕСТВО")["total"] == 1
    assert hist.list_entries(q="paper 0")["total"] == 1
    assert hist.list_entries(q="нет такого")["total"] == 0


def test_order_newest_first_and_paging():
    ids = [hist.add("a", "search", f"q{n}", {}, {}, "s") for n in range(5)]
    items = hist.list_entries(limit=2)["items"]
    assert [x["id"] for x in items] == [ids[4], ids[3]]
    assert [x["id"] for x in hist.list_entries(limit=2, offset=2)["items"]] == [ids[2], ids[1]]


def test_pin_toggle_delete_and_clear_keeps_pinned():
    a = hist.add("a", "search", "keep", {}, {}, "s")
    b = hist.add("a", "search", "drop", {}, {}, "s")
    assert hist.set_pinned(a) is True and hist.get(a)["pinned"] is True
    assert hist.set_pinned(a) is False
    assert hist.set_pinned(a, True) is True
    assert hist.set_pinned(12345) is None
    assert hist.clear_unpinned() == 1
    assert hist.get(b) is None and hist.get(a) is not None
    assert hist.delete(a) is True and hist.delete(a) is False


def test_cap_never_removes_pinned(monkeypatch):
    monkeypatch.setattr(hist, "MAX_ENTRIES", 5)
    pinned = hist.add("a", "search", "important", {}, {}, "s")
    hist.set_pinned(pinned, True)
    for n in range(9):
        hist.add("a", "search", f"q{n}", {}, {}, "s")
    lst = hist.list_entries(limit=100)
    assert lst["total"] == 5
    assert pinned in [x["id"] for x in lst["items"]]
    queries = [x["query"] for x in lst["items"]]
    assert "q8" in queries and "q0" not in queries  # newest survive, oldest unpinned dropped


def test_history_db_lives_in_data_dir_not_in_book_db():
    hist.add("a", "search", "x", {}, {}, "s")
    assert os.path.isfile(os.path.join(str(config.DATA_DIR), "history.db"))
