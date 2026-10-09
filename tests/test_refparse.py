
from app.llm.refparse import parse_reference


def test_colon_range():
    r = parse_reference("147:2.4-11")
    assert r.as_tuple() == (147, 2, 4, 11)


def test_colon_single_para():
    r = parse_reference("147:2.4")
    assert r.as_tuple() == (147, 2, 4, 4)


def test_colon_section_only():
    r = parse_reference("147:2")
    assert r.as_tuple() == (147, 2, None, None)


def test_paper_section_english():
    r = parse_reference("Paper 147 Section 2")
    assert r.as_tuple() == (147, 2, None, None)


def test_paper_section_russian():
    r = parse_reference("Документ 147 раздел 2")
    assert r.as_tuple() == (147, 2, None, None)


def test_paper_only_english():
    r = parse_reference("Paper 147")
    assert r.as_tuple() == (147, None, None, None)


def test_paper_only_russian_ves():
    r = parse_reference("весь Paper 147")
    assert r.as_tuple() == (147, None, None, None)


def test_document_only_russian():
    r = parse_reference("Документ 147")
    assert r.as_tuple() == (147, None, None, None)


def test_no_match():
    assert parse_reference("привет мир") is None
    assert parse_reference("") is None
    assert parse_reference(None) is None


def test_case_insensitive():
    r = parse_reference("paper 5 section 1")
    assert r.as_tuple() == (5, 1, None, None)


def test_parse_all_references_multiple():
    from app.llm.refparse import parse_all_references
    refs = parse_all_references("см. 147:2 и ещё 5:1.3-4")
    tuples = [r.as_tuple() for r in refs]
    assert (147, 2, None, None) in tuples
    assert (5, 1, 3, 4) in tuples
