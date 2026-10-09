
import json

from app.llm.translation import (
    parse_llm_json, make_batches, export_markdown, TranslationAnalyzer,
)


def test_parse_llm_json_clean_array():
    text = json.dumps([{"id": 1, "severity": "minor", "category": "style",
                         "problem": "p", "why_significant": "w", "suggestion": "s"}])
    parsed = parse_llm_json(text)
    assert parsed[0]["id"] == 1


def test_parse_llm_json_with_code_fence():
    text = "```json\n" + json.dumps([{"id": 2, "severity": "major", "category": "grammar",
                                       "problem": "x"}]) + "\n```"
    parsed = parse_llm_json(text)
    assert parsed[0]["id"] == 2


def test_parse_llm_json_wrapped_in_prose():
    text = "Here is the result:\n[{\"id\": 3, \"severity\": \"critical\", \"category\": \"omissions\", \"problem\": \"x\"}]\nDone."
    parsed = parse_llm_json(text)
    assert parsed[0]["id"] == 3


def test_parse_llm_json_garbage_returns_empty():
    assert parse_llm_json("not json at all") == []
    assert parse_llm_json("") == []


def test_make_batches_respects_budget():
    paras = [{"id": i, "en": "word " * 50, "ru": "слово " * 50} for i in range(20)]
    batches = make_batches(paras, max_context_tokens=2000, reserved_tokens=500)
    assert sum(len(b) for b in batches) == 20
    assert len(batches) > 1


def test_export_markdown_no_issues():
    md = export_markdown({"summary": "все ок", "issues": []})
    assert "все ок" in md
    assert "не обнаружено" in md.lower() or "Проблем" in md


def test_export_markdown_with_issue():
    report = {"summary": "1 проблема", "issues": [
        {"id": 5, "severity": "critical", "category": "distortions", "problem": "искажение",
         "why_significant": "меняет смысл", "suggestion": "исправить", "en": "hello", "ru": "прощай"}
    ]}
    md = export_markdown(report)
    assert "[5]" in md
    assert "hello" in md and "прощай" in md


class FakeStats:
    stop_reason = "stop"
    completion_tokens = 5
    prompt_tokens = 10
    reasoning_tokens = 0
    elapsed_s = 0.1
    ttft_s = 0.05
    tokens_per_sec = 50.0

    def to_dict(self):
        return {"model": "test-model", "elapsed_s": 0.1, "ttft_s": 0.05,
                "prompt_tokens": 10, "completion_tokens": 5, "reasoning_tokens": 0,
                "total_tokens": 15, "tokens_per_sec": 50.0, "stop_reason": "stop"}

    def summary_line(self):
        return "test-model · 50.0 ток/с · первый токен 0.05 с · 10 → 5 ток. · 0.1 с"


class FakeLLM:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def chat(self, messages, model, temperature, max_tokens, timeout, stream=False, reasoning_effort=None):
        resp = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        return resp

    def chat_with_stats(self, messages, model, temperature, max_tokens, timeout, reasoning_effort=None):
        resp = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        return resp, FakeStats()


def test_analyzer_end_to_end(tmp_path):
    db_path = str(tmp_path / "book.db")
    from app.core.db import open_db
    from tests.helpers import create_fake_book_db
    create_fake_book_db(db_path)
    conn = open_db(db_path, readonly=True)

    fake_response = json.dumps([
        {"id": "147.2.2", "severity": "minor", "category": "style", "problem": "тон изменён",
         "en_quote": "Jesus spoke", "ru_quote": "Иисус говорил",
         "why_significant": "незначительно", "suggestion": "оставить как есть"}
    ])
    llm = FakeLLM([fake_response])
    analyzer = TranslationAnalyzer(conn, llm, model="test-model", max_context_length=8000)

    progress_calls = []

    def progress(stage, current, total, current_id, message):
        progress_calls.append((stage, current, total))

    result = analyzer.analyze(147, 2, 1, 4, progress=progress)
    assert result["paragraphs_analyzed"] == 4
    assert len(result["issues"]) == 1
    assert result["issues"][0]["id"] == "147.2.2"
    assert result["issues"][0]["en"] == "Jesus spoke of the Father's love."
    stage_names = {c[0] for c in progress_calls}
    assert "Сравнение EN/RU" in stage_names
    assert "Анализ смысловых расхождений" in stage_names
    assert "Формирование отчёта" in stage_names


class FailingLLM:
    """LLM double whose every call raises the given LMStudioError."""

    def __init__(self, exc):
        self.exc = exc

    def chat_with_stats(self, *a, **kw):
        raise self.exc


def test_analyzer_surfaces_no_model_loaded_as_error(tmp_path):
    """Regression: a missing/unreachable model must NEVER be reported as
    'no significant issues found' -- it must come back as an explicit error."""
    from app.core.db import open_db
    from app.llm.lmstudio import NoModelLoaded
    from tests.helpers import create_fake_book_db
    db_path = str(tmp_path / "book.db")
    create_fake_book_db(db_path)
    conn = open_db(db_path, readonly=True)
    analyzer = TranslationAnalyzer(conn, FailingLLM(NoModelLoaded()), model="")
    result = analyzer.analyze(147, 2, 1, 4)
    assert result["error"]
    assert "не загружена" in result["error"]
    assert result["summary"] is None  # no reassuring "nothing found" summary
    assert result["issues"] == []


def test_analyzer_surfaces_connection_error(tmp_path):
    from app.core.db import open_db
    from app.llm.lmstudio import ConnectionRefused
    from tests.helpers import create_fake_book_db
    db_path = str(tmp_path / "book.db")
    create_fake_book_db(db_path)
    conn = open_db(db_path, readonly=True)
    analyzer = TranslationAnalyzer(conn, FailingLLM(ConnectionRefused("http://localhost:1234")), model="m")
    result = analyzer.analyze(147, 2)
    assert result["error"] and "1234" in result["error"]


class ScriptedLLM:
    """Returns scripted answers in order: first the review, then verification verdicts."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def chat_with_stats(self, messages, model, temperature, max_tokens, timeout, reasoning_effort=None):
        self.calls += 1
        return self.answers.pop(0), FakeStats()


_BATCH = [{"id": "p1", "paper": 0, "section": 1, "paragraph": 1,
           "en": "She did not go to the market.", "ru": "Она пошла на рынок."}]


def test_quote_supported():
    from app.llm.translation import quote_supported
    assert quote_supported("did not go", "She did not go to the market.")
    assert not quote_supported("fabricated sentence about elephants", "She did not go to the market.")
    assert not quote_supported("", "anything")
    assert quote_supported("пошла на рынок", "Она пошла на рынок.")


def test_review_batch_drops_fabricated_evidence():
    from app.llm.translation import review_batch
    review = json.dumps([{"id": "p1", "severity": "major", "category": "omissions",
                          "en_quote": "elephants dance wildly", "ru_quote": "", "problem": "x"}])
    llm = ScriptedLLM([review])
    res = review_batch(llm, "m", _BATCH, verify=True)
    assert res["issues"] == []
    assert res["dropped"][0]["reason"] == "en_quote_not_in_source"
    assert llm.calls == 1  # no verification call wasted on a fabricated claim


def test_review_batch_verification_confirms_and_rejects():
    from app.llm.translation import review_batch
    review = json.dumps([{"id": "p1", "severity": "critical", "category": "distortions",
                          "en_quote": "did not go", "ru_quote": "пошла", "problem": "negation lost"}])
    confirmed = review_batch(ScriptedLLM([review, '{"verdict":"confirmed","reason":"negation missing"}']),
                             "m", _BATCH, verify=True)
    assert len(confirmed["issues"]) == 1 and confirmed["issues"][0].verified is True
    rejected = review_batch(ScriptedLLM([review, '{"verdict":"rejected","reason":"same meaning"}']),
                            "m", _BATCH, verify=True)
    assert rejected["issues"] == [] and rejected["dropped"][0]["reason"] == "rejected_by_verification"
    # unparseable verification answer => conservative: rejected
    garbage = review_batch(ScriptedLLM([review, "hmm not sure"]), "m", _BATCH, verify=True)
    assert garbage["issues"] == []


def test_review_batch_no_verify_keeps_issue():
    from app.llm.translation import review_batch
    review = json.dumps([{"id": "p1", "severity": "major", "category": "distortions",
                          "en_quote": "did not go", "ru_quote": "пошла", "problem": "negation lost"}])
    llm = ScriptedLLM([review])
    res = review_batch(llm, "m", _BATCH, verify=False)
    assert len(res["issues"]) == 1 and res["issues"][0].verified is None and llm.calls == 1


def test_review_batch_empty_array_means_no_issues():
    from app.llm.translation import review_batch
    res = review_batch(ScriptedLLM(["[]"]), "m", _BATCH)
    assert res["issues"] == [] and res["dropped"] == []


def test_review_batch_addition_with_only_ru_quote_is_kept():
    from app.llm.translation import review_batch
    batch = [{"id": "p1", "paper": 0, "section": 1, "paragraph": 1,
              "en": "The sun rose.", "ru": "Солнце взошло, и все птицы запели."}]
    review = json.dumps([{"id": "p1", "severity": "major", "category": "additions",
                          "en_quote": "", "ru_quote": "и все птицы запели", "problem": "invented"}])
    res = review_batch(ScriptedLLM([review]), "m", batch, verify=False)
    assert len(res["issues"]) == 1 and res["issues"][0].ru_quote == "и все птицы запели"


def test_review_batch_requires_some_evidence():
    from app.llm.translation import review_batch
    review = json.dumps([{"id": "p1", "severity": "major", "category": "omissions",
                          "en_quote": "", "ru_quote": "", "problem": "vague"}])
    res = review_batch(ScriptedLLM([review]), "m", _BATCH, verify=False)
    assert res["issues"] == [] and res["dropped"][0]["reason"] == "no_evidence"


def test_make_batches_caps_items_per_batch():
    paras = [{"id": str(i), "en": "a b c", "ru": "а б в"} for i in range(20)]
    batches = make_batches(paras, max_context_tokens=100000, max_items=8)
    assert [len(b) for b in batches] == [8, 8, 4]


def test_length_hint_flags_unusual_ratio():
    from app.llm.translation import length_hint
    en = "x" * 200
    assert "SHORT" in length_hint(en, "y" * 120)
    assert "LONG" in length_hint(en, "y" * 300)
    assert length_hint(en, "y" * 200) == ""


def test_review_batch_flags_truncated_answer():
    from app.llm.translation import review_batch

    class Cut(ScriptedLLM):
        def chat_with_stats(self, *a, **kw):
            raw, st = super().chat_with_stats(*a, **kw)
            st.stop_reason = "length"
            return raw, st

    class S(FakeStats):
        pass

    res = review_batch(Cut(["[]"]), "m", _BATCH, verify=False)
    assert res["truncated"] is True


# ------------------------------------------------------------------ options / prompts / snapshot ---
class RecordingLLM:
    """Answers every review with the same issues JSON and every verification with 'confirmed'."""

    def __init__(self, review_json):
        self.review_json = review_json
        self.calls = []

    def chat_with_stats(self, messages, model, temperature, max_tokens, timeout, reasoning_effort=None):
        self.calls.append(messages)
        if "CLAIMED ERROR" in messages[1]["content"] or messages[1]["content"].startswith("VERIFY"):
            return '{"verdict":"confirmed","reason":"ok"}', FakeStats()
        return self.review_json, FakeStats()


def _open_book(tmp_path):
    from app.core.db import open_db
    from tests.helpers import create_fake_book_db
    db_path = str(tmp_path / "book.db")
    create_fake_book_db(db_path)
    return open_db(db_path, readonly=True)


_QUOTES = {"147.2.1": "beginning was the Word", "147.2.2": "Jesus spoke", "147.2.3": "disciples listened",
           "147.2.4": "evening in Jerusalem"}


def _issue(pid, severity):
    return {"id": pid, "severity": severity, "category": "omissions", "problem": "p",
            "en_quote": _QUOTES[pid], "ru_quote": "", "why_significant": "", "suggestion": ""}


def test_batch_size_option_controls_number_of_llm_calls(tmp_path):
    conn = _open_book(tmp_path)
    llm = RecordingLLM("[]")
    TranslationAnalyzer(conn, llm, "m", options={"translation_batch_size": 2}).analyze(147, 2)
    assert len(llm.calls) == 2  # 4 paragraphs / 2 per request
    llm2 = RecordingLLM("[]")
    res = TranslationAnalyzer(conn, llm2, "m", options={"translation_batch_size": 30}).analyze(147, 2)
    assert len(llm2.calls) == 1 and res["experiment"]["retrieval"]["batches"] == 1


def test_min_severity_hides_smaller_issues_and_says_so(tmp_path):
    conn = _open_book(tmp_path)
    review = json.dumps([_issue("147.2.2", "minor"), _issue("147.2.3", "major"), _issue("147.2.4", "critical")])
    for threshold, expected in (("minor", 3), ("major", 2), ("critical", 1)):
        res = TranslationAnalyzer(conn, RecordingLLM(review), "m", options={
            "translation_min_severity": threshold, "translation_verify": False}).analyze(147, 2)
        assert len(res["issues"]) == expected, threshold
        assert res["below_threshold"] == 3 - expected
        if expected < 3:
            assert "Скрыто замечаний ниже выбранного порога" in res["summary"]


def test_verify_option_off_skips_second_pass(tmp_path):
    conn = _open_book(tmp_path)
    review = json.dumps([_issue("147.2.3", "major")])
    on, off = RecordingLLM(review), RecordingLLM(review)
    TranslationAnalyzer(conn, on, "m", options={"translation_verify": True}).analyze(147, 2)
    TranslationAnalyzer(conn, off, "m", options={"translation_verify": False}).analyze(147, 2)
    assert len(on.calls) == len(off.calls) + 1  # one candidate -> one verification call
    exp = TranslationAnalyzer(conn, RecordingLLM("[]"), "m", options={"translation_verify": False}).analyze(147, 2)["experiment"]
    assert exp["prompts"]["verify_system"] is None and "translation_verify" not in exp["prompt_texts"]


def test_custom_translation_prompts_are_used_and_recorded(tmp_path):
    from app import prompts as prompts_mod
    conn = _open_book(tmp_path)
    rp = prompts_mod.ResolvedPrompts({
        "translation_review": "MY REVIEW PROMPT", "translation_verify": "MY VERIFY PROMPT",
        "translation_verify_user": "VERIFY EN={en} RU={ru} CLAIM={problem} Q={en_quote}"}, preset="e1")
    llm = RecordingLLM(json.dumps([_issue("147.2.3", "major")]))
    res = TranslationAnalyzer(conn, llm, "m", prompts=rp, meta={"reasoning_setting": "off"}).analyze(147, 2)
    assert llm.calls[0][0]["content"] == "MY REVIEW PROMPT"
    verify_call = llm.calls[1]
    assert verify_call[0]["content"] == "MY VERIFY PROMPT"
    assert verify_call[1]["content"].startswith("VERIFY EN=") and "CLAIM=p" in verify_call[1]["content"]
    exp = res["experiment"]
    assert exp["kind"] == "translation" and exp["preset"] == "e1" and exp["reasoning"]["setting"] == "off"
    assert exp["prompts"]["system"] == "MY REVIEW PROMPT" and exp["prompts"]["verify_system"] == "MY VERIFY PROMPT"
    assert exp["prompts"]["user"].startswith("### ID 147.2.1")  # exact user message of the first batch
    assert exp["prompt_versions"]["translation_review"]["source"] == "custom"
    assert exp["options"]["translation_batch_size"] == 8 and exp["app"]["version"]
    assert exp["retrieval"]["batches"] == 1 and exp["timings"]["total_s"] >= 0
