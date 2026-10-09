"""OpenRouter errors in plain Russian, backup models, retry policy of free models, partial answers,
account state, and the corrected speed figures."""
import json

import httpx
import pytest

from app.llm import provider_errors as PE
from app.llm import providers as P
from app.llm.lmstudio import GenerationStats, speed_figures
from tests.provider_fakes import FakeOpenAI, sse, sse_events

KEY = "sk-or-v1-SECRETSECRETSECRET1234"
MSG = [{"role": "user", "content": "hi"}]
FREE = "google/gemma-4-26b-a4b-it:free"

BUSY_429 = {"error": {"message": "Provider returned error", "code": 429,
                      "metadata": {"provider_name": "Google AI Studio",
                                   "raw": "google/gemma-4-26b-a4b-it:free is temporarily rate-limited upstream."}}}


def mk(kind="openrouter", fake=None, **prof):
    fake = fake or FakeOpenAI()
    P._TEST_TRANSPORT = fake.transport()
    profile = P.normalize_profile({"id": "p1", "name": {"openrouter": "OpenRouter", "ollama": "Ollama"}.get(kind, "LM Studio (локально)"),
                                   "kind": kind, "base_url": P.KINDS[kind]["default_url"], **prof})
    return P.ProviderClient(profile, api_key=KEY if kind == "openrouter" else None, timeout=5), fake


# ------------------------------------------------------------- error texts ---

def test_provider_busy_is_not_called_account_limit():
    e = PE.parse_error(BUSY_429)
    assert (e["code"], e["provider"]) == (429, "Google AI Studio")
    msg, kind = PE.explain(e, "OpenRouter", FREE, 429)
    assert kind == "provider_busy" and PE.is_retryable(kind)
    assert "Провайдер модели (Google AI Studio) перегружен" in msg
    assert "не лимит вашей учётной записи" in msg and "openrouter/free" in msg and "пополнение баланса" in msg
    assert "temporarily rate-limited" in msg           # the provider's own words are kept
    assert "LM Studio" not in msg


def test_midstream_upstream_error_names_the_company():
    body = {"error": {"message": "Upstream error from Nvidia: ResourceExhausted: Worker local total request limit "
                                 "reached (16/16)", "code": 502}}
    e = PE.parse_error(body)
    assert e["provider"] == "Nvidia"
    msg, kind = PE.explain(e, "OpenRouter", "nvidia/x:free", None, stream=True)
    assert kind == "provider_busy" and "(Nvidia)" in msg and "Ответ оборвался" in msg and "16/16" in msg


@pytest.mark.parametrize("raw,kind,word", [
    ("Rate limit exceeded: free-models-per-day. Add 10 credits", "account_day", "дневной лимит"),
    ("Rate limit exceeded: free-models-per-min", "account_minute", "в минуту"),
])
def test_account_quota_is_recognised(raw, kind, word):
    e = PE.parse_error({"error": {"message": raw, "code": 429}})
    msg, k = PE.explain(e, "OpenRouter", FREE, 429)
    assert k == kind and word in msg and "вашей учётной записи" in msg
    assert not PE.is_retryable(k)                      # retrying cannot help and wastes quota


def test_raw_can_be_a_json_string_or_object():
    e = PE.parse_error({"error": {"message": "Provider returned error", "code": 429, "metadata": {
        "provider_name": "X", "raw": json.dumps({"error": {"message": "slow down"}})}}})
    assert e["raw"] == "slow down"
    e = PE.parse_error({"error": {"message": "m", "metadata": {"raw": {"message": "obj text"}}}})
    assert e["raw"] == "obj text"
    assert PE.parse_error("not json")["message"] == "not json" and PE.parse_error(None)["code"] is None


def test_http_429_message_through_client_and_no_key_in_text():
    fake = FakeOpenAI()
    fake.script = [httpx.Response(429, json=BUSY_429)] * 3
    c, _ = mk(fake=fake)
    with pytest.raises(P.ProviderError) as ei:
        c.chat_with_stats(MSG, FREE)
    assert ei.value.retryable and ei.value.kind == "provider_busy"
    assert "Google AI Studio" in str(ei.value) and KEY not in str(ei.value)


def test_non_openrouter_429_does_not_mention_openrouter():
    fake = FakeOpenAI()
    fake.script = [httpx.Response(429, json={"error": {"message": "busy"}})] * 5
    c, _ = mk("ollama", fake=fake)
    with pytest.raises(P.ProviderError) as ei:
        c.chat_with_stats(MSG, "m")
    assert "openrouter" not in str(ei.value).lower() and "LM Studio" not in str(ei.value)


# --------------------------------------------------------- retry policy ---

def test_free_model_429_is_retried_only_once(monkeypatch):
    sleeps = []
    monkeypatch.setattr(P, "_sleep", sleeps.append)
    fake = FakeOpenAI()
    fake.script = [httpx.Response(429, json=BUSY_429)] * 6
    c, _ = mk(fake=fake)
    with pytest.raises(P.ProviderError):
        c.chat_with_stats(MSG, FREE)
    assert len(fake.chats) == 2 and sleeps == [1.0]     # one short retry, then give up


def test_free_model_honours_retry_after(monkeypatch):
    sleeps = []
    monkeypatch.setattr(P, "_sleep", sleeps.append)
    fake = FakeOpenAI()
    fake.script = [httpx.Response(429, headers={"retry-after": "7"}), httpx.Response(429, headers={"retry-after": "7"})]
    c, _ = mk(fake=fake)
    text, _ = c.chat_with_stats(MSG, FREE)
    assert text == fake.reply and sleeps == [7.0, 7.0]


# ---------------------------------------------------------- backup models ---

def test_backup_model_answers_after_primary_is_busy():
    fake = FakeOpenAI(reply="Из запасной.")
    fake.script = [httpx.Response(429, json=BUSY_429)] * 4           # primary: first try + 3 retries
    c, _ = mk("ollama", fake=fake, fallback_models=["nvidia/nemotron:free"])
    status = []
    text, st = c.chat_with_stats(MSG, FREE, on_status=status.append)
    assert text == "Из запасной."
    assert [ch["body"]["model"] for ch in fake.chats] == [FREE] * 4 + ["nvidia/nemotron:free"]
    assert st.model == "nvidia/nemotron:free" and st.requested_model == FREE
    assert "не ответила" in st.fallback_note and "ответила «nvidia/nemotron:free»" in st.fallback_note
    assert status and "запасную «nvidia/nemotron:free»" in status[0]
    assert "ответила модель «nvidia/nemotron:free» (вместо «" + FREE + "»)" in st.summary_line()


def test_openrouter_gets_native_models_array_and_reports_who_answered():
    fake = FakeOpenAI()
    fake.script = [httpx.Response(200, content=sse("ok", {"prompt_tokens": 5, "completion_tokens": 3},
                                                  model="b/second:free"), headers={"content-type": "text/event-stream"})]
    c, _ = mk(fake=fake, fallback_models=["b/second:free", "c/third:free"])
    text, st = c.chat_with_stats(MSG, "a/first:free")
    body = fake.last_chat()["body"]
    assert body["models"] == ["a/first:free", "b/second:free", "c/third:free"] and body["model"] == "a/first:free"
    assert st.model == "b/second:free" and "переключил" in st.fallback_note


def test_non_openrouter_has_no_models_array_and_uses_backup_in_a_loop():
    fake = FakeOpenAI()
    fake.script = [httpx.Response(503, json={"error": {"message": "down"}})]
    c, _ = mk("ollama", fake=fake, fallback_models=["m2"])
    text, st = c.chat_with_stats(MSG, "m1")
    assert [ch["body"]["model"] for ch in fake.chats] == ["m1", "m2"] and all("models" not in ch["body"] for ch in fake.chats)
    assert st.model == "m2"


def test_all_models_fail_lists_what_was_tried():
    fake = FakeOpenAI()
    fake.script = [httpx.Response(503, json={"error": {"message": "down"}})] * 3
    c, _ = mk("ollama", fake=fake, fallback_models=["m2", "m3"])
    with pytest.raises(P.ProviderError) as ei:
        c.chat_with_stats(MSG, "m1")
    assert "«m1», «m2», «m3»" in str(ei.value) and len(fake.chats) == 3


def test_account_quota_error_is_not_retried_and_does_not_switch_models(monkeypatch):
    sleeps = []
    monkeypatch.setattr(P, "_sleep", sleeps.append)
    fake = FakeOpenAI()
    fake.script = [httpx.Response(429, json={"error": {"message": "Rate limit exceeded: free-models-per-day", "code": 429}})] * 3
    c, _ = mk(fake=fake, fallback_models=["x:free"])
    with pytest.raises(P.ProviderError) as ei:
        c.chat_with_stats(MSG, "a:free")
    assert ei.value.kind == "account_day" and not ei.value.retryable
    assert len(fake.chats) == 1 and sleeps == []          # a cap of the whole account: waiting/retrying only wastes quota


def test_fallback_models_validation_and_profile():
    prof = P.validate_profile({"fallback_models": ["a", " b ", "a", ""]}, partial=True)
    assert prof["fallback_models"] == ["a", "b"]
    assert P.validate_profile({"fallback_models": "x\ny, z"}, partial=True)["fallback_models"] == ["x", "y", "z"]
    with pytest.raises(ValueError):
        P.validate_profile({"fallback_models": ["a"] * 6 + ["b"]}, partial=True)
    with pytest.raises(ValueError):
        P.validate_profile({"fallback_models": [5]}, partial=True)
    assert P.normalize_profile({"id": "x", "kind": "ollama"})["fallback_models"] == []


# ----------------------------------------------------- mid-stream failures ---

def test_error_event_before_any_text_is_retryable_and_triggers_backup():
    fake = FakeOpenAI(reply="Запасная")
    err = {"error": {"message": "Upstream error from Nvidia: ResourceExhausted: limit reached (16/16)", "code": 502}}
    fake.script = [httpx.Response(200, content=sse_events([err]), headers={"content-type": "text/event-stream"})]
    c, _ = mk("ollama", fake=fake, fallback_models=["m2"])
    text, st = c.chat_with_stats(MSG, "m1")
    assert text == "Запасная" and st.model == "m2"


def test_error_after_partial_text_keeps_the_partial_answer():
    fake = FakeOpenAI()
    events = [{"choices": [{"delta": {"content": "Начало ответа "}}]}, {"choices": [{"delta": {"content": "[0:1.2]"}}]},
              {"error": {"message": "Upstream error from Nvidia: ResourceExhausted: limit reached (16/16)", "code": 502,
                         "metadata": {"provider_name": "Nvidia"}}}]
    fake.script = [httpx.Response(200, content=sse_events(events), headers={"content-type": "text/event-stream"})]
    c, _ = mk(fake=fake, fallback_models=["b:free"])
    with pytest.raises(P.ProviderError) as ei:
        c.chat_with_stats(MSG, "a:free")
    e = ei.value
    assert e.partial == "Начало ответа [0:1.2]" and not e.retryable      # no silent switch: text is already there
    assert "Nvidia" in str(e) and "Ответ оборвался" in str(e)
    assert e.partial_stats is not None and e.partial_stats.model == "a:free"
    assert len(fake.chats) == 1


# ------------------------------------------------------------ account state ---

def test_account_state_in_plain_words():
    fake = FakeOpenAI(native={"/api/v1/key": {"data": {"label": "k", "is_free_tier": True, "usage": 0.5,
                                                       "limit": 5, "limit_remaining": 4.5}}})
    c, _ = mk(fake=fake)
    r = c.account_state()
    assert r["available"] and r["is_free_tier"] is True and r["limit_remaining"] == 4.5
    assert "ни разу не пополнялся" in r["summary"] and "осталось $4.5 из $5" in r["summary"]
    assert "$0.5000" in r["summary"]
    paid = PE.describe_account({"data": {"is_free_tier": False, "usage": 0, "limit": None}})
    assert "пополнялся" in paid["summary"] and "не задан" in paid["summary"] and "ничего не потрачено" in paid["summary"]


def test_account_state_only_for_openrouter():
    c, _ = mk("ollama")
    assert c.account_state()["available"] is False


# ----------------------------------------------------------------- speed ---

def test_speed_figures_exclude_the_wait_for_the_first_token():
    gen, overall, gt = speed_figures(233, 19.0, 10.6)
    assert gt == pytest.approx(8.4) and gen == pytest.approx(232 / 8.4) and overall == pytest.approx(233 / 19)
    assert round(gen, 1) == 27.6 and overall < gen
    # nothing to measure -> the overall figure; no tokens -> nothing
    assert speed_figures(2, 5.0, 4.99)[0] == pytest.approx(2 / 5.0)
    assert speed_figures(None, 5.0, 1.0) == (None, None, None)


def test_summary_line_format():
    st = GenerationStats(model="m", elapsed_s=19.0, ttft_s=10.6, prompt_tokens=3175, completion_tokens=233,
                         reasoning_tokens=0, total_tokens=3408, tokens_per_sec=16.4, stop_reason="stop")
    assert st.summary_line() == "генерация 16.4 ток/с · до первого токена 10.6 с · всего 19 с · 3175 → 233 ток."
    st.cost = 0.0123
    st.stats_source = "LM Studio"
    assert st.summary_line().endswith("стоимость ≈ $0.0123 · скорость по данным LM Studio")
    assert "LM Studio" not in GenerationStats(model="m", elapsed_s=1, ttft_s=None, prompt_tokens=None,
                                              completion_tokens=None, reasoning_tokens=None, total_tokens=None,
                                              tokens_per_sec=None, stop_reason=None).summary_line()


def test_first_token_is_measured_on_reasoning_too(monkeypatch):
    clock = iter([0.0, 4.0, 10.0])        # request start, first delta (a REASONING one), end
    monkeypatch.setattr(P.time, "monotonic", lambda: next(clock))
    events = [{"choices": [{"delta": {"reasoning_content": "думаю"}}]},
              {"choices": [{"delta": {"content": "ответ"}}]},
              {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 61,
                                        "completion_tokens_details": {"reasoning_tokens": 40}}}]
    fake = FakeOpenAI()
    fake.script = [httpx.Response(200, content=sse_events(events), headers={"content-type": "text/event-stream"})]
    c, _ = mk("ollama", fake=fake)
    text, st = c.chat_with_stats(MSG, "m")
    assert text == "ответ" and st.ttft_s == 4.0 and st.elapsed_s == 10.0
    assert st.tokens_per_sec == 10.0 and st.overall_tokens_per_sec == 6.1 and st.gen_time_s == 6.0
    assert st.stats_source == "расчёт" and st.reasoning_tokens == 40


def test_lmstudio_native_statistics_win_when_available():
    native = {"choices": [{"message": {"content": "1 2 3"}, "finish_reason": "stop"}],
              "usage": {"prompt_tokens": 24, "completion_tokens": 111},
              "stats": {"tokens_per_second": 22.4, "time_to_first_token": 0.76, "generation_time": 5.68}}
    fake = FakeOpenAI(native={"/api/v0/chat/completions": native})
    c, _ = mk("lmstudio", fake=fake)
    text, st = c.chat_with_stats(MSG, "m")                       # no on_delta: the native endpoint is used
    assert text == "1 2 3" and st.tokens_per_sec == 22.4 and st.ttft_s == 0.76 and st.gen_time_s == 5.68
    assert st.stats_source == "LM Studio" and "по данным LM Studio" in st.summary_line()
    assert fake.last_chat is not None and not fake.chats       # went to the native path, not /v1
    # with a live preview (on_delta) the stream is used and the figure is calculated
    text, st = c.chat_with_stats(MSG, "m", on_delta=lambda t: None)
    assert st.stats_source == "расчёт" and fake.chats


def test_lmstudio_native_endpoint_missing_falls_back_to_stream():
    fake = FakeOpenAI(native={"/api/v0/chat/completions": httpx.Response(404, json={"error": "nope"})})
    c, _ = mk("lmstudio", fake=fake)
    text, st = c.chat_with_stats(MSG, "m")
    assert text == fake.reply and st.stats_source == "расчёт"
    c.chat_with_stats(MSG, "m")
    assert len(fake.chats) == 2                                  # the failed native probe is not repeated
