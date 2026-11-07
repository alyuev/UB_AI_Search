"""Claude CLI provider: the `claude -p` process is replaced by a fake that prints stream-json lines."""
import io
import json

import pytest

from app.llm import claude_cli as C
from app.llm import providers as P


def _lines(text="Привет, мир", model="claude-sonnet-5-5", result=None):
    ev = [{"type": "system", "subtype": "init", "model": model}]
    for ch in [x for x in (text[:3], text[3:]) if x]:
        ev.append({"type": "stream_event", "event": {"type": "content_block_delta",
                                                     "delta": {"type": "text_delta", "text": ch}}})
    ev.append(result or {"type": "result", "subtype": "success", "is_error": False, "result": text,
                         "stop_reason": "end_turn", "total_cost_usd": 0.01,
                         "usage": {"input_tokens": 10, "cache_creation_input_tokens": 90, "output_tokens": 5},
                         "modelUsage": {model: {}}})
    return "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in ev)


class _Stdin(io.StringIO):
    def close(self):   # keep the text readable for the assertions
        pass


class FakeProc:
    calls = []

    def __init__(self, args, out="", err="", code=0, **kw):
        self.args, self.kw = args, kw
        self.stdin = _Stdin()
        self.stdout = io.StringIO(out)
        self.stderr = io.StringIO(err)
        self.returncode = code
        FakeProc.calls.append(self)

    def communicate(self, timeout=None):
        return self.stdout.read(), self.stderr.read()

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        pass


@pytest.fixture
def cli(monkeypatch):
    FakeProc.calls = []
    C._MODEL_CACHE.clear()
    outputs = []
    monkeypatch.setattr(C, "_which", lambda cmd: "/usr/bin/claude")

    def run(args, **kw):
        out = outputs.pop(0) if outputs else _lines()
        if isinstance(out, tuple):
            return FakeProc(args, *out, **kw)
        return FakeProc(args, out, **kw)
    monkeypatch.setattr(C, "_run", run)
    prof = P.normalize_profile({"id": "c", "name": "Claude", "kind": "claude_cli", "base_url": "claude"})
    return P.client_for({"allow_cloud_llm": True}, prof), outputs


def test_kind_is_cloud_and_gated():
    prof = P.normalize_profile({"id": "c", "name": "Claude", "kind": "claude_cli"})
    assert prof["base_url"] == "claude" and P.is_cloud("claude_cli", "claude")
    with pytest.raises(P.ProviderError, match="Облачный"):
        P.client_for({}, prof)


def test_stream_stats_and_bare_args(cli):
    c, _ = cli
    deltas = []
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "вопрос"}]
    text, st = c.chat_with_stats(msgs, "sonnet", on_delta=deltas.append, reasoning_effort="high")
    assert text == "Привет, мир" and "".join(deltas) == text
    assert st.model == "sonnet" and st.prompt_tokens == 100 and st.completion_tokens == 5
    assert st.cost is None
    p = FakeProc.calls[-1]
    a = p.args
    assert a[0] == "/usr/bin/claude" and "-p" in a and a[a.index("--model") + 1] == "sonnet"
    assert a[a.index("--effort") + 1] == "high" and a[-2:] == ["--tools", ""]
    for flag in ("--safe-mode", "--strict-mcp-config", "--no-session-persistence", "--system-prompt-file"):
        assert flag in a
    assert p.stdin.getvalue() == "вопрос"


def test_errors_and_fallback(cli):
    c, outputs = cli
    busy = {"type": "result", "subtype": "error", "is_error": True, "result": "Overloaded", "api_error_status": 529}
    outputs += [_lines("", result=busy), _lines("ok answer", model="claude-haiku-5-5")]
    text, st = c.chat_with_stats([{"role": "user", "content": "q"}], "opus", fallback_models=["haiku"])
    assert text == "ok answer" and st.model == "haiku" and st.requested_model == "opus" and "opus" in st.fallback_note
    outputs.append(_lines("", result={"type": "result", "subtype": "success", "is_error": True,
                                  "result": "Not logged in · Please run /login"}))
    with pytest.raises(P.ProviderError, match="вход"):
        c.chat_with_stats([{"role": "user", "content": "q"}], "sonnet")
    with pytest.raises(P.ProviderError, match="Недопустимое"):
        c.chat_with_stats([{"role": "user", "content": "q"}], "sonnet & calc")
    outputs.append(("", "error: unknown option '--safe-mode'", 1))
    with pytest.raises(P.ProviderError, match="claude update"):
        c.chat_with_stats([{"role": "user", "content": "q"}], "sonnet")


def test_check_connection_and_models(cli):
    c, outputs = cli
    outputs += [("2.1.318 (Claude Code)\n", "", 0), (json.dumps({"loggedIn": False}), "", 1)]
    r = c.check_connection()
    assert not r["ok"] and "/login" in r["message"]
    outputs += [("2.1.318 (Claude Code)\n", "", 0), (json.dumps({"loggedIn": True, "authMethod": "claude.ai"}), "", 0)]
    r = c.check_connection()
    assert r["ok"] and "2.1.318" in r["message"]
    assert [m["id"] for m in c.list_models()][:3] == ["sonnet", "opus", "haiku"]
    assert c.resolve_model(None)["model"] == "sonnet"


def test_missing_program(monkeypatch, cli):
    c, _ = cli
    monkeypatch.setattr(C, "_which", lambda cmd: None)
    assert "Не найдена программа" in c.check_connection()["message"]


def test_validate_command():
    assert P.validate_profile({"name": "x", "kind": "claude_cli", "base_url": ""})["base_url"] == "claude"
    assert P.validate_profile({"name": "x", "kind": "claude_cli",
                               "base_url": r"C:\Program Files\claude\claude.exe"})["base_url"].endswith("claude.exe")
    with pytest.raises(ValueError):
        P.validate_profile({"name": "x", "kind": "claude_cli", "base_url": "claude & calc"})


def test_multi_turn_flatten():
    sys_, prompt = C._flatten([{"role": "system", "content": "S"}, {"role": "user", "content": "a"},
                               {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"},
                               {"role": "assistant", "content": "<think>\n\n</think>\n\n"}])
    assert sys_ == "S" and "Пользователь:\na" in prompt and "Ассистент:\nb" in prompt and prompt.count("Ассистент") == 1


def test_models_from_cli_handshake(cli):
    c, outputs = cli
    models = [{"value": "default", "resolvedModel": "claude-opus-5-5", "displayName": "Default (recommended)",
               "description": "Opus 5.5 · Best"},
              {"value": "sonnet", "resolvedModel": "claude-sonnet-5-5", "displayName": "Sonnet 5.5"},
              {"value": "claude-opus-5", "resolvedModel": "claude-opus-5", "displayName": "Opus 5"},
              {"value": "bad name & x", "displayName": "skip me"}]
    resp = {"type": "control_response", "response": {"subtype": "success", "request_id": "models",
                                                     "response": {"models": models}}}
    outputs.append(json.dumps(resp) + "\n")
    rows = c.list_models()
    assert [m["id"] for m in rows] == ["sonnet", "claude-opus-5"]
    assert rows[0]["name"] == "Sonnet 5.5 — всегда последняя Sonnet (сейчас claude-sonnet-5-5)"
    assert rows[1]["name"] == "Opus 5"
    req = json.loads(FakeProc.calls[-1].stdin.getvalue())
    assert req["request"]["subtype"] == "initialize"
    n = len(FakeProc.calls)
    c.list_models()                      # cached: no second process
    assert len(FakeProc.calls) == n
