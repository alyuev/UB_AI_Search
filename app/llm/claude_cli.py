"""Claude CLI as an LLM provider: the `claude` program installed on this computer answers the questions.

Anyone who has Claude Code installed and is logged in (claude.ai subscription or ANTHROPIC_API_KEY) can use
it - no key is entered in SmartSearch. Each request starts `claude -p` in a bare mode: no tools, no MCP
servers, no CLAUDE.md / skills / hooks (--safe-mode), no saved session; our own system prompt replaces the
Claude Code one. The answer is read from the stream-json output, so the text appears as it is generated.
The texts DO leave the computer (they go to Anthropic), so the provider counts as a cloud one.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Callable, Optional

from app.llm.lmstudio import GenerationStats, LMStudioError, speed_figures, strip_reasoning
from app.llm.providers import ProviderError, _clean_models, _first_sentence

try:
    from app.log import get_logger
except ImportError:  # pragma: no cover
    from tests.fakes.log import get_logger

logger = get_logger("app")

DEFAULT_COMMAND = "claude"
DEFAULT_MODEL = "sonnet"
DEFAULT_CONTEXT = 200_000
HOST = "Claude CLI"
# Aliases the CLI itself resolves to the newest model of each family; a full model name works too.
MODELS = (
    ("sonnet", "Claude Sonnet — последняя версия (быстрая и умная, рекомендуется)"),
    ("opus", "Claude Opus — последняя версия (самая сильная, медленнее и дороже по лимитам)"),
    ("haiku", "Claude Haiku — последняя версия (самая быстрая)"),
    ("fable", "Claude Fable — последняя версия"),
)
_MODEL_RE = re.compile(r"^[A-Za-z0-9._:\-\[\]]{1,100}$")
_BAD_CMD_CHARS = set('"&|<>^%\n\r\t')
EFFORTS = {"none": "low", "low": "low", "medium": "medium", "high": "high"}

_run: Callable = subprocess.Popen          # tests: a fake Popen
_which: Callable = shutil.which
_now: Callable[[], float] = time.monotonic
MODEL_CACHE_TTL = 600.0
_MODEL_CACHE: dict[str, tuple[float, list[dict]]] = {}   # command -> (time, models the CLI reported)
_ALIAS_NOTE = {"default": "по умолчанию в Claude Code", "opus": "всегда последняя Opus", "sonnet": "всегда последняя Sonnet",
               "haiku": "всегда последняя Haiku", "fable": "всегда последняя Fable"}


def validate_command(cmd: str) -> str:
    """The program to run (usually just `claude`); raises ValueError with a plain-Russian message."""
    c = (cmd or "").strip() or DEFAULT_COMMAND
    if len(c) > 300 or any(ch in _BAD_CMD_CHARS for ch in c):
        raise ValueError("Программа: укажите команду запуска Claude CLI (обычно просто claude) или полный путь к ней, "
                         "без кавычек и спецсимволов")
    return c


def _flatten(messages: list[dict]) -> tuple[str, str]:
    """(system prompt, prompt): the CLI takes one prompt, so a multi-turn chat is written out as a transcript."""
    system = "\n\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "system").strip()
    rest = [m for m in messages if m.get("role") != "system"]
    if rest and rest[-1].get("role") == "assistant":   # a prefill (e.g. an empty <think> block) - not supported
        rest = rest[:-1]
    if len(rest) == 1:
        return system, str(rest[0].get("content") or "")
    names = {"user": "Пользователь", "assistant": "Ассистент"}
    lines = [f"{names.get(m.get('role'), m.get('role'))}:\n{m.get('content') or ''}" for m in rest]
    return system, "\n\n".join(lines) + "\n\nОтветь на последнее сообщение пользователя."


class ClaudeCliClient:
    """Same interface as ProviderClient (chat_with_stats, list_models, resolve_model, check_connection...)."""

    def __init__(self, profile: dict, timeout: Optional[float] = None):
        self.profile = profile
        self.kind = profile["kind"]
        self.command = profile.get("base_url") or DEFAULT_COMMAND
        self.base_url = self.command
        self.timeout = float(timeout or profile.get("timeout") or 300)
        self.cloud = True
        self.host = HOST
        self.api_key = None
        self.on_status: Optional[Callable[[str], None]] = None

    # -- process --------------------------------------------------------
    def _exe(self) -> str:
        found = _which(self.command)
        if not found:
            raise ProviderError(f"Не найдена программа Claude CLI («{self.command}»). Установите Claude Code "
                                f"(https://claude.com/claude-code) и войдите в него, либо укажите в настройках "
                                f"провайдера полный путь к программе.")
        return found

    @staticmethod
    def _popen_kw() -> dict:
        kw = {"stdin": subprocess.PIPE, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
              "encoding": "utf-8", "errors": "replace", "bufsize": 1}
        if os.name == "nt":
            kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return kw

    def _workdir(self) -> str:
        """An empty folder: the CLI must not pick up any project files (CLAUDE.md etc.) from the current one."""
        d = os.path.join(tempfile.gettempdir(), "smartsearch-claude-cli")
        os.makedirs(d, exist_ok=True)
        return d

    def _quick(self, args: list[str], timeout: float = 30) -> tuple[int, str, str]:
        try:
            p = _run([self._exe(), *args], cwd=self._workdir(), **self._popen_kw())
            out, err = p.communicate(timeout=timeout)
            return p.returncode, out or "", err or ""
        except subprocess.TimeoutExpired:
            p.kill()
            raise ProviderError(f"Claude CLI не ответил за {timeout:g} сек.")
        except OSError as exc:
            raise ProviderError(f"Не удалось запустить Claude CLI: {exc}") from exc

    # -- models ---------------------------------------------------------
    def check_connection(self) -> dict:
        try:
            t0 = time.monotonic()
            code, out, err = self._quick(["--version"], timeout=30)
            if code != 0:
                raise ProviderError(f"Claude CLI завершился с ошибкой: {(err or out).strip()[:200]}")
            version = out.strip().split(" ")[0]
            code, out, err = self._quick(["auth", "status"], timeout=30)
            how = ""
            try:
                st = json.loads(out)
                if st.get("loggedIn") is False:
                    raise ProviderError("Claude CLI установлен, но вход не выполнен. Откройте терминал, запустите "
                                        "claude и войдите (команда /login), затем проверьте снова.")
                how = {"claude.ai": "подписка claude.ai", "api_key": "ключ API"}.get(st.get("authMethod"),
                                                                                       st.get("authMethod") or "")
            except (ValueError, AttributeError):
                pass   # an older CLI without `auth status`: the first real request will tell
            ms = int((time.monotonic() - t0) * 1000)
            return {"ok": True, "latency_ms": ms,
                    "message": f"Claude CLI {version} доступен" + (f" (вход: {how})" if how else "")}
        except LMStudioError as exc:
            return {"ok": False, "message": str(exc)}

    def account_state(self) -> dict:
        return {"available": False, "reason": "Состояние счёта показывается только для OpenRouter."}

    def _cli_models(self, timeout: float = 30) -> list[dict]:
        """The models this account may use, as Claude Code itself lists them (the SDK `initialize` handshake:
        aliases with the version they point to, plus concrete versions such as claude-opus-5). No request to
        the model is made. Raises ProviderError."""
        args = ["-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose", "--safe-mode",
                "--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands", "--tools", ""]
        try:
            proc = _run([self._exe(), *args], cwd=self._workdir(), **self._popen_kw())
        except OSError as exc:
            raise ProviderError(f"Не удалось запустить Claude CLI: {exc}") from exc
        timer = threading.Timer(timeout, proc.kill)
        timer.daemon = True
        timer.start()
        models = None
        try:
            proc.stdin.write(json.dumps({"type": "control_request", "request_id": "models",
                                         "request": {"subtype": "initialize"}}) + "\n")
            proc.stdin.flush()
            for line in proc.stdout:
                if '"control_response"' not in line:
                    continue
                try:
                    resp = (json.loads(line).get("response") or {}).get("response") or {}
                except (ValueError, AttributeError):
                    continue
                models = resp.get("models")
                break
        except (OSError, ValueError):
            pass
        finally:
            timer.cancel()
            try:
                proc.stdin.close()
            except (OSError, ValueError):
                pass
            if proc.poll() is None:
                proc.kill()
        if not isinstance(models, list) or not models:
            raise ProviderError("Claude CLI не сообщил список моделей (возможно, версия слишком старая).")
        rows = []
        for m in models:
            if not isinstance(m, dict) or not _MODEL_RE.match(str(m.get("value") or "")):
                continue
            mid, real = m["value"], str(m.get("resolvedModel") or "")
            title = str(m.get("displayName") or mid)
            if mid == "default":
                title = (str(m.get("description") or "").split("·")[0].strip() or title)
            name = f"{title} — {_ALIAS_NOTE[mid]} (сейчас {real or title})" if mid in _ALIAS_NOTE else title
            rows.append({"id": mid, "name": name, "resolved": real or mid})
        return rows

    def list_models(self, include_embeddings: bool = False, refresh: bool = False) -> list[dict]:
        hit = _MODEL_CACHE.get(self.command)
        if hit and not refresh and _now() - hit[0] < MODEL_CACHE_TTL:
            found = hit[1]
        else:
            try:
                found = self._cli_models()
                _MODEL_CACHE[self.command] = (_now(), found)
            except ProviderError as exc:
                logger.info("Claude CLI model list unavailable (%s): built-in aliases used", exc)
                found = []
        rows = [(m["id"], m["name"]) for m in found if m["id"] != "default"] or [(m, d) for m, d in MODELS]
        ids = [m for m, _ in rows]
        for extra in [self.profile.get("default_model") or ""] + list(self.profile.get("fallback_models") or []):
            if extra and extra not in ids:
                rows.append((extra, extra))
                ids.append(extra)
        ctx = self.profile.get("context_length") or DEFAULT_CONTEXT
        return [{"id": m, "name": d, "context_length": ctx, "loaded": None, "price_in": None, "price_out": None,
                 "free": False, "supports_reasoning": True, "kind": self.kind, "is_embedding": False} for m, d in rows]

    def discover_context(self, model: str) -> Optional[int]:
        return int(self.profile.get("context_length") or DEFAULT_CONTEXT)

    def resolve_model(self, configured: Optional[str]) -> dict:
        model = configured or self.profile.get("default_model") or DEFAULT_MODEL
        return {"model": model, "context": self.discover_context(model), "note": None}

    # -- chat -----------------------------------------------------------
    def _args(self, model: str, system_file: Optional[str], reasoning_effort: Optional[str]) -> list[str]:
        a = ["-p", "--output-format", "stream-json", "--verbose", "--include-partial-messages",
             "--model", model, "--safe-mode", "--strict-mcp-config", "--no-session-persistence",
             "--disable-slash-commands"]
        if system_file:
            a += ["--system-prompt-file", system_file]
        if reasoning_effort in EFFORTS:
            a += ["--effort", EFFORTS[reasoning_effort]]
        return a + ["--tools", ""]

    def _error_from_result(self, res: dict, model: str, stderr: str) -> ProviderError:
        text = str(res.get("result") or "").strip()
        status = res.get("api_error_status")
        low = (text + " " + stderr).lower()
        if "unrecognized_model" in low or "model" in low and ("not found" in low or "not_found" in low or "invalid" in low):
            return ProviderError(f"Claude CLI не знает модель «{model}». Выберите sonnet, opus или haiku "
                                 f"(или полное название модели).", kind="model")
        if "login" in low or "not logged" in low or "authentication" in low or status == 401:
            return ProviderError("Claude CLI: вход не выполнен или истёк. Откройте терминал, запустите claude и "
                                 "войдите (команда /login).", kind="auth")
        if status in (429, 529) or "rate limit" in low or "overloaded" in low or "usage limit" in low:
            return ProviderError(f"Claude: превышен лимит или сервис перегружен ({_first_sentence(text) or status}). "
                                 f"Подождите немного или выберите другую модель.", retryable=True, kind="provider_busy")
        if isinstance(status, int) and status >= 500:
            return ProviderError(f"Claude: ошибка на стороне сервиса ({status}). Попробуйте позже.",
                                 retryable=True, kind="server")
        detail = _first_sentence(text) or _first_sentence(stderr.strip()) or str(res.get("terminal_reason") or "")
        return ProviderError(f"Claude CLI вернул ошибку: {detail or 'без пояснений'}")

    def _chat_once(self, messages, model, eff, reasoning_effort, on_delta) -> tuple[str, GenerationStats]:
        if not _MODEL_RE.match(model or ""):
            raise ProviderError(f"Недопустимое название модели «{model}».")
        system, prompt = _flatten(messages)
        sys_file = None
        if system:
            fd, sys_file = tempfile.mkstemp(prefix="sys-", suffix=".txt", dir=self._workdir())
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(system)
        env = dict(os.environ)
        if reasoning_effort == "none":
            env["MAX_THINKING_TOKENS"] = "0"
        t0 = time.monotonic()
        first_at: Optional[float] = None
        parts: list[str] = []
        result: Optional[dict] = None
        answered = model
        err_tail: list[str] = []
        timed_out = threading.Event()
        try:
            try:
                proc = _run([self._exe(), *self._args(model, sys_file, reasoning_effort)], cwd=self._workdir(),
                            env=env, **self._popen_kw())
            except OSError as exc:
                raise ProviderError(f"Не удалось запустить Claude CLI: {exc}") from exc

            def kill():
                timed_out.set()
                proc.kill()

            timer = threading.Timer(eff, kill)
            timer.daemon = True
            timer.start()

            def read_err():
                for line in proc.stderr:
                    err_tail.append(line)
                    del err_tail[:-50]

            et = threading.Thread(target=read_err, daemon=True)
            et.start()
            try:
                try:
                    proc.stdin.write(prompt)
                    proc.stdin.close()
                except (OSError, ValueError):
                    pass   # the process died at once: its stderr / exit code tell why
                for line in proc.stdout:
                    line = line.strip()
                    if not line.startswith("{"):
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    t = ev.get("type")
                    if t == "system" and ev.get("subtype") == "init" and ev.get("model"):
                        answered = ev["model"]
                    elif t == "stream_event":
                        e = ev.get("event") or {}
                        if e.get("type") == "content_block_delta":
                            d = e.get("delta") or {}
                            if d.get("type") in ("text_delta", "thinking_delta") and first_at is None:
                                first_at = time.monotonic()
                            if d.get("type") == "text_delta" and d.get("text"):
                                parts.append(d["text"])
                                if on_delta:
                                    on_delta(d["text"])
                        elif e.get("type") == "message_start":
                            answered = (e.get("message") or {}).get("model") or answered
                    elif t == "result":
                        result = ev
                proc.wait(timeout=10)
            finally:
                timer.cancel()
                if proc.poll() is None:
                    proc.kill()
                et.join(timeout=2)
        finally:
            if sys_file:
                try:
                    os.remove(sys_file)
                except OSError:
                    pass
        elapsed = time.monotonic() - t0
        stderr = "".join(err_tail)
        if timed_out.is_set():
            err = ProviderError(f"Claude CLI не ответил за {eff:g} сек. Увеличьте таймаут в настройках провайдера.")
            if parts:
                err.partial, _ = strip_reasoning("".join(parts), keep=True)
            raise err
        if result is None:
            if "unknown option" in stderr:
                raise ProviderError("Установленная версия Claude CLI слишком старая. Обновите её командой "
                                    "«claude update» в терминале.")
            raise ProviderError("Claude CLI завершился без ответа"
                                + (f": {_first_sentence(stderr.strip())}" if stderr.strip() else ".")
                                + " Проверьте, что вход выполнен: запустите claude в терминале.")
        if result.get("is_error") or result.get("subtype") != "success":
            err = self._error_from_result(result, model, stderr)
            if parts:
                err.partial, _ = strip_reasoning("".join(parts), keep=True)
                err.retryable = False
            raise err
        text = "".join(parts) or str(result.get("result") or "")
        usage = result.get("usage") or {}
        mu = result.get("modelUsage") or {}
        if len(mu) == 1:
            answered = next(iter(mu))
        pt = sum(int(usage.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens",
                                                   "cache_read_input_tokens")) or None
        ct = usage.get("output_tokens")
        rt = (usage.get("output_tokens_details") or {}).get("thinking_tokens")
        ttft = (first_at - t0) if first_at is not None else None
        gen_tps, overall, gen_time = speed_figures(ct, elapsed, ttft)
        # the alias asked for (sonnet) stays the model name: the UI takes any other name for a backup model
        stats = GenerationStats(
            model=model, elapsed_s=round(elapsed, 2), ttft_s=round(ttft, 2) if ttft is not None else None,
            prompt_tokens=pt, completion_tokens=ct, reasoning_tokens=rt or None,
            total_tokens=(pt or 0) + (ct or 0) if (pt or ct) else None,
            tokens_per_sec=round(gen_tps, 1) if gen_tps is not None else None,
            stop_reason=result.get("stop_reason"),
            # total_cost_usd is a list-price figure: a claude.ai subscriber pays nothing per request -> not shown
            cost=None, cost_estimated=False,
            overall_tokens_per_sec=round(overall, 1) if overall is not None else None,
            gen_time_s=round(gen_time, 2) if gen_time is not None else None, stats_source="расчёт",
            requested_model=None)
        logger.info("LLM host=%s model=%s (%s) status=ok elapsed=%.1fs", HOST, model, answered, elapsed)
        visible, _ = strip_reasoning(text, keep=True)
        return visible, stats

    def chat_with_stats(self, messages: list[dict], model: str, temperature: float = 0.2,
                        max_tokens: int = 1024, timeout: Optional[float] = None,
                        reasoning_effort: Optional[str] = None,
                        on_delta: Optional[Callable[[str], None]] = None,
                        fallback_models: Optional[list] = None,
                        on_status: Optional[Callable[[str], None]] = None) -> tuple[str, GenerationStats]:
        """temperature / max_tokens are not supported by the CLI and are ignored."""
        if not model:
            raise ProviderError("Не выбрана модель.")
        eff = float(timeout or self.timeout)
        backups = fallback_models if fallback_models is not None else self.profile.get("fallback_models")
        cands = [model] + [m for m in _clean_models(backups) if m != model]
        first_err: Optional[str] = None
        for i, m in enumerate(cands):
            try:
                text, stats = self._chat_once(messages, m, eff, reasoning_effort, on_delta)
            except ProviderError as exc:
                if exc.partial or not exc.retryable or i + 1 >= len(cands):
                    raise
                first_err = first_err or str(exc)
                if on_status:
                    on_status(f"Модель «{m}» не ответила — пробую запасную «{cands[i + 1]}»…")
                continue
            if i:
                stats.requested_model = model
                stats.fallback_note = (f"Основная модель «{model}» не ответила ({_first_sentence(first_err)}); "
                                       f"ответила «{m}».")
            return text, stats
        raise ProviderError("Не удалось получить ответ от модели.")  # pragma: no cover

    def chat(self, messages: list[dict], model: str, temperature: float = 0.2, max_tokens: int = 1024,
             timeout: Optional[float] = None, stream: bool = False, reasoning_effort: Optional[str] = None) -> str:
        return self.chat_with_stats(messages, model, temperature, max_tokens, timeout, reasoning_effort)[0]

    def test_completion(self, model: str) -> dict:
        try:
            text, st = self.chat_with_stats([{"role": "user", "content": "Ответь одним словом: привет"}],
                                            model=model, timeout=min(self.timeout, 120))
            return {"ok": True, "latency_s": st.elapsed_s, "cost": None, "cost_estimated": False, "model": model,
                    "reply": text[:80], "message": f"Модель ответила за {st.elapsed_s} с"}
        except LMStudioError as exc:
            return {"ok": False, "model": model, "message": str(exc)}

