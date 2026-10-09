"""LM Studio HTTP client: connection check, model listing, chat completion (with SSE streaming).

The URL and model are always configurable (from app.config settings) - never hardcoded.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Iterator, Optional

import httpx

try:
    from app.log import get_logger
except ImportError:  # pragma: no cover
    from tests.fakes.log import get_logger

logger = get_logger("app")


class LMStudioError(Exception):
    """Base error for LM Studio client problems."""


class ConnectionRefused(LMStudioError):
    def __init__(self, url: str):
        super().__init__(f"Не удалось подключиться к LM Studio по адресу {url}. "
                          f"Убедитесь, что LM Studio запущен и сервер включён.")


class LMTimeout(LMStudioError):
    def __init__(self, timeout: float):
        super().__init__(f"LM Studio не ответил за {timeout} сек. Попробуйте увеличить таймаут "
                          f"в настройках или проверьте, не перегружена ли модель.")


class NoModelLoaded(LMStudioError):
    def __init__(self, hint: str = ""):
        super().__init__("В LM Studio не загружена ни одна модель. Загрузите модель в LM Studio "
                          "и выберите её в настройках." + hint)


@dataclass
class ModelInfo:
    id: str
    state: str = "unknown"  # loaded|not-loaded|unknown
    max_context_length: Optional[int] = None
    loaded_context_length: Optional[int] = None  # only meaningful when state == "loaded"
    is_embedding: bool = False

    @property
    def effective_context(self) -> Optional[int]:
        """Context actually available for requests: the loaded context if the model is
        loaded (it is usually far smaller than max_context_length), else the max."""
        if self.state == "loaded" and self.loaded_context_length:
            return self.loaded_context_length
        return self.max_context_length

    def to_dict(self):
        return {
            "id": self.id,
            "state": self.state,
            "max_context_length": self.max_context_length,
            "loaded_context_length": self.loaded_context_length,
            "effective_context": self.effective_context,
            "is_embedding": self.is_embedding,
        }


@dataclass
class GenerationStats:
    """Per-call generation metrics, gathered from streaming + the final usage payload.

    tokens_per_sec is the GENERATION speed: tokens / (total time - time to the first token), i.e. without
    the wait for prompt processing - the same figure LM Studio shows in its own log. The overall figure
    (tokens / total time) is kept as overall_tokens_per_sec. stats_source says where the numbers come from:
    "LM Studio" (the server's own statistics) or "расчёт" (measured by this program).
    """
    model: str                       # the model that really answered
    elapsed_s: float
    ttft_s: Optional[float]          # time to the first token of ANY kind (reasoning included), seconds
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    reasoning_tokens: Optional[int]
    total_tokens: Optional[int]
    tokens_per_sec: Optional[float]  # generation speed
    stop_reason: Optional[str]
    cost: Optional[float] = None        # USD for this call (cloud providers); None = unknown / local
    cost_estimated: bool = False        # True when computed from the price list rather than reported
    overall_tokens_per_sec: Optional[float] = None   # tokens / total time (includes waiting for the first token)
    gen_time_s: Optional[float] = None               # time spent generating (total - time to first token)
    stats_source: str = "расчёт"
    requested_model: Optional[str] = None            # set when another (fallback) model answered
    fallback_note: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_line(self) -> str:
        """Plain-Russian summary, e.g.
        'генерация 16.4 ток/с · до первого токена 10.6 с · всего 19 с · 3175 → 233 ток.'"""
        bits = []
        if self.requested_model and self.requested_model != self.model:
            bits.append(f"ответила модель «{self.model}» (вместо «{self.requested_model}»)")
        if self.tokens_per_sec is not None:
            bits.append(f"генерация {self.tokens_per_sec} ток/с")
        if self.ttft_s is not None:
            bits.append(f"до первого токена {self.ttft_s:.1f} с")
        bits.append(f"всего {fmt_seconds(self.elapsed_s)} с")
        if self.prompt_tokens is not None and self.completion_tokens is not None:
            bits.append(f"{self.prompt_tokens} → {self.completion_tokens} ток.")
        if self.cost is not None:
            bits.append("бесплатно" if self.cost == 0 else f"стоимость ≈ ${self.cost:.4f}")
        if self.stats_source == "LM Studio":
            bits.append("скорость по данным LM Studio")
        return " · ".join(bits)


def llm_title(llm) -> str:
    """Display name of the provider behind an LLM client («OpenRouter», «LM Studio (локально)»)."""
    prof = getattr(llm, "profile", None)
    if isinstance(prof, dict) and prof.get("name"):
        return str(prof["name"])
    return "модель"


def fmt_seconds(x: float) -> str:
    return f"{x:.1f}" if x < 10 else f"{x:.0f}"


def speed_figures(completion_tokens, elapsed: float, ttft: Optional[float]) -> tuple:
    """(generation tokens/s, overall tokens/s, generation seconds).

    Generation speed = (tokens - 1) / (total - time to the first token): the first token marks the end of
    prompt processing, so the wait for it must not count (checked against LM Studio's own log: within 1%).
    Falls back to the overall figure when the generation phase is too short to measure."""
    overall = (completion_tokens / elapsed) if completion_tokens and elapsed > 0 else None
    if not completion_tokens or elapsed <= 0:
        return None, overall, None
    gen_time = elapsed - ttft if ttft is not None else elapsed
    if completion_tokens >= 3 and gen_time >= 0.1 and gen_time >= 0.02 * elapsed:
        return (completion_tokens - 1) / gen_time, overall, gen_time
    return overall, overall, gen_time if gen_time > 0 else None


_EMBEDDING_HINTS = ("embed", "e5-", "bge", "gte", "nomic-embed", "text-embedding")


def _looks_like_embedding_model(model_id: str) -> bool:
    low = model_id.lower()
    return any(h in low for h in _EMBEDDING_HINTS)


class LMStudioClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, timeout: Optional[float] = None):
        url = f"{self.base_url}{path}"
        try:
            resp = httpx.get(url, timeout=timeout or self.timeout)
            resp.raise_for_status()
            return resp.json()
        except httpx.ConnectError as exc:
            raise ConnectionRefused(self.base_url) from exc
        except httpx.TimeoutException as exc:
            raise LMTimeout(timeout or self.timeout) from exc
        except httpx.HTTPStatusError as exc:
            raise LMStudioError(f"LM Studio вернул ошибку: {exc}") from exc

    def check_connection(self) -> dict:
        try:
            self._get("/v1/models", timeout=5)
            return {"ok": True, "message": "LM Studio доступен"}
        except LMStudioError as exc:
            return {"ok": False, "message": str(exc)}

    def list_models(self, include_embeddings: bool = False) -> list[dict]:
        """List models, preferring /api/v0/models for state/context length, falling back to /v1/models."""
        try:
            data = self._get("/api/v0/models")
            models = []
            for m in data.get("data", []):
                info = ModelInfo(
                    id=m.get("id", ""),
                    state=m.get("state", "unknown"),
                    max_context_length=m.get("max_context_length"),
                    loaded_context_length=m.get("loaded_context_length"),
                    is_embedding=(m.get("type") == "embeddings") or _looks_like_embedding_model(m.get("id", "")),
                )
                models.append(info)
        except ConnectionRefused:
            raise  # server is down: a second attempt via /v1/models would only double the wait
        except LMStudioError:
            data = self._get("/v1/models")
            models = [
                ModelInfo(id=m.get("id", ""), is_embedding=_looks_like_embedding_model(m.get("id", "")))
                for m in data.get("data", [])
            ]
        if not include_embeddings:
            models = [m for m in models if not m.is_embedding]
        return [m.to_dict() for m in models]

    def resolve_model(self, configured: Optional[str]) -> dict:
        """Decide which model to actually use for a request.

        Returns {"model", "context", "note"}. Order: the configured model if it is
        currently loaded; otherwise the single/first loaded LLM (with a visible RU
        note explaining the substitution); otherwise raises NoModelLoaded (with a
        hint about the configured model, if any). Never loads a model itself.
        """
        models = self.list_models()
        loaded = [m for m in models if m.get("state") == "loaded"]
        by_id = {m["id"]: m for m in models}
        if configured and configured in by_id and by_id[configured].get("state") == "loaded":
            m = by_id[configured]
            return {"model": configured, "context": m.get("effective_context"), "note": None}
        # state unknown (older LM Studio without /api/v0): trust the configured name
        if configured and configured in by_id and by_id[configured].get("state") == "unknown":
            return {"model": configured, "context": by_id[configured].get("effective_context"), "note": None}
        if loaded:
            m = loaded[0]
            note = (f"Настроенная модель «{configured}» не загружена в LM Studio — "
                    f"используется загруженная «{m['id']}»." if configured else
                    f"Модель не выбрана в настройках — используется загруженная «{m['id']}».")
            return {"model": m["id"], "context": m.get("effective_context"), "note": note}
        hint = (f" Настроена модель «{configured}», но она не загружена." if configured else "")
        raise NoModelLoaded(hint)

    def chat(self, messages: list[dict], model: str, temperature: float = 0.2,
             max_tokens: int = 1024, timeout: Optional[float] = None,
             stream: bool = False, reasoning_effort: Optional[str] = None):
        """Call /v1/chat/completions. If stream=True, returns a generator of text deltas;
        otherwise returns the full response text (string).

        reasoning_effort: one of "none"/"low"/"medium"/"high" (LM Studio / OpenAI-style
        reasoning-effort passthrough). "none" reliably disables reasoning for models that
        support it (verified empirically: reasoning_tokens drops to 0, much faster).
        Models that don't understand the field simply ignore it. Pass None to omit it
        entirely and use the model's default behaviour.
        """
        if not model:
            raise NoModelLoaded()
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        eff_timeout = timeout or self.timeout
        try:
            if not stream:
                resp = httpx.post(url, json=payload, timeout=eff_timeout)
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise NoModelLoaded()
                message = choices[0]["message"]
                text = message.get("content") or ""
                # Some models (e.g. this Gemma build) return reasoning in a separate
                # "reasoning_content" field rather than inline <think> tags; strip both.
                visible, _ = strip_reasoning(text, keep=True)
                return visible
            else:
                return self._stream_chat(url, payload, eff_timeout)
        except httpx.ConnectError as exc:
            raise ConnectionRefused(self.base_url) from exc
        except httpx.TimeoutException as exc:
            raise LMTimeout(eff_timeout) from exc
        except httpx.HTTPStatusError as exc:
            raise LMStudioError(f"LM Studio вернул ошибку: {exc}") from exc

    def chat_with_stats(self, messages: list[dict], model: str, temperature: float = 0.2,
                        max_tokens: int = 1024, timeout: Optional[float] = None,
                        reasoning_effort: Optional[str] = None,
                        on_delta: Optional[callable] = None) -> tuple[str, "GenerationStats"]:
        """Like chat(), but always streams internally so we can measure time-to-first-token
        and tokens/sec, and returns (visible_text, GenerationStats).

        on_delta(text_delta: str) is called for each visible content chunk as it arrives,
        so callers (job progress) can report live tok/s / a running preview.
        """
        if not model:
            raise NoModelLoaded()
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        eff_timeout = timeout or self.timeout
        t0 = time.monotonic()
        first_token_at: Optional[float] = None
        visible_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict = {}
        stop_reason: Optional[str] = None
        try:
            with httpx.stream("POST", url, json=payload, timeout=eff_timeout) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta", {})
                    if choice.get("finish_reason"):
                        stop_reason = choice["finish_reason"]
                    content = delta.get("content")
                    reasoning = delta.get("reasoning_content")
                    if (reasoning or content) and first_token_at is None:
                        first_token_at = time.monotonic()   # first token of ANY kind (reasoning included)
                    if reasoning:
                        reasoning_parts.append(reasoning)
                    if content:
                        visible_parts.append(content)
                        if on_delta:
                            on_delta(content)
        except httpx.ConnectError as exc:
            raise ConnectionRefused(self.base_url) from exc
        except httpx.TimeoutException as exc:
            raise LMTimeout(eff_timeout) from exc
        except httpx.HTTPStatusError as exc:
            raise LMStudioError(f"LM Studio вернул ошибку: {exc}") from exc

        elapsed = time.monotonic() - t0
        completion_tokens = usage.get("completion_tokens")
        reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        ttft = (first_token_at - t0) if first_token_at is not None else None
        gen_tps, overall_tps, gen_time = speed_figures(completion_tokens, elapsed, ttft)
        stats = GenerationStats(
            model=model,
            elapsed_s=round(elapsed, 2),
            ttft_s=round(ttft, 2) if ttft is not None else None,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=usage.get("total_tokens"),
            tokens_per_sec=round(gen_tps, 1) if gen_tps is not None else None,
            stop_reason=stop_reason,
            overall_tokens_per_sec=round(overall_tps, 1) if overall_tps is not None else None,
            gen_time_s=round(gen_time, 2) if gen_time is not None else None,
        )
        text = "".join(visible_parts)
        visible, _ = strip_reasoning(text, keep=True)
        return visible, stats

    def _stream_chat(self, url: str, payload: dict, timeout: float) -> Iterator[str]:
        with httpx.stream("POST", url, json=payload, timeout=timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content


def strip_reasoning(text: str, keep: bool = False) -> str:
    """Strip <think>...</think> reasoning blocks from LLM output for display.

    If keep=True, returns (visible_text, reasoning_text) tuple instead of just stripping.
    """
    import re
    matches = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    visible = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if keep:
        return visible, "\n".join(matches)
    return visible
