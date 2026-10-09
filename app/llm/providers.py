"""LLM providers: where the model comes from (local program or cloud service).

One generic OpenAI-compatible client (`ProviderClient`) serves every kind; the differences
(model listing, context discovery, reasoning switch, headers, cost) are isolated per kind.
Profiles live in settings.json WITHOUT keys; keys are in app/secrets_store.py.
"""
from __future__ import annotations

import json
import re
import time
from typing import Callable, Optional
from urllib.parse import urlparse

import httpx

from app import secrets_store
from app.llm import provider_errors
from app.llm.lmstudio import (GenerationStats, LMStudioError, _looks_like_embedding_model, speed_figures,
                               strip_reasoning)

try:
    from app.log import get_logger
except ImportError:  # pragma: no cover
    from tests.fakes.log import get_logger

logger = get_logger("app")

LOCAL_ID = "lmstudio-local"
TASKS = ("rag", "translation", "rewrite")
MAX_RETRIES = 3
MODEL_CACHE_TTL = 600.0

_TEST_TRANSPORT = None          # tests: an httpx transport (MockTransport) used instead of the network
_sleep: Callable[[float], None] = time.sleep
_now: Callable[[], float] = time.monotonic
_MODEL_CACHE: dict[str, tuple[float, list[dict]]] = {}


class ProviderError(LMStudioError):
    """Any provider problem; the message is plain Russian and never contains a key.

    retryable: the cause is on the provider's side (busy / 5xx) - a backup model may be tried.
    partial / partial_stats: the answer received before a stream broke off (None = nothing received)."""

    def __init__(self, message: str = "", retryable: bool = False, partial: Optional[str] = None,
                 partial_stats: Optional[GenerationStats] = None, kind: str = ""):
        super().__init__(message)
        self.retryable = retryable
        self.partial = partial
        self.partial_stats = partial_stats
        self.kind = kind        # "provider_busy" | "account_quota" | "account_minute" | "server" | "" ...


KINDS = {
    "lmstudio": {"title": "LM Studio", "icon": "🖥", "default_url": "http://localhost:1234", "needs_key": False,
                 "desc": "Программа на этом компьютере (по умолчанию). Тексты никуда не уходят."},
    "ollama": {"title": "Ollama", "icon": "🖥", "default_url": "http://localhost:11434", "needs_key": False,
               "desc": "Ещё одна программа для локальных моделей. Работает на этом компьютере."},
    "llamacpp": {"title": "llama.cpp (llama-server)", "icon": "🖥", "default_url": "http://localhost:8080",
                 "needs_key": False, "desc": "Сервер llama.cpp на этом компьютере."},
    "openai_compatible": {"title": "Другой совместимый сервер (OpenAI API)", "icon": "🔌", "default_url": "",
                          "needs_key": "optional",
                          "desc": "Любой сервер с OpenAI-совместимым адресом: vLLM, LocalAI, text-generation-webui, "
                                  "сам OpenAI и др. Если адрес не локальный — это облако."},
    "openrouter": {"title": "OpenRouter", "icon": "☁", "default_url": "https://openrouter.ai/api/v1",
                   "needs_key": True,
                   "desc": "Облачный сервис с сотнями моделей (есть бесплатные). Нужен ключ. "
                           "Тексты абзацев и вопросы отправляются на внешний сервер."},
    "huggingface": {"title": "Hugging Face", "icon": "☁", "default_url": "https://router.huggingface.co/v1",
                    "needs_key": True,
                    "desc": "Облачный сервис с открытыми моделями. Адрес уже указан — нужен только токен (на сайте "
                            "huggingface.co: Settings → Access Tokens, с правом «Make calls to Inference Providers»). "
                            "Бесплатно даётся лишь небольшой ежемесячный кредит (около $0,10): хватит на редкие запросы "
                            "к дешёвым моделям. Тексты абзацев и вопросы отправляются на внешний сервер."},
    "sharellm": {"title": "ShareLLM", "icon": "☁", "default_url": "https://sharellm.net/v1", "needs_key": True,
                 "desc": "Облачный шлюз sharellm.net с общим доступом к моделям (OpenAI-совместимый). Нужен токен "
                         "с сайта. Это не официальный провайдер моделей, а посредник: условия и лимиты задаёт сам "
                         "сервис, он может изменить их или отключиться. Тексты абзацев и вопросы проходят через "
                         "его сервер — не отправляйте закрытые материалы."},
}

HF_WHOAMI = "https://huggingface.co/api/whoami-v2"
HF_CHEAP_IN, HF_CHEAP_OUT = 0.10, 0.30   # $ per 1M tokens: «cheap» models that the monthly free credit lasts on

_LOCAL_HOST_RE = re.compile(r"^(localhost|127\.\d+\.\d+\.\d+|\[?::1\]?|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|"
                            r"172\.(1[6-9]|2\d|3[01])\.\d+\.\d+|[^.]+|.*\.(local|lan|home|internal))$", re.I)


def host_of(url: str) -> str:
    return (urlparse(url or "").hostname or "").lower()


def is_cloud(kind: str, base_url: str) -> bool:
    if kind in ("openrouter", "huggingface"):
        return True
    host = host_of(base_url)
    if not host:
        return False
    return not _LOCAL_HOST_RE.match(host)  # single-label names (docker, nas) count as local


# ------------------------------------------------------------- profiles ---

PROFILE_FIELDS = ("id", "name", "kind", "base_url", "default_model", "timeout", "extra_headers",
                  "context_length", "send_reasoning_effort", "params", "fallback_models")


def migrated_profile(settings: dict) -> dict:
    return {"id": LOCAL_ID, "name": "LM Studio (локально)", "kind": "lmstudio",
            "base_url": settings.get("lmstudio_url") or KINDS["lmstudio"]["default_url"],
            "default_model": settings.get("llm_model") or "", "timeout": None, "extra_headers": {},
            "context_length": None, "send_reasoning_effort": False, "params": {}, "fallback_models": []}


def _clean_models(v) -> list[str]:
    out: list[str] = []
    for m in (v if isinstance(v, list) else []):
        m = str(m).strip()
        if m and m not in out:
            out.append(m)
    return out[:5]


def normalize_profile(p: dict) -> dict:
    kind = p.get("kind") if p.get("kind") in KINDS else "lmstudio"
    out = {
        "id": str(p.get("id") or ""), "name": str(p.get("name") or KINDS[kind]["title"]), "kind": kind,
        "base_url": str(p.get("base_url") or KINDS[kind]["default_url"]).strip(),
        "default_model": str(p.get("default_model") or ""),
        "timeout": p.get("timeout") or None,
        "extra_headers": dict(p.get("extra_headers") or {}),
        "context_length": p.get("context_length") or None,
        "send_reasoning_effort": bool(p.get("send_reasoning_effort")),
        "params": {k: v for k, v in (p.get("params") or {}).items() if k in ("temperature", "max_tokens")},
        "fallback_models": _clean_models(p.get("fallback_models")),
    }
    return out


def get_profiles(settings: dict) -> list[dict]:
    """All profiles; an old installation (no `llm_providers`) gets «LM Studio (локально)» from lmstudio_url/llm_model."""
    raw = settings.get("llm_providers")
    if isinstance(raw, list) and raw:
        return [normalize_profile(p) for p in raw if isinstance(p, dict) and p.get("id")]
    return [migrated_profile(settings)]


def active_id(settings: dict) -> str:
    profs = get_profiles(settings)
    a = settings.get("active_llm_provider")
    return a if any(p["id"] == a for p in profs) else profs[0]["id"]


def find_profile(settings: dict, pid: Optional[str]) -> Optional[dict]:
    pid = pid or active_id(settings)
    return next((p for p in get_profiles(settings) if p["id"] == pid), None)


def profile_for_task(settings: dict, task: Optional[str]) -> dict:
    """The provider used for a task (rag / translation / rewrite); "" or unknown task = the active one."""
    pid = (settings.get(f"llm_provider_{task}") or "") if task in TASKS else ""
    if pid:
        p = find_profile(settings, pid)
        if p is None:
            raise ProviderError("Провайдер, выбранный для этой задачи, удалён. Выберите другой в «Параметрах ответа».")
        return p
    return find_profile(settings, active_id(settings))


def model_for_task(settings: dict, prof: dict, task: Optional[str]) -> str:
    """Model preference for the task ("" = decide at resolve time)."""
    m = (settings.get(f"llm_model_{task}") or "") if task in TASKS else ""
    if m:
        return m
    if prof["id"] == active_id(settings):
        return settings.get("llm_model") or prof.get("default_model") or ""
    return prof.get("default_model") or ""


def check_allowed(settings: dict, prof: dict) -> None:
    if is_cloud(prof["kind"], prof["base_url"]) and not settings.get("allow_cloud_llm"):
        raise ProviderError(
            f"Облачный провайдер «{prof['name']}» выключен: в программе запрещена отправка текстов на внешние "
            f"серверы. Разрешить можно в Настройки → Модель LLM → «Облачные модели» (после подтверждения).")


def validate_profile(d: dict, partial: bool = False) -> dict:
    """Clean user input for a profile; raises ValueError with a plain-Russian message."""
    out = {}
    if "name" in d or not partial:
        n = str(d.get("name") or "").strip()
        if not n or len(n) > 60:
            raise ValueError("Название: от 1 до 60 символов")
        out["name"] = n
    if "kind" in d or not partial:
        if d.get("kind") not in KINDS:
            raise ValueError("Тип провайдера: выберите из списка")
        out["kind"] = d["kind"]
    if "base_url" in d or not partial:
        u = str(d.get("base_url") or "").strip().rstrip("/")
        pu = urlparse(u)
        if pu.scheme not in ("http", "https") or not pu.hostname:
            raise ValueError("Адрес: нужен полный адрес, например http://localhost:1234 или https://openrouter.ai/api/v1")
        if pu.username or pu.password or pu.query:
            raise ValueError("Адрес: не указывайте в нём логин, пароль или ключ — для ключа есть отдельное поле")
        out["base_url"] = u
    if "default_model" in d:
        out["default_model"] = str(d.get("default_model") or "").strip()
    if "timeout" in d:
        t = d["timeout"]
        if t in (None, "", 0):
            out["timeout"] = None
        elif isinstance(t, bool) or not isinstance(t, (int, float)) or not 1 <= t <= 36000:
            raise ValueError("Таймаут: число секунд от 1 до 36000 (пусто — как в общих настройках)")
        else:
            out["timeout"] = t
    if "context_length" in d:
        c = d["context_length"]
        if c in (None, "", 0):
            out["context_length"] = None
        elif isinstance(c, bool) or not isinstance(c, int) or not 512 <= c <= 10_000_000:
            raise ValueError("Размер контекста: целое число токенов от 512 (пусто — определять автоматически)")
        else:
            out["context_length"] = c
    if "extra_headers" in d:
        h = d["extra_headers"] or {}
        if not isinstance(h, dict):
            raise ValueError("Дополнительные заголовки: ожидается список «имя: значение»")
        clean = {}
        for k, v in h.items():
            if not isinstance(k, str) or not re.fullmatch(r"[A-Za-z0-9-]{1,60}", k) or not isinstance(v, str) \
                    or len(v) > 300 or "\n" in v or "\r" in v:
                raise ValueError(f"Заголовок «{k}»: недопустимое имя или значение")
            if k.lower() in ("authorization", "content-type", "host", "content-length"):
                raise ValueError(f"Заголовок «{k}» задаётся программой (ключ вводится в отдельное поле)")
            clean[k] = v
        out["extra_headers"] = clean
    if "fallback_models" in d:
        fm = d["fallback_models"] or []
        if isinstance(fm, str):
            fm = [x for x in re.split(r"[\n,;]+", fm) if x.strip()]
        if not isinstance(fm, list) or any(not isinstance(x, str) or len(x) > 200 for x in fm):
            raise ValueError("Запасные модели: ожидается список названий моделей (по одному в строке)")
        clean_fm = _clean_models(fm)
        if len([x for x in fm if x.strip()]) > 5:
            raise ValueError("Запасные модели: не больше 5")
        out["fallback_models"] = clean_fm
    if "send_reasoning_effort" in d:
        if not isinstance(d["send_reasoning_effort"], bool):
            raise ValueError("«Передавать режим рассуждения»: ожидается «вкл» или «выкл»")
        out["send_reasoning_effort"] = d["send_reasoning_effort"]
    if "params" in d:
        p = d["params"] or {}
        if not isinstance(p, dict):
            raise ValueError("Свои параметры модели: ожидается объект")
        clean = {}
        for k, lo, hi, lab in (("temperature", 0, 2, "Температура"), ("max_tokens", 1, 200000, "Максимум токенов")):
            v = p.get(k)
            if v in (None, ""):
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
                raise ValueError(f"{lab} (для провайдера): число от {lo} до {hi}")
            clean[k] = v
        out["params"] = clean
    return out


def make_id(name: str, existing: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower().translate(str.maketrans(
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя", "abvgdeezzijklmnoprstufhccss_y_eua"))).strip("-") or "provider"
    cand, i = base, 2
    while cand in existing:
        cand, i = f"{base}-{i}", i + 1
    return cand


def public_info(settings: dict, prof: dict, model: Optional[str] = None) -> dict:
    """Key-free description of a provider for snapshots / UI."""
    return {"id": prof["id"], "name": prof["name"], "kind": prof["kind"], "host": host_of(prof["base_url"]),
            "is_cloud": is_cloud(prof["kind"], prof["base_url"]), "model": model}


def client_for(settings: dict, prof: dict, timeout: Optional[float] = None) -> "ProviderClient":
    """Build a client: gating by allow_cloud_llm, key from the secret store / env."""
    check_allowed(settings, prof)
    key = secrets_store.get_key(prof["id"], prof["kind"], prof["base_url"])
    app_headers = {}
    if prof["kind"] == "openrouter":
        if settings.get("openrouter_referer"):
            app_headers["HTTP-Referer"] = str(settings["openrouter_referer"])
        if settings.get("openrouter_title"):
            app_headers["X-Title"] = str(settings["openrouter_title"])
    return ProviderClient(prof, api_key=key, timeout=timeout or prof.get("timeout") or settings.get("llm_timeout"),
                          app_headers=app_headers)


# ------------------------------------------------------- reasoning/cost ---

def reasoning_params(kind: str, effort: Optional[str], send_generic: bool = False,
                     supports: Optional[bool] = None) -> dict:
    """Per-kind mapping of our reasoning effort (none/low/medium/high) to request fields."""
    if not effort:
        return {}
    if kind == "lmstudio":
        return {"reasoning_effort": effort}
    if kind == "openrouter":
        if supports is False:
            return {}
        if effort == "none":
            return {"reasoning": {"enabled": False}}
        return {"reasoning": {"effort": effort, "exclude": True}}
    return {"reasoning_effort": effort} if send_generic else {}


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _model_entry(kind: str, m: dict) -> dict:
    mid = m.get("id") or m.get("name") or ""
    out = {"id": mid, "name": m.get("name") or mid, "context_length": None, "loaded": None, "price_in": None,
           "price_out": None, "free": False, "supports_reasoning": None, "kind": kind}
    ctx = m.get("context_length") or m.get("max_model_len") or m.get("max_context_length")
    out["context_length"] = int(ctx) if isinstance(ctx, (int, float)) and ctx else None
    return out


def filter_models(models: list[dict], q: str = "", free_only: bool = False, min_context: int = 0) -> list[dict]:
    q = (q or "").strip().lower()
    out = []
    for m in models:
        if q and q not in (m.get("id") or "").lower() and q not in (m.get("name") or "").lower():
            continue
        if free_only and not (m.get("free") or m.get("cheap")):
            continue
        if min_context and (m.get("context_length") or 0) < min_context:
            continue
        out.append(m)
    return out


def _price_of(models: list[dict], model: str) -> Optional[dict]:
    return next((m for m in models if m["id"] == model), None)


# --------------------------------------------------------------- client ---

class ProviderClient:
    """OpenAI-compatible chat client for every provider kind (same interface as LMStudioClient)."""

    def __init__(self, profile: dict, api_key: Optional[str] = None, timeout: Optional[float] = None,
                 app_headers: Optional[dict] = None):
        self.profile = profile
        self.kind = profile["kind"]
        self.api_key = api_key or None
        self.base_url = profile["base_url"].rstrip("/")
        self.timeout = float(timeout or profile.get("timeout") or 120)
        self.app_headers = dict(app_headers or {})
        self.cloud = is_cloud(self.kind, self.base_url)
        self.host = host_of(self.base_url)
        pu = urlparse(self.base_url)
        path = pu.path.rstrip("/")
        if self.kind in ("openrouter", "huggingface") or path not in ("", "/v1"):
            self.v1 = self.base_url                      # explicit full API root (…/api/v1)
            self.native = self.base_url[: -len("/v1")] if path.endswith("/v1") else self.base_url
        else:
            self.native = f"{pu.scheme}://{pu.netloc}"  # host root; OpenAI API lives under /v1
            self.v1 = self.native + "/v1"

    # -- transport ------------------------------------------------------
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        h.update(self.app_headers)
        h.update(self.profile.get("extra_headers") or {})
        return h

    def _redact(self, text: str) -> str:
        return secrets_store.redact(text, self.api_key)

    def _conn_error(self) -> ProviderError:
        if self.cloud:
            return ProviderError(f"Нет соединения с «{self.host}» — нет интернета или сервис недоступен. "
                                 f"Облачный провайдер недоступен; проверьте сеть или переключитесь на локальную модель.")
        name = KINDS[self.kind]["title"]
        return ProviderError(f"Не удалось подключиться к {name} по адресу {self.base_url}. "
                             f"Убедитесь, что программа запущена и сервер включён.")

    def _title(self) -> str:
        return self.profile.get("name") or KINDS[self.kind]["title"]

    def _explain(self, e: dict, model: Optional[str], status: Optional[int], stream: bool = False) -> ProviderError:
        for k in ("message", "raw", "text"):
            e[k] = self._redact(e.get(k) or "")
        msg, kind = provider_errors.explain(e, self._title(), model, status, stream=stream,
                                            openrouter=self.kind == "openrouter")
        return ProviderError(self._redact(msg), retryable=provider_errors.is_retryable(kind), kind=kind)

    def _http_error(self, resp: httpx.Response, model: Optional[str]) -> ProviderError:
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001
            body = None
        e = provider_errors.parse_error(body) if body is not None else provider_errors.parse_error({})
        return self._explain(e, model, resp.status_code)

    def _retries_allowed(self, resp: httpx.Response, model: Optional[str]) -> int:
        """How many times a 429 may be retried. OpenRouter free models share one pool: every retry burns
        free-tier quota and rarely helps, so ONE short retry unless the server says when to come back."""
        if resp.headers.get("retry-after") is not None:
            return MAX_RETRIES
        try:    # an account-wide cap (daily / per-minute free quota) does not go away in a second: never retry it
            resp.read()
            body = resp.json()
            if provider_errors.classify(provider_errors.parse_error(body), 429).startswith("account"):
                return 0
        except Exception:  # noqa: BLE001
            pass
        if self.kind == "openrouter" and model and (str(model).endswith(":free") or str(model) == "openrouter/free"):
            return 1
        return MAX_RETRIES

    @staticmethod
    def _retry_after(resp: httpx.Response, attempt: int) -> float:
        ra = resp.headers.get("retry-after")
        try:
            if ra is not None:
                return max(0.0, min(float(ra), 60.0))
        except ValueError:
            pass
        return float(2 ** attempt)

    def _send(self, method: str, url: str, body: Optional[dict] = None, timeout: Optional[float] = None,
              stream: bool = False, model: Optional[str] = None):
        eff = timeout or self.timeout
        attempt = 0
        while True:
            client = httpx.Client(transport=_TEST_TRANSPORT, timeout=eff)
            try:
                req = client.build_request(method, url, json=body, headers=self._headers())
                resp = client.send(req, stream=stream)
            except httpx.TimeoutException as exc:
                client.close()
                raise ProviderError(f"Провайдер «{self.host}» не ответил за {eff:g} сек. Увеличьте таймаут "
                                    f"в настройках или проверьте, не перегружена ли модель.") from exc
            except httpx.ConnectError as exc:
                client.close()
                raise self._conn_error() from exc
            except httpx.HTTPError as exc:
                client.close()
                raise ProviderError(self._redact(f"Ошибка сети при обращении к «{self.host}»: {type(exc).__name__}")) from exc
            if resp.status_code == 429 and attempt < self._retries_allowed(resp, model):
                wait = self._retry_after(resp, attempt)
                resp.close()
                client.close()
                logger.info("LLM host=%s status=429 retry=%d wait=%.1fs", self.host, attempt + 1, wait)
                _sleep(wait)
                attempt += 1
                continue
            if resp.status_code >= 400:
                try:
                    resp.read()
                except httpx.HTTPError:
                    pass
                err = self._http_error(resp, model)
                logger.info("LLM host=%s model=%s status=%s", self.host, model, resp.status_code)
                resp.close()
                client.close()
                raise err
            return client, resp

    def _get_json(self, url: str, timeout: Optional[float] = None):
        client, resp = self._send("GET", url, timeout=timeout)
        try:
            return resp.json()
        except ValueError as exc:
            raise ProviderError(f"Провайдер «{self.host}» вернул не JSON — проверьте адрес.") from exc
        finally:
            resp.close()
            client.close()

    # -- models ---------------------------------------------------------
    def check_connection(self) -> dict:
        try:
            t0 = time.monotonic()
            if self.kind == "openrouter":
                self._get_json(f"{self.v1}/key", timeout=10)  # models list is public; /key verifies the key
            elif self.kind == "huggingface":
                if not self.api_key:
                    raise ProviderError("Не задан токен Hugging Face. Введите его в настройках провайдера.")
                self._get_json(HF_WHOAMI, timeout=10)           # the models list is public: check the token itself
            elif self.kind == "ollama":
                self._get_json(f"{self.native}/api/tags", timeout=5)
            else:
                self._get_json(f"{self.v1}/models", timeout=5 if not self.cloud else 10)
            ms = int((time.monotonic() - t0) * 1000)
            name = "LM Studio" if self.kind == "lmstudio" else self.profile["name"]
            return {"ok": True, "message": f"{name} доступен", "latency_ms": ms}
        except LMStudioError as exc:
            return {"ok": False, "message": str(exc)}

    def account_state(self) -> dict:
        """OpenRouter: the state of the account behind the key (GET /key), in plain words. Raises ProviderError."""
        if self.kind != "openrouter":
            return {"available": False, "reason": "Состояние счёта показывается только для OpenRouter."}
        data = self._get_json(f"{self.v1}/key", timeout=10)
        if not isinstance(data, dict):
            raise ProviderError("OpenRouter вернул непонятный ответ о состоянии счёта.")
        return {"available": True, **provider_errors.describe_account(data)}

    def list_models(self, include_embeddings: bool = False, refresh: bool = False) -> list[dict]:
        if self.kind in ("openrouter", "huggingface"):
            ck = self.base_url
            hit = _MODEL_CACHE.get(ck)
            if hit and not refresh and _now() - hit[0] < MODEL_CACHE_TTL:
                return hit[1]
            models = self._list_openrouter() if self.kind == "openrouter" else self._list_huggingface()
            _MODEL_CACHE[ck] = (_now(), models)
            return models
        if self.kind == "lmstudio":
            models = self._list_lmstudio()
        elif self.kind == "ollama":
            data = self._get_json(f"{self.native}/api/tags")
            models = [_model_entry("ollama", {"id": m.get("name") or m.get("model"), "name": m.get("name")})
                      for m in data.get("models", [])]
        else:
            data = self._get_json(f"{self.v1}/models")
            models = [_model_entry(self.kind, m) for m in data.get("data", []) if isinstance(m, dict)]
        for m in models:
            m.setdefault("is_embedding", _looks_like_embedding_model(m["id"]))
        if not include_embeddings:
            models = [m for m in models if not m["is_embedding"]]
        return models

    def _list_lmstudio(self) -> list[dict]:
        try:
            data = self._get_json(f"{self.native}/api/v0/models")
            rows = []
            for m in data.get("data", []):
                mid = m.get("id", "")
                e = _model_entry("lmstudio", m)
                state = m.get("state", "unknown")
                loaded_ctx = m.get("loaded_context_length")
                e.update({
                    "state": state, "loaded": (state == "loaded") if state in ("loaded", "not-loaded") else None,
                    "max_context_length": m.get("max_context_length"), "loaded_context_length": loaded_ctx,
                    "is_embedding": (m.get("type") == "embeddings") or _looks_like_embedding_model(mid),
                })
                e["effective_context"] = loaded_ctx if state == "loaded" and loaded_ctx else m.get("max_context_length")
                e["context_length"] = e["effective_context"]
                rows.append(e)
            return rows
        except ProviderError as exc:
            if "подключиться" in str(exc) or "не ответил" in str(exc):
                raise  # server down: a second attempt would only double the wait
        data = self._get_json(f"{self.v1}/models")
        rows = []
        for m in data.get("data", []):
            e = _model_entry("lmstudio", {"id": m.get("id", "")})
            e.update({"state": "unknown", "max_context_length": None, "loaded_context_length": None,
                      "effective_context": None, "is_embedding": _looks_like_embedding_model(m.get("id", ""))})
            rows.append(e)
        return rows

    def _list_openrouter(self) -> list[dict]:
        data = self._get_json(f"{self.v1}/models", timeout=max(self.timeout, 30))
        rows = []
        for m in data.get("data", []):
            if not isinstance(m, dict) or not m.get("id"):
                continue
            e = _model_entry("openrouter", m)
            pr = m.get("pricing") or {}
            pin, pout = _num(pr.get("prompt")), _num(pr.get("completion"))
            e["price_in"] = round(pin * 1_000_000, 4) if pin is not None and pin >= 0 else None
            e["price_out"] = round(pout * 1_000_000, 4) if pout is not None and pout >= 0 else None
            e["free"] = (e["price_in"] == 0 and e["price_out"] == 0) or str(m["id"]).endswith(":free")
            sp = m.get("supported_parameters")
            e["supports_reasoning"] = (("reasoning" in sp) or ("include_reasoning" in sp)) if isinstance(sp, list) else None
            e["is_embedding"] = _looks_like_embedding_model(e["id"])
            rows.append(e)
        rows.sort(key=lambda r: r["id"])
        return [r for r in rows if not r["is_embedding"]]

    def _list_huggingface(self) -> list[dict]:
        """Public catalogue of the HF router: each model has several providers with their own price / context.
        Price shown = the cheapest live provider; `free` = at least one provider charges $0; `cheap` = within
        HF_CHEAP_* (the monthly free credit goes a long way). Embedding / non-text models are dropped."""
        data = self._get_json(f"{self.v1}/models", timeout=max(self.timeout, 30))
        rows = []
        for m in data.get("data", []):
            if not isinstance(m, dict) or not m.get("id"):
                continue
            if "text" not in ((m.get("architecture") or {}).get("output_modalities") or ["text"]):
                continue
            live = [pv for pv in m.get("providers") or [] if isinstance(pv, dict) and pv.get("status", "live") == "live"]
            priced = [(_num((pv.get("pricing") or {}).get("input")), _num((pv.get("pricing") or {}).get("output")))
                      for pv in live if isinstance(pv.get("pricing"), dict)]
            priced = [(a, b) for a, b in priced if a is not None and b is not None]
            e = _model_entry("huggingface", m)
            ctxs = [pv.get("context_length") for pv in live if isinstance(pv.get("context_length"), (int, float))]
            e["context_length"] = int(max(ctxs)) if ctxs else None
            if priced:
                a, b = min(priced, key=lambda x: x[0] + x[1])
                e["price_in"], e["price_out"] = round(a, 4), round(b, 4)
                e["free"] = a == 0 and b == 0
                e["cheap"] = a <= HF_CHEAP_IN and b <= HF_CHEAP_OUT
            e["supports_tools"] = any(pv.get("supports_tools") for pv in live)
            e["is_embedding"] = _looks_like_embedding_model(e["id"])
            if live and not e["is_embedding"]:
                rows.append(e)
        rows.sort(key=lambda r: ((r["price_in"] if r["price_in"] is not None else 1e9), r["id"]))
        return rows

    def _cached_entry(self, model: str) -> Optional[dict]:
        hit = _MODEL_CACHE.get(self.base_url) if self.kind in ("openrouter", "huggingface") else None
        return _price_of(hit[1], model) if hit else None

    def discover_context(self, model: str) -> Optional[int]:
        """Context window of the model if the provider tells (best effort, never raises)."""
        ov = self.profile.get("context_length")
        if ov:
            return int(ov)
        try:
            if self.kind in ("openrouter", "huggingface"):
                e = self._cached_entry(model)
                if e is None:
                    e = _price_of(self.list_models(), model)
                return e["context_length"] if e else None
            if self.kind == "ollama":
                j = self._send_json_post(f"{self.native}/api/show", {"model": model})
                info = j.get("model_info") or {}
                ctx = next((v for k, v in info.items() if k.endswith(".context_length")), None)
                # Ollama serves 4096 by default unless num_ctx is raised; be conservative
                return min(int(ctx), 8192) if ctx else None
            if self.kind == "llamacpp":
                j = self._get_json(f"{self.native}/props", timeout=5)
                n = (j.get("default_generation_settings") or {}).get("n_ctx")
                return int(n) if n else None
            e = _price_of(self.list_models(), model)
            return e["context_length"] if e else None
        except (ProviderError, ValueError, TypeError, AttributeError):
            return None

    def _send_json_post(self, url: str, body: dict, timeout: Optional[float] = None):
        client, resp = self._send("POST", url, body=body, timeout=timeout or 10)
        try:
            return resp.json()
        finally:
            resp.close()
            client.close()

    def resolve_model(self, configured: Optional[str]) -> dict:
        """{"model", "context", "note"}. LM Studio: only a LOADED model is usable (never loads one itself);
        other kinds: the configured model (or the provider's default)."""
        if self.kind == "lmstudio":
            models = self.list_models()
            loaded = [m for m in models if m.get("state") == "loaded"]
            by_id = {m["id"]: m for m in models}
            if configured and configured in by_id and by_id[configured].get("state") in ("loaded", "unknown"):
                return {"model": configured, "context": by_id[configured].get("effective_context"), "note": None}
            if loaded:
                m = loaded[0]
                note = (f"Настроенная модель «{configured}» не загружена в LM Studio — используется загруженная «{m['id']}»."
                        if configured else f"Модель не выбрана в настройках — используется загруженная «{m['id']}».")
                return {"model": m["id"], "context": m.get("effective_context"), "note": note}
            hint = f" Настроена модель «{configured}», но она не загружена." if configured else ""
            raise ProviderError("В LM Studio не загружена ни одна модель. Загрузите модель в LM Studio "
                                "и выберите её в настройках." + hint)
        model = configured or self.profile.get("default_model") or ""
        if not model:
            raise ProviderError(f"Для провайдера «{self.profile['name']}» не выбрана модель: "
                                f"выберите её в Настройки → Модель LLM.")
        return {"model": model, "context": self.discover_context(model), "note": None}

    # -- chat -----------------------------------------------------------
    def _payload(self, messages, model, temperature, max_tokens, reasoning_effort) -> dict:
        p = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
             "stream": True}
        if self.kind == "openrouter":
            p["usage"] = {"include": True}
            ent = self._cached_entry(model)
            p.update(reasoning_params("openrouter", reasoning_effort, supports=ent["supports_reasoning"] if ent else None))
        else:
            p["stream_options"] = {"include_usage": True}
            p.update(reasoning_params(self.kind, reasoning_effort, self.profile.get("send_reasoning_effort", False)))
        return p

    def _estimate_cost(self, model: str, pt, ct) -> Optional[float]:
        ent = self._cached_entry(model)
        if not ent or ent.get("price_in") is None or ent.get("price_out") is None or pt is None or ct is None:
            return None
        return (pt * ent["price_in"] + ct * ent["price_out"]) / 1_000_000

    def _make_stats(self, model: str, requested: str, elapsed: float, first_at_rel: Optional[float], usage: dict,
                    stop_reason: Optional[str], native: Optional[dict] = None, n_chars: int = 0) -> GenerationStats:
        ct = usage.get("completion_tokens")
        pt = usage.get("prompt_tokens")
        rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        cost = _num(usage.get("cost"))
        estimated = False
        if cost is None and self.cloud:
            cost = self._estimate_cost(model, pt, ct)
            estimated = cost is not None
        ttft = first_at_rel
        source = "расчёт"
        gen_tps, overall, gen_time = speed_figures(ct, elapsed, ttft)
        if native and _num(native.get("tokens_per_second")):
            # the server's own figures (LM Studio): exact, they win over our measurement
            gen_tps, source = _num(native["tokens_per_second"]), "LM Studio"
            if _num(native.get("time_to_first_token")) is not None:
                ttft = _num(native["time_to_first_token"])
            gen_time = _num(native.get("generation_time")) or gen_time
        return GenerationStats(
            model=model, elapsed_s=round(elapsed, 2), ttft_s=round(ttft, 2) if ttft is not None else None,
            prompt_tokens=pt, completion_tokens=ct, reasoning_tokens=rt, total_tokens=usage.get("total_tokens"),
            tokens_per_sec=round(gen_tps, 1) if gen_tps is not None else None, stop_reason=stop_reason,
            cost=cost, cost_estimated=estimated,
            overall_tokens_per_sec=round(overall, 1) if overall is not None else None,
            gen_time_s=round(gen_time, 2) if gen_time is not None else None, stats_source=source,
            requested_model=requested if requested != model else None)

    def _chat_native(self, messages, model, temperature, max_tokens, eff, reasoning_effort):
        """LM Studio only, no live preview needed: its own endpoint /api/v0/chat/completions returns exact
        statistics (tokens_per_second, time_to_first_token, generation_time). None = not available -> stream."""
        if getattr(self, "_native_failed", False):
            return None
        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                "stream": False}
        body.update(reasoning_params("lmstudio", reasoning_effort))
        t0 = time.monotonic()
        try:
            client, resp = self._send("POST", f"{self.native}/api/v0/chat/completions", body=body, timeout=eff,
                                      model=model)
        except ProviderError as exc:
            if exc.retryable or "подключиться" in str(exc) or "не ответил" in str(exc):
                raise
            self._native_failed = True
            return None
        try:
            data = resp.json()
        except ValueError:
            data = None
        finally:
            resp.close()
            client.close()
        elapsed = time.monotonic() - t0
        if not isinstance(data, dict) or not data.get("choices") or not isinstance(data.get("stats"), dict):
            self._native_failed = True
            return None
        msg = data["choices"][0].get("message") or {}
        text = msg.get("content") or ""
        visible, _ = strip_reasoning(text, keep=True)
        stats = self._make_stats(model, model, elapsed, None, data.get("usage") or {},
                                 data["choices"][0].get("finish_reason"), native=data["stats"])
        logger.info("LLM host=%s model=%s status=200 elapsed=%.1fs (native stats)", self.host, model, elapsed)
        return visible, stats

    def _chat_once(self, messages, models: list[str], temperature, max_tokens, eff, reasoning_effort, on_delta):
        """ONE streamed request (for OpenRouter `models` may hold up to 3 names: the service itself falls back
        to the next one). Returns (visible text, stats); the model that really answered is in stats.model."""
        model = models[0]
        url = f"{self.v1}/chat/completions"
        payload = self._payload(messages, model, temperature, max_tokens, reasoning_effort)
        if self.kind == "openrouter" and len(models) > 1:
            payload["models"] = list(models)
        t0 = time.monotonic()
        first_at: Optional[float] = None
        parts: list[str] = []
        usage: dict = {}
        stop_reason: Optional[str] = None
        answered = model
        client, resp = self._send("POST", url, body=payload, timeout=eff, stream=True, model=model)
        try:
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
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("error"):
                    e = provider_errors.parse_error(chunk)
                    err = self._explain(e, answered, None, stream=True)
                    if parts:   # something was already received: keep it, tell that the answer is cut off
                        visible, _ = strip_reasoning("".join(parts), keep=True)
                        err.partial = visible
                        err.partial_stats = self._make_stats(answered, model, time.monotonic() - t0,
                                                             (first_at - t0) if first_at else None, usage, "error")
                        err.retryable = False
                    raise err
                if isinstance(chunk.get("model"), str) and chunk["model"]:
                    answered = chunk["model"]
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                ch = choices[0]
                if ch.get("finish_reason"):
                    stop_reason = ch["finish_reason"]
                delta = ch.get("delta") or {}
                content = delta.get("content")
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if (content or reasoning) and first_at is None:
                    first_at = time.monotonic()      # first token of ANY kind: reasoning counts too
                if content:
                    parts.append(content)
                    if on_delta:
                        on_delta(content)
        except httpx.TimeoutException as exc:
            err = ProviderError(f"Провайдер «{self.host}» не ответил за {eff:g} сек. Увеличьте таймаут "
                                f"в настройках или проверьте, не перегружена ли модель.")
            self._attach_partial(err, parts, answered, model, t0, first_at, usage)
            raise err from exc
        except httpx.ConnectError as exc:
            raise self._conn_error() from exc
        except httpx.HTTPError as exc:
            err = ProviderError(self._redact(f"Соединение с «{self.host}» прервано: {type(exc).__name__}"))
            self._attach_partial(err, parts, answered, model, t0, first_at, usage)
            raise err from exc
        finally:
            resp.close()
            client.close()
        elapsed = time.monotonic() - t0
        stats = self._make_stats(answered, model, elapsed, (first_at - t0) if first_at is not None else None, usage,
                                 stop_reason)
        logger.info("LLM host=%s model=%s status=200 elapsed=%.1fs", self.host, answered, elapsed)
        visible, _ = strip_reasoning("".join(parts), keep=True)
        return visible, stats

    def _attach_partial(self, err, parts, answered, model, t0, first_at, usage) -> None:
        if parts:
            visible, _ = strip_reasoning("".join(parts), keep=True)
            err.partial = visible
            err.partial_stats = self._make_stats(answered, model, time.monotonic() - t0,
                                                 (first_at - t0) if first_at else None, usage, "error")

    def chat_with_stats(self, messages: list[dict], model: str, temperature: float = 0.2,
                        max_tokens: int = 1024, timeout: Optional[float] = None,
                        reasoning_effort: Optional[str] = None,
                        on_delta: Optional[Callable[[str], None]] = None,
                        fallback_models: Optional[list] = None,
                        on_status: Optional[Callable[[str], None]] = None) -> tuple[str, GenerationStats]:
        """Streams the answer. When the provider is busy / fails (429, 5xx, an error event before any text),
        the backup models from the profile (`fallback_models`) are tried one after another; stats.model is
        the model that really answered. A stream that breaks AFTER text arrived raises ProviderError with
        `.partial` (the text so far)."""
        if not model:
            raise ProviderError("Не выбрана модель.")
        eff = timeout or self.timeout
        backups = fallback_models if fallback_models is not None else self.profile.get("fallback_models")
        cands = [model] + [m for m in _clean_models(backups) if m != model]
        tried: list[tuple[str, str]] = []
        i = 0
        while i < len(cands):
            group = cands[i:i + 3] if (self.kind == "openrouter" and len(cands) - i > 1) else [cands[i]]
            try:
                if self.kind == "lmstudio" and on_delta is None and len(group) == 1:
                    res = self._chat_native(messages, group[0], temperature, max_tokens, eff, reasoning_effort)
                    if res is not None:
                        return res
                text, stats = self._chat_once(messages, group, temperature, max_tokens, eff, reasoning_effort, on_delta)
            except ProviderError as exc:
                nxt = i + len(group)
                if exc.partial or not exc.retryable or nxt >= len(cands):
                    if tried:
                        names = ", ".join(f"«{m}»" for m, _ in tried + [(group[0], "")])
                        exc.args = (f"{exc} (Пробовали модели по очереди: {names} — ни одна не ответила.)",)
                    raise
                tried.append((group[0], str(exc)))
                logger.info("LLM host=%s model=%s failed (%s) -> next backup model %s", self.host, group[0], exc.kind,
                            cands[nxt])
                if on_status:
                    on_status(f"Модель «{group[0]}» не ответила — пробую запасную «{cands[nxt]}»…")
                i = nxt
                continue
            if tried:
                stats.requested_model = model if stats.model != model else None
                stats.fallback_note = (f"Основная модель «{model}» не ответила ({_first_sentence(tried[0][1])}); "
                                       f"ответила «{stats.model}».")
            elif stats.model != model and self.kind == "openrouter" and len(group) > 1:
                stats.fallback_note = (f"Основная модель «{model}» была недоступна — OpenRouter переключил запрос на "
                                       f"«{stats.model}».")
            return text, stats
        raise ProviderError("Не удалось получить ответ от модели.")  # pragma: no cover

    def chat(self, messages: list[dict], model: str, temperature: float = 0.2, max_tokens: int = 1024,
             timeout: Optional[float] = None, stream: bool = False, reasoning_effort: Optional[str] = None) -> str:
        return self.chat_with_stats(messages, model, temperature, max_tokens, timeout, reasoning_effort)[0]

    def test_completion(self, model: str) -> dict:
        """Tiny real request (10 tokens) for the «Проверить» button: latency and cost."""
        try:
            text, st = self.chat_with_stats([{"role": "user", "content": "Ответь одним словом: привет"}],
                                            model=model, temperature=0.0, max_tokens=10, timeout=min(self.timeout, 60))
            return {"ok": True, "latency_s": st.elapsed_s, "cost": st.cost, "cost_estimated": st.cost_estimated,
                    "model": model, "reply": text[:80], "message": f"Модель ответила за {st.elapsed_s} с"
                    + (f", стоимость ≈ ${st.cost:.5f}" if st.cost else "")}
        except LMStudioError as exc:
            return {"ok": False, "model": model, "message": str(exc)}


def _first_sentence(text: str) -> str:
    t = (text or "").strip()
    i = t.find(". ")
    return (t[:i] if i > 0 else t.rstrip("."))[:160]


def clear_model_cache() -> None:
    _MODEL_CACHE.clear()
