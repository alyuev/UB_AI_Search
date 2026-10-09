"""Understanding error answers of LLM providers (especially OpenRouter) and explaining them in plain Russian.

OpenRouter errors come in two shapes: a normal HTTP error with a JSON body, or - for a request that already
started streaming - an `error` event inside the stream. In both cases the body looks like
{"error": {"code": 429, "message": "...", "metadata": {"provider_name": "...", "raw": "..."}}}.
The important thing for the user: WHO failed (the provider of the model, or their own account) and WHAT to do.
"""
from __future__ import annotations

import json
import re
from typing import Optional

_UPSTREAM_RE = re.compile(r"Upstream error from ([A-Za-z0-9 ._-]{2,40})", re.I)


def _raw_text(raw) -> str:
    """The provider's own message out of metadata.raw (a string or an already parsed object)."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        t = raw.strip()
        if t[:1] in "{[":
            try:
                return _raw_text(json.loads(t))
            except ValueError:
                return t
        return t
    if isinstance(raw, dict):
        err = raw.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err.get("status") or json.dumps(err, ensure_ascii=False))
        if isinstance(err, str):
            return err
        return str(raw.get("message") or json.dumps(raw, ensure_ascii=False))
    return str(raw)


def parse_error(obj) -> dict:
    """Normalise an error object/body. Returns {code, message, provider, raw, free_tier_cap}; never raises."""
    out = {"code": None, "message": "", "provider": "", "raw": "", "text": ""}
    try:
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except ValueError:
                obj = {"message": obj}
        err = obj.get("error") if isinstance(obj, dict) else None
        if err is None and isinstance(obj, dict):
            err = obj
        if isinstance(err, str):
            err = {"message": err}
        if not isinstance(err, dict):
            return out
        code = err.get("code")
        out["code"] = int(code) if isinstance(code, (int, float)) or (isinstance(code, str) and code.isdigit()) else code
        out["message"] = str(err.get("message") or "")
        meta = err.get("metadata") if isinstance(err.get("metadata"), dict) else {}
        out["provider"] = str(meta.get("provider_name") or "")
        out["raw"] = _raw_text(meta.get("raw"))
        if not out["provider"]:
            m = _UPSTREAM_RE.search(out["message"] + " " + out["raw"])
            if m:
                out["provider"] = m.group(1).strip()
    except Exception:  # noqa: BLE001
        pass
    out["text"] = " ".join(x for x in (out["message"], out["raw"]) if x)
    return out


def _short(text: str, n: int = 240) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def classify(e: dict, status: Optional[int] = None) -> str:
    """"account_day" | "account_minute" | "provider_busy" | "server" | "key" | "money" | "access" | "missing" | "other"."""
    code = e.get("code") if isinstance(e.get("code"), int) else status
    if status and not isinstance(e.get("code"), int):
        code = status
    low = (e.get("text") or "").lower()
    if "free-models-per-day" in low or "per-day" in low or "per day" in low or "daily" in low:
        return "account_day"
    if "free-models-per-min" in low or "per-min" in low or "per minute" in low:
        return "account_minute"
    if code == 401:
        return "key"
    if code == 402:
        return "money"
    if code == 403:
        return "access"
    if code == 404:
        return "missing"
    if code == 429 or "resourceexhausted" in low or "rate limit" in low or "rate-limited" in low \
            or "limit reached" in low or "overloaded" in low or "capacity" in low:
        return "provider_busy"
    if isinstance(code, int) and code >= 500:
        return "server"
    if "upstream" in low:
        return "provider_busy"
    return "other"


TIPS_BUSY = ("Что можно сделать: подождать около минуты и повторить; выбрать другую бесплатную модель; "
             "указать модель «openrouter/free» (сервис сам подберёт свободную бесплатную модель); "
             "добавить запасные модели в настройках провайдера — программа тогда сама переключится на следующую; "
             "пополнение баланса на OpenRouter открывает платные модели и больше бесплатных запросов.")


def explain(e: dict, provider_title: str, model: Optional[str] = None, status: Optional[int] = None,
            stream: bool = False, openrouter: bool = True) -> tuple[str, str]:
    """(plain-Russian message, kind). `provider_title` is the display name of OUR provider ("OpenRouter"),
    e["provider"] the company that serves the model (Nvidia, Google AI Studio...)."""
    kind = classify(e, status)
    who = e.get("provider") or ""
    detail = _short(e.get("raw") or e.get("message") or "")
    if e.get("raw") and e.get("message") and e["message"].lower() not in ("provider returned error",) \
            and e["message"] not in e["raw"]:
        detail = _short(e["message"] + ": " + e["raw"])
    suffix = f" Сообщение сервера: {detail}" if detail else ""
    mid = " Ответ оборвался на середине." if stream else ""
    if kind == "account_day":
        return (f"Исчерпан дневной лимит бесплатных запросов вашей учётной записи {provider_title}. "
                f"Это лимит именно вашего аккаунта, он обновляется раз в сутки.{mid} Что можно сделать: "
                f"дождаться завтрашнего дня, пополнить баланс (после этого бесплатных запросов в день становится "
                f"больше и открываются платные модели) или переключиться на локальную модель.{suffix}", kind)
    if kind == "account_minute":
        return (f"Превышено число бесплатных запросов в минуту для вашей учётной записи {provider_title}.{mid} "
                f"Подождите минуту и повторите. Пополнение баланса поднимает этот лимит.{suffix}", kind)
    if kind == "provider_busy":
        code = f" ({status})" if status else ""
        if not openrouter:
            return (f"Сервер {provider_title} сообщает, что запросов слишком много или он перегружен{code}.{mid} "
                    f"Подождите немного и повторите.{suffix}", kind)
        head = f"Провайдер модели ({who}) перегружен" if who else "Провайдер модели перегружен"
        return (f"{head}{code}: сейчас у него нет свободных мест для запросов.{mid} Это временная нехватка мощности "
                f"у самого поставщика (бесплатные модели делят общую очередь), а не лимит вашей учётной записи. "
                f"{TIPS_BUSY}{suffix}", kind)
    if kind == "server":
        head = (f"Ошибка на стороне провайдера модели ({who})" if who else
                "Ошибка на стороне провайдера" + (f" ({status})" if status else ""))
        return (f"{head}.{mid} Это сбой у поставщика, не у вас. Подождите немного и повторите или выберите другую "
                f"модель (можно добавить запасные в настройках провайдера).{suffix}", kind)
    if kind == "key":
        return "Неверный или отсутствующий ключ API (401). Проверьте ключ в настройках провайдера." + suffix, kind
    if kind == "money":
        return (f"Недостаточно средств на счёте {provider_title} (402). Пополните баланс или выберите бесплатную "
                f"модель (с пометкой :free).{suffix}"), kind
    if kind == "access":
        return ("Доступ запрещён (403): у ключа нет прав на эту модель, запрос не прошёл проверку содержимого "
                "или регион недоступен." + suffix), kind
    if kind == "missing":
        base = (f"Модель «{model}» или адрес не найдены (404)." if model else "Адрес не найден (404).")
        return (base + " Возможно, бесплатную модель отключили или переименовали — обновите список моделей и выберите "
                       "другую." + suffix), kind
    return f"Провайдер отклонил запрос{f' ({status})' if status else ''}.{mid}{suffix}", kind


def is_retryable(kind: str) -> bool:
    return kind in ("provider_busy", "server")


def describe_account(data: dict) -> dict:
    """Account state from OpenRouter's GET /key in plain words (no numbers of our own: only what the answer says)."""
    d = data.get("data") if isinstance(data.get("data"), dict) else data
    lines: list[str] = []
    is_free = d.get("is_free_tier")
    if is_free is True:
        lines.append("Бесплатный уровень: баланс ещё ни разу не пополнялся, поэтому действуют самые строгие "
                     "ограничения на бесплатные запросы.")
    elif is_free is False:
        lines.append("Баланс пополнялся — ограничения на бесплатные запросы мягче, доступны и платные модели.")
    usage = d.get("usage")
    if isinstance(usage, (int, float)):
        lines.append(f"Потрачено по этому ключу всего: ${usage:.4f}" if usage else
                     "По этому ключу пока ничего не потрачено.")
    limit, remaining = d.get("limit"), d.get("limit_remaining")
    if isinstance(limit, (int, float)):
        if isinstance(remaining, (int, float)):
            lines.append(f"Лимит расходов на ключе: осталось ${remaining:g} из ${limit:g}.")
        else:
            lines.append(f"Лимит расходов на ключе: ${limit:g}.")
    elif "limit" in d:
        lines.append("Лимит расходов на ключе не задан.")
    label = d.get("label")
    return {"is_free_tier": is_free, "usage": usage, "limit": limit, "limit_remaining": remaining,
            "label": label if isinstance(label, str) else None, "lines": lines,
            "summary": " ".join(lines) if lines else "Ключ принят, подробностей о счёте сервис не сообщил."}
