"""Registry of EVERY prompt the app sends to the LLM, user overrides, validation and presets.

* Defaults live in code (app/llm/rag.py, app/llm/translation_prompts.py) and are never modified.
* The user's edits are stored as overrides only: data/settings.json -> "prompts": {id: text}.
* A prompt may contain placeholders written as {name}. Only the names listed for that prompt are
  allowed; required ones must be present. Literal braces in JSON examples are fine (a placeholder
  is a bare identifier in braces).
* An invalid override can never break a request: it is ignored (default used) and reported as a
  warning that the UI shows.
* Presets (named sets of overrides) live in data/prompt_presets.json and can be exported/imported.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from typing import Optional

from app import config

PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

GROUPS = [
    ("rag", "Вопрос к тексту"),
    ("translation", "Анализ перевода"),
]


def _specs() -> list[dict]:
    """Prompt specs; defaults imported lazily (rag.py imports this module)."""
    from app.llm import rag as R
    from app.llm import translation_prompts as T

    return [
        dict(id="rag_system", group="rag", title="Основная инструкция для ответа на вопрос",
             description="Главные правила: отвечать только по найденным абзацам, ставить номера абзацев после каждого "
                         "утверждения, честно говорить, если данных мало. Отправляется модели первой, в роли «system».",
             placeholders={}, required=[], default=R.SYSTEM_PROMPT),
        dict(id="rag_inference_rules", group="rag", title="Правила помеченных выводов",
             description="Добавляется к основной инструкции только если в настройках выбрано «Разрешить помеченные выводы»: "
                         "как оформлять «Вывод:», шаги, допущения, пример.",
             placeholders={}, required=[], default=R.INFERENCE_ADDENDUM),
        dict(id="rag_answer_lang_ru", group="rag", title="Правило «отвечать всегда по-русски»",
             description="Добавляется к инструкции только если в параметрах ответа выбран язык «всегда по-русски».",
             placeholders={}, required=[], default=R.ANSWER_LANG_RU_PROMPT),
        dict(id="rag_formatting", group="rag", title="Оформление ответа",
             description="Как оформлять ответ: Markdown, жирный/курсив, списки, ссылки на абзацы. Добавляется в самый конец "
                         "инструкции. Чтобы ответ приходил простым текстом, замените содержимое на «Пиши простым текстом без разметки.»",
             placeholders={}, required=[], default=R.FORMATTING_PROMPT),
        dict(id="rag_user_template", group="rag", title="Сообщение с вопросом и источниками",
             description="Как в запрос к модели складываются найденные абзацы и ваш вопрос.",
             placeholders={"sources": "найденные абзацы с номерами (обязательно)",
                           "question": "текст вашего вопроса (обязательно)",
                           "inference_reminder": "напоминание про помеченные выводы (пусто, если выводы запрещены)"},
             required=["sources", "question"], default=R.USER_TEMPLATE),
        dict(id="rag_inference_reminder", group="rag", title="Напоминание про выводы (в конце сообщения)",
             description="Короткое напоминание, которое ставится после вопроса, если разрешены помеченные выводы: "
                         "сначала проверить, нельзя ли вычислить ответ, и не делать «вывод» из пересказа.",
             placeholders={}, required=[], default=R.INFERENCE_REMINDER),
        dict(id="query_rewrite", group="rag", title="Подготовка запроса (план поиска)",
             description="Короткий запрос к модели перед поиском: выделить ключевые слова вопроса (русские и английские) "
                         "и фразу для поиска по смыслу. Модель должна вернуть JSON.",
             placeholders={"max_terms": "максимум ключевых слов (из параметров ответа)"}, required=[],
             default=R.QUERY_REWRITE_PROMPT),
        dict(id="translation_review", group="translation", title="Проверка перевода: основная инструкция",
             description="Как искать существенные расхождения между английским и русским абзацем и в каком виде "
                         "вернуть замечания (JSON с цитатами).",
             placeholders={}, required=[], default=T.SYSTEM_PROMPT_V2),
        dict(id="translation_verify", group="translation", title="Проверка перевода: перепроверка замечания",
             description="Инструкция второго прохода: подтвердить или отклонить найденное замечание.",
             placeholders={}, required=[], default=T.VERIFY_PROMPT),
        dict(id="translation_verify_user", group="translation", title="Проверка перевода: сообщение перепроверки",
             description="Как в запрос второго прохода подставляются абзац и спорное замечание.",
             placeholders={"en": "английский абзац (обязательно)", "ru": "русский абзац (обязательно)",
                           "problem": "описание замечания (обязательно)", "category": "категория замечания",
                           "severity": "важность замечания", "en_quote": "цитата из английского текста",
                           "ru_quote": "цитата из русского текста"},
             required=["en", "ru", "problem"], default=T.VERIFY_USER_TEMPLATE),
    ]


_SPEC_CACHE: Optional[dict] = None


def specs() -> dict[str, dict]:
    global _SPEC_CACHE
    if _SPEC_CACHE is None:
        _SPEC_CACHE = {s["id"]: s for s in _specs()}
    return _SPEC_CACHE


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 3)


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def validate(pid: str, text: str) -> Optional[str]:
    """None if OK, else a plain-Russian error message."""
    spec = specs().get(pid)
    if spec is None:
        return f"неизвестный промт «{pid}»"
    if not isinstance(text, str):
        return "текст промта должен быть строкой"
    if len(text) > 60000:
        return "промт слишком длинный (больше 60 000 знаков)"
    found = set(PLACEHOLDER_RE.findall(text))
    allowed = set(spec["placeholders"])
    unknown = sorted(found - allowed)
    if unknown:
        names = ", ".join("{" + n + "}" for n in unknown)
        ok = ", ".join("{" + n + "}" for n in sorted(allowed)) or "в этом промте подстановок нет"
        return f"неизвестные подстановки: {names}. Допустимо: {ok}"
    missing = sorted(set(spec["required"]) - found)
    if missing:
        return "не хватает обязательных подстановок: " + ", ".join("{" + n + "}" for n in missing)
    return None


def render(text: str, **values) -> str:
    """Replace known {placeholders}; everything else (JSON braces etc.) stays untouched."""
    return PLACEHOLDER_RE.sub(lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), text)


class ResolvedPrompts:
    """Effective prompt texts for one request: valid overrides over the defaults."""

    def __init__(self, overrides: Optional[dict] = None, preset: Optional[str] = None, replay: bool = False,
                 sources: Optional[dict] = None):
        self.preset = preset
        self.replay = replay  # overrides come from a history snapshot (exact re-run)
        # replay only: the saved prompt_versions, so a re-run labels each prompt «стандартный/свой» exactly
        # as the original did (even if the built-in default changed since)
        self._sources = {pid: v.get("source") for pid, v in (sources or {}).items() if isinstance(v, dict)}
        self.overrides: dict[str, str] = {}
        self.warnings: list[str] = []
        sp = specs()
        for pid, text in (overrides or {}).items():
            if pid not in sp:
                self.warnings.append(f"Промт «{pid}» неизвестен — пропущен.")
                continue
            err = validate(pid, text)
            if err:
                self.warnings.append(f"Свой вариант промта «{sp[pid]['title']}» не применён: {err}. "
                                     f"Использован стандартный.")
                continue
            if text.strip() and (replay or text != sp[pid]["default"]):
                self.overrides[pid] = text

    def text(self, pid: str) -> str:
        return self.overrides.get(pid, specs()[pid]["default"])

    def is_custom(self, pid: str) -> bool:
        if pid not in self.overrides:
            return False
        if self.replay:
            src = self._sources.get(pid)
            return src == "custom" if src else self.overrides[pid] != specs()[pid]["default"]
        return True

    def info(self, pid: str) -> dict:
        t = self.text(pid)
        h = text_hash(t)
        custom = self.is_custom(pid)
        return {"id": pid, "source": "custom" if custom else "default", "hash": h,
                "label": (f"свой · {h}" if custom else f"стандартный · {h}"),
                "chars": len(t), "tokens": estimate_tokens(t)}

    def versions(self, ids: list[str]) -> dict:
        out = {}
        for pid in ids:
            i = self.info(pid)
            out[pid] = {"source": i["source"], "hash": i["hash"], "label": i["label"]}
        return out

    def texts(self, ids: list[str]) -> dict:
        return {pid: self.text(pid) for pid in ids}


def from_settings(settings: Optional[dict]) -> ResolvedPrompts:
    s = settings or {}
    return ResolvedPrompts(s.get("prompts") or {}, s.get("prompt_preset"))


def public_list(settings: Optional[dict]) -> dict:
    """Everything the editor needs: specs, defaults, current text, version info, warnings, preset state."""
    rp = from_settings(settings)
    items = []
    for pid, sp in specs().items():
        d = {k: sp[k] for k in ("id", "group", "title", "description", "placeholders", "required", "default")}
        d["text"] = rp.text(pid)
        d["custom"] = rp.is_custom(pid)
        d["info"] = rp.info(pid)
        d["default_info"] = {"hash": text_hash(sp["default"]), "chars": len(sp["default"]),
                             "tokens": estimate_tokens(sp["default"])}
        items.append(d)
    active = (settings or {}).get("prompt_preset")
    presets = load_presets()
    dirty = None
    if active and active in presets:
        dirty = _norm(presets[active]["prompts"]) != _norm(rp.overrides)
    return {"groups": [{"id": g, "title": t} for g, t in GROUPS], "prompts": items, "warnings": rp.warnings,
            "preset": active, "preset_dirty": dirty}


def _norm(d: dict) -> dict:
    return {k: v for k, v in sorted((d or {}).items())}


# ------------------------------------------------------------------------------ presets ---

def _presets_path() -> str:
    return os.path.join(str(config.DATA_DIR), "prompt_presets.json")


def load_presets() -> dict:
    p = _presets_path()
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        pr = data.get("presets", {})
        return pr if isinstance(pr, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_presets(presets: dict) -> None:
    p = _presets_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"format": "smartsearch-prompt-presets", "version": 1, "presets": presets}, f,
                  ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def _clean_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("Укажите название набора")
    if "/" in name or "\\" in name:
        raise ValueError("В названии нельзя использовать символы / и \\")
    if len(name) > 80:
        raise ValueError("Название набора слишком длинное (больше 80 знаков)")
    return name


def save_preset(name: str, overrides: dict, note: str = "") -> str:
    name = _clean_name(name)
    presets = load_presets()
    presets[name] = {"prompts": dict(overrides or {}), "note": note or "",
                     "saved_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    _save_presets(presets)
    return name


def delete_preset(name: str) -> bool:
    presets = load_presets()
    if name not in presets:
        return False
    del presets[name]
    _save_presets(presets)
    return True


def rename_preset(old: str, new: str) -> str:
    new = _clean_name(new)
    presets = load_presets()
    if old not in presets:
        raise KeyError(old)
    if new != old and new in presets:
        raise ValueError(f"Набор «{new}» уже существует")
    presets = {(new if k == old else k): v for k, v in presets.items()}
    _save_presets(presets)
    return new


def get_preset(name: str) -> Optional[dict]:
    return load_presets().get(name)


def export_json(names: Optional[list[str]] = None) -> dict:
    presets = load_presets()
    if names:
        presets = {n: presets[n] for n in names if n in presets}
    return {"format": "smartsearch-prompt-presets", "version": 1, "presets": presets}


def import_json(obj: dict) -> dict:
    """Import presets from an exported file. Invalid prompts are dropped (reported), name clashes get
    a numeric suffix (existing presets are never overwritten)."""
    if not isinstance(obj, dict) or obj.get("format") != "smartsearch-prompt-presets" or \
            not isinstance(obj.get("presets"), dict):
        raise ValueError("Это не файл с наборами промтов этой программы")
    presets = load_presets()
    imported, problems = [], []
    for name, p in obj["presets"].items():
        if not isinstance(p, dict) or not isinstance(p.get("prompts"), dict):
            problems.append(f"«{name}»: неверный формат — пропущен")
            continue
        clean = {}
        for pid, text in p["prompts"].items():
            err = validate(pid, text)
            if err:
                problems.append(f"«{name}», промт «{pid}»: {err} — пропущен")
            else:
                clean[pid] = text
        final = _clean_name(str(name))
        n = 2
        while final in presets:
            final = f"{_clean_name(str(name))} ({n})"
            n += 1
        presets[final] = {"prompts": clean, "note": str(p.get("note", "")),
                          "saved_at": p.get("saved_at") or _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        imported.append(final)
    _save_presets(presets)
    return {"imported": imported, "problems": problems}
