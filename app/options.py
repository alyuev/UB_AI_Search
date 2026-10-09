"""Tunable answer/analysis parameters ("Параметры ответа" in Настройки).

One registry drives everything: the defaults used by the pipelines, the validation of saved
values, and the form rendered in the UI (GET /api/options). Defaults reproduce the behaviour the
app had before these knobs existed, so nothing changes until the user edits a value.
Values live in data/settings.json next to the other settings and apply to the next question,
without a restart.
"""
from __future__ import annotations

from typing import Any

GROUPS = [
    ("model_behavior", "Поведение модели"),
    ("rag_sources", "Вопрос к тексту: поиск и источники"),
    ("rag_answer", "Вопрос к тексту: ответ"),
    ("translation", "Анализ перевода"),
]

_CH_REASON = [
    {"value": "auto", "label": "авто (рекомендация по данным)"},
    {"value": "on", "label": "вкл"},
    {"value": "off", "label": "выкл"},
]

# fmt: off
OPTIONS: list[dict] = [
    # ---- model behaviour -------------------------------------------------------------------------
    dict(key="reasoning_rag", group="model_behavior", type="choice", default="auto", choices=_CH_REASON,
         label="Рассуждение модели при ответе на вопрос",
         hint="«Выкл» быстрее и надёжнее для этой модели (по результатам проверок); «авто» = выкл."),
    dict(key="reasoning_translation", group="model_behavior", type="choice", default="auto", choices=_CH_REASON,
         label="Рассуждение модели при анализе перевода",
         hint="То же для анализа перевода: «выкл» быстрее и стабильнее."),
    dict(key="rag_strictness", group="model_behavior", type="choice", default="allow_inferences",
         choices=[{"value": "allow_inferences", "label": "Разрешить помеченные выводы (рекомендуется)"},
                  {"value": "facts_only", "label": "Только прямые факты"}],
         label="Если прямого ответа в книге нет",
         hint="«Помеченные выводы»: ответ, следующий из нескольких найденных фактов (например, разность лет), "
              "показывается отдельным блоком «Вывод» с шагами, ссылками и допущениями."),
    dict(key="rag_query_rewrite", group="model_behavior", type="bool", default=True,
         label="Подготовка запроса перед поиском",
         hint="Модель выделяет ключевые слова вопроса, в том числе английские, и поиск находит больше нужного. "
              "Стоит несколько секунд."),
    dict(key="rag_rewrite_max_terms", group="model_behavior", type="int", default=6, min=1, max=12, step=1,
         label="Сколько ключевых слов можно искать",
         hint="Верхняя граница числа слов в плане поиска (отдельно для русских и английских)."),
    # ---- RAG: retrieval and sources ----------------------------------------------------------------
    dict(key="rag_top_k", group="rag_sources", type="int", default=12, min=1, max=40, step=1,
         label="Сколько абзацев передавать модели",
         hint="Больше абзацев — шире охват, но длиннее запрос и медленнее ответ."),
    dict(key="rag_context_share", group="rag_sources", type="float", default=1.0, min=0.1, max=1.0, step=0.05,
         label="Какую долю свободного контекста можно отдать источникам",
         hint="1 — всё свободное место; меньше — оставить запас под ответ. Лишние абзацы отбрасываются с конца."),
    dict(key="rag_candidates_fts", group="rag_sources", type="int", default=0, min=0, max=200, step=1,
         label="Кандидатов из поиска по словам (0 — авто)",
         hint="Сколько абзацев берёт поиск по словам до объединения результатов. Авто: втрое больше числа источников (не меньше 30)."),
    dict(key="rag_candidates_semantic", group="rag_sources", type="int", default=0, min=0, max=200, step=1,
         label="Кандидатов из поиска по смыслу (0 — авто)",
         hint="То же для поиска по смыслу."),
    dict(key="rag_rrf_k", group="rag_sources", type="int", default=60, min=1, max=200, step=1,
         label="Константа объединения результатов (RRF)",
         hint="Чем меньше, тем сильнее выигрывают самые верхние места списков; обычно 60."),
    dict(key="rag_rare_term_max_hits", group="rag_sources", type="int", default=8, min=0, max=50, step=1,
         label="Редкое слово: не больше скольки абзацев",
         hint="Слово из вопроса, которое встречается в книге в таком числе абзацев или реже, считается редким. "
              "Абзацы с ним всегда попадают к модели. 0 — не использовать."),
    dict(key="rag_rare_term_slots", group="rag_sources", type="int", default=5, min=0, max=20, step=1,
         label="Мест под абзацы с редкими словами",
         hint="Сколько из переданных абзацев резервируется под те, где есть редкое слово вопроса (даже если по смыслу они не в топе)."),
    dict(key="rag_neighbors", group="rag_sources", type="int", default=0, min=0, max=2, step=1,
         label="Соседние абзацы вокруг каждого источника",
         hint="0 — только найденные; 1 или 2 — добавить столько абзацев до и после каждого (даёт контекст, но удлиняет запрос). "
              "Соседей тоже можно цитировать."),
    dict(key="rag_auto_expand", group="rag_sources", type="bool", default=True,
         label="Подтягивать разделы и списки целиком, когда это нужно",
         hint="Программа сама решает: если вопрос просит перечислить («перечисли…», «кто входит…») или совпадает "
              "с названием раздела книги, модели передаётся весь раздел (сколько влезет в контекст); найденный "
              "нумерованный список передаётся со всеми пунктами."),
    dict(key="rag_min_relevance", group="rag_sources", type="float", default=0.0, min=0.0, max=0.95, step=0.05,
         label="Отбрасывать слабо подходящие абзацы",
         hint="0 — не отбрасывать; 0,5 — оставить только абзацы, набравшие не меньше половины очков лучшего."),
    dict(key="rag_sources_lang", group="rag_sources", type="choice", default="both",
         choices=[{"value": "both", "label": "английский и русский"}, {"value": "ru", "label": "только русский"},
                  {"value": "en", "label": "только английский"}],
         label="На каком языке показывать абзацы модели",
         hint="Один язык вдвое короче: в контекст помещается больше абзацев. Карточки источников всегда двуязычные."),
    # ---- RAG: answer -------------------------------------------------------------------------------------
    dict(key="rag_answer_lang", group="rag_answer", type="choice", default="question",
         choices=[{"value": "question", "label": "на языке вопроса"}, {"value": "ru", "label": "всегда по-русски"}],
         label="Язык ответа",
         hint="«Всегда по-русски» добавляет в инструкцию отдельное правило (его можно править в «Промтах»)."),
    # ---- translation analysis ---------------------------------------------------------------------------
    dict(key="translation_batch_size", group="translation", type="int", default=8, min=1, max=30, step=1,
         label="Сколько абзацев проверять за один запрос к модели",
         hint="Меньше — внимательнее и точнее, но больше запросов; больше — быстрее, но модель может пропускать."),
    dict(key="translation_verify", group="translation", type="bool", default=True,
         label="Проверять найденные замечания вторым проходом",
         hint="Модель перепроверяет каждое серьёзное замечание и отбрасывает ложные. Дольше, но точнее."),
    dict(key="translation_min_severity", group="translation", type="choice", default="minor",
         choices=[{"value": "minor", "label": "показывать все"}, {"value": "major", "label": "значительные и критичные"},
                  {"value": "critical", "label": "только критичные"}],
         label="Какие замечания показывать в отчёте",
         hint="Порог важности: более мелкие замечания в отчёт не попадают."),
]
# fmt: on

SPEC = {o["key"]: o for o in OPTIONS}
DEFAULTS = {o["key"]: o["default"] for o in OPTIONS}


def coerce(key: str, value: Any) -> Any:
    """Validate and normalise one option value; raises ValueError with a plain-Russian message."""
    spec = SPEC.get(key)
    if spec is None:
        raise ValueError(f"неизвестный параметр «{key}»")
    t = spec["type"]
    label = spec["label"]
    if t == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false", "1", "0"):
            return value.lower() in ("true", "1")
        raise ValueError(f"«{label}»: ожидается «вкл» или «выкл»")
    if t in ("int", "float"):
        if isinstance(value, bool):  # True/False must not silently become 1/0
            raise ValueError(f"«{label}»: ожидается число")
        try:
            v = int(value) if t == "int" and not isinstance(value, bool) else float(value)
        except (TypeError, ValueError):
            raise ValueError(f"«{label}»: ожидается число")
        if t == "int" and isinstance(value, float) and value != int(value):
            raise ValueError(f"«{label}»: ожидается целое число")
        if v != v:  # NaN
            raise ValueError(f"«{label}»: ожидается число")
        lo, hi = spec["min"], spec["max"]
        if v < lo or v > hi:
            raise ValueError(f"«{label}»: допустимо от {lo:g} до {hi:g}")
        return int(v) if t == "int" else float(v)
    if t == "choice":
        allowed = [c["value"] for c in spec["choices"]]
        if value not in allowed:
            raise ValueError(f"«{label}»: допустимо одно из {', '.join(allowed)}")
        return value
    raise ValueError(f"«{label}»: неподдерживаемый тип")


def effective(settings: dict | None) -> dict:
    """All option values: stored ones (if valid) over the defaults. A bad stored value never
    breaks a request: it silently falls back to the default."""
    out = dict(DEFAULTS)
    for k in DEFAULTS:
        if settings and k in settings and settings[k] is not None:
            try:
                out[k] = coerce(k, settings[k])
            except ValueError:
                pass
    return out


def public_spec(settings: dict | None = None) -> dict:
    """What the UI needs to render the form: groups, per-option spec and the current value."""
    cur = effective(settings)
    items = []
    for o in OPTIONS:
        d = {k: v for k, v in o.items()}
        d["value"] = cur[o["key"]]
        items.append(d)
    return {"groups": [{"id": g, "title": t} for g, t in GROUPS], "options": items}
