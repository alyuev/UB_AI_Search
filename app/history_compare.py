"""Side-by-side comparison of two history entries: what differed (parameters, prompt versions,
prompt text lines) and how the runs measured up (speed, tokens, sources)."""
from __future__ import annotations

import difflib
import json
from typing import Any, Optional

from app import options as opts_mod

RETRIEVAL_LABELS = {
    "k": "Абзацев передано модели (запрошено)", "fallback_plain_hybrid": "Поиск без подготовки запроса",
    "candidates_fts_asked": "Кандидатов по словам (запрошено)",
    "candidates_semantic_asked": "Кандидатов по смыслу (запрошено)", "rrf_k": "Константа объединения (RRF)",
    "retrieved": "Найдено абзацев", "after_relevance_filter": "Осталось после порога релевантности",
    "with_neighbors": "С учётом соседних абзацев", "sources_sent": "Абзацев реально передано модели",
    "sources_lang": "Язык абзацев для модели", "context_share": "Доля контекста под источники",
    "paragraphs": "Абзацев проверено", "batches": "Порций", "batch_size": "Абзацев в порции",
    "verify": "Второй проход проверки", "dropped_candidates": "Отброшено кандидатов",
    "truncated_batches": "Порций с обрезанным ответом", "below_threshold": "Скрыто ниже порога",
    "min_severity": "Порог важности",
}
TOP_LABELS = {
    "model": "Модель", "temperature": "Температура", "max_tokens": "Максимум токенов ответа",
    "context_length": "Длина контекста модели", "preset": "Набор промтов",
    "reasoning.setting": "Рассуждение модели", "reasoning.effort": "Рассуждение (передано модели)",
    "app.version": "Версия программы", "plan.summary": "Поисковые слова",
    "search_settings.fts_weights": "Веса поиска по словам", "search_settings.fts_mode_hybrid": "Режим слов (гибридный)",
    "search_settings.fts_mode_standalone": "Режим слов (fts)",
    "search_settings.fts_stopwords": "Стоп-слова в поиске",
}


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "да" if v else "нет"
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    return str(v)


def _flatten(exp: dict) -> dict:
    kind = exp.get("kind")
    flat: dict[str, Any] = {}
    for k in ("model", "temperature", "max_tokens", "context_length", "preset"):
        flat[k] = exp.get(k)
    r = exp.get("reasoning") or {}
    flat["reasoning.setting"], flat["reasoning.effort"] = r.get("setting"), r.get("effort")
    flat["app.version"] = (exp.get("app") or {}).get("version")
    flat["plan.summary"] = (exp.get("plan") or {}).get("summary") if kind == "rag" else None
    for k, v in (exp.get("search_settings") or {}).items():
        flat[f"search_settings.{k}"] = v
    prefix = "translation_" if kind == "translation" else "rag_"
    reasoning_key = "reasoning_translation" if kind == "translation" else "reasoning_rag"
    for k, v in (exp.get("options") or {}).items():
        if k.startswith(prefix) or k == reasoning_key:
            flat[f"options.{k}"] = v
    for k, v in (exp.get("retrieval") or {}).items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"retrieval.{k}.{kk}"] = vv
        else:
            flat[f"retrieval.{k}"] = v
    for pid, ver in (exp.get("prompt_versions") or {}).items():
        flat[f"prompt_versions.{pid}"] = ver.get("label") if isinstance(ver, dict) else ver
    return flat


def _label(key: str) -> str:
    if key in TOP_LABELS:
        return TOP_LABELS[key]
    if key.startswith("options."):
        spec = opts_mod.SPEC.get(key[8:])
        return spec["label"] if spec else key
    if key.startswith("retrieval."):
        rest = key[10:]
        return RETRIEVAL_LABELS.get(rest.split(".")[0], rest) + (
            f" ({rest.split('.', 1)[1]})" if "." in rest else "")
    if key.startswith("prompt_versions."):
        return "Версия промта: " + _prompt_title(key[16:])
    return key


def _prompt_title(pid: str) -> str:
    try:
        from app import prompts as P

        return P.specs()[pid]["title"]
    except Exception:  # noqa: BLE001
        return pid


def _text_diff(a: str, b: str) -> list[dict]:
    lines = list(difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=1))
    out = []
    for ln in lines:
        if ln.startswith("---") or ln.startswith("+++"):
            continue
        if ln.startswith("@@"):
            out.append({"op": "@", "text": "…"})
        elif ln.startswith("+"):
            out.append({"op": "+", "text": ln[1:]})
        elif ln.startswith("-"):
            out.append({"op": "-", "text": ln[1:]})
        else:
            out.append({"op": "=", "text": ln[1:] if ln.startswith(" ") else ln})
    return out


def _metrics(entry: dict) -> dict:
    exp = entry.get("experiment") or {}
    st = exp.get("stats") or {}
    tm = exp.get("timings") or {}
    res = entry.get("result") or {}
    ret = exp.get("retrieval") or {}
    m = {
        "Время всего, с": tm.get("total_s") or entry.get("duration_s"),
        "Скорость генерации, ток/с": st.get("tokens_per_sec") or st.get("avg_tokens_per_sec"),
        "Скорость общая (с ожиданием), ток/с": st.get("overall_tokens_per_sec"),
        "До первого токена, с": st.get("ttft_s"),
        "Токенов в запросе": st.get("prompt_tokens") or st.get("total_prompt_tokens"),
        "Токенов в ответе": st.get("completion_tokens") or st.get("total_completion_tokens"),
        "Абзацев передано модели": ret.get("sources_sent"),
        "Сослался на абзацев": len(res.get("cited_ids") or []) if entry.get("mode") == "rag" else None,
        "Замечаний в отчёте": len(res.get("issues") or []) if entry.get("mode") == "translation" else None,
        "Длина ответа, знаков": len(res.get("answer") or "") if entry.get("mode") == "rag" else None,
    }
    return {k: v for k, v in m.items() if v is not None}


def compare(a: dict, b: dict) -> dict:
    """a, b: full history entries (with result and experiment)."""
    ea, eb = a.get("experiment"), b.get("experiment")
    out: dict[str, Any] = {
        "same_query": (a.get("query") or "").strip().lower() == (b.get("query") or "").strip().lower(),
        "same_mode": a.get("mode") == b.get("mode"),
        "has_experiment": {"a": ea is not None, "b": eb is not None},
        "param_diffs": [], "prompt_diffs": [], "metrics": [],
    }
    ma, mb = _metrics(a), _metrics(b)
    for k in list(ma) + [k for k in mb if k not in ma]:
        out["metrics"].append({"label": k, "a": ma.get(k), "b": mb.get(k)})
    if ea is None or eb is None:
        return out
    fa, fb = _flatten(ea), _flatten(eb)
    for k in list(fa) + [k for k in fb if k not in fa]:
        va, vb = fa.get(k), fb.get(k)
        if json.dumps(va, sort_keys=True, default=str) != json.dumps(vb, sort_keys=True, default=str):
            out["param_diffs"].append({"key": k, "label": _label(k), "a": _fmt(va), "b": _fmt(vb)})
    ta, tb = ea.get("prompt_texts") or {}, eb.get("prompt_texts") or {}
    for pid in list(ta) + [k for k in tb if k not in ta]:
        title = _prompt_title(pid)
        if pid in ta and pid in tb:
            if ta[pid] == tb[pid]:
                out["prompt_diffs"].append({"id": pid, "title": title, "changed": False})
            else:
                out["prompt_diffs"].append({"id": pid, "title": title, "changed": True,
                                            "diff": _text_diff(ta[pid], tb[pid])})
        else:
            only = "a" if pid in ta else "b"
            out["prompt_diffs"].append({"id": pid, "title": title, "changed": True, "only_in": only,
                                        "text": ta.get(pid) or tb.get(pid)})
    return out
