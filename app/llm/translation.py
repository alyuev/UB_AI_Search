"""Translation quality analysis: direct DB retrieval (get_range, NOT semantic search),
batched LLM review against the 10-criteria spec, JSON parsing with fallback, and a Markdown report.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from app import options as opts_mod
from app import prompts as prompts_mod
from app.llm.lmstudio import LMStudioClient, LMStudioError, llm_title
from app.version import __version__
from app.llm.rag import estimate_tokens

try:
    from app.log import get_logger
except ImportError:  # pragma: no cover
    from tests.fakes.log import get_logger

logger = get_logger("analysis")

from app.llm.translation_prompts import (  # noqa: F401
    CRITERIA, SYSTEM_PROMPT, SYSTEM_PROMPT_V1, SYSTEM_PROMPT_V2, VERIFY_PROMPT, VERIFY_USER_TEMPLATE,
)

SEVERITIES = ("critical", "major", "minor")
MAX_BATCH_PROMPTS = 40  # per-batch prompt hashes kept in the experiment snapshot (the rest is only counted)


@dataclass
class Issue:
    id: int
    severity: str
    category: str
    problem: str
    why_significant: str = ""
    suggestion: str = ""
    en: str = ""
    ru: str = ""
    en_quote: str = ""   # exact EN fragment the model points at (evidence)
    ru_quote: str = ""   # exact RU fragment ("" for pure omissions)
    verified: Optional[bool] = None  # None = verification pass not run
    verify_reason: str = ""
    paper_title: Optional[dict] = None
    section_title: Optional[dict] = None

    def to_dict(self):
        d = {
            "id": self.id, "severity": self.severity, "category": self.category,
            "problem": self.problem, "why_significant": self.why_significant,
            "suggestion": self.suggestion, "en": self.en, "ru": self.ru,
            "en_quote": self.en_quote, "ru_quote": self.ru_quote,
        }
        if self.verified is not None:
            d["verified"] = self.verified
            d["verify_reason"] = self.verify_reason
        if self.paper_title:
            d["paper_title"] = self.paper_title
        if self.section_title:
            d["section_title"] = self.section_title
        return d


def _format_batch(paragraphs: list[dict], hints: bool = True) -> str:
    lines = []
    for p in paragraphs:
        lines.append(f"### ID {p['id']} (Paper {p.get('paper')}, Section {p.get('section')}, Para {p.get('paragraph')})")
        lines.append(f"EN: {p.get('en','')}")
        lines.append(f"RU: {p.get('ru','')}")
        hint = length_hint(p.get("en", ""), p.get("ru", "")) if hints else ""
        if hint:
            lines.append(hint)
        lines.append("")
    return "\n".join(lines)


# RU/EN character-length ratio over the whole corpus (13k paragraphs): 5%-95% = 0.85-1.16,
# 1%-99% = 0.78-1.25. A paragraph far outside that band is a cheap, objective signal that a
# clause may be missing (short) or invented (long) -- surfaced to the model as a hint.
RATIO_LOW, RATIO_HIGH = 0.80, 1.30


def length_hint(en: str, ru: str) -> str:
    if len(en) < 60 or not ru:
        return ""
    ratio = len(ru) / len(en)
    if ratio < RATIO_LOW:
        return (f"[hint: RU/EN length ratio {ratio:.2f} (typical 0.85-1.16) - the RU is unusually "
                f"SHORT; check carefully whether a clause, qualifier or sentence of the EN is missing]")
    if ratio > RATIO_HIGH:
        return (f"[hint: RU/EN length ratio {ratio:.2f} (typical 0.85-1.16) - the RU is unusually "
                f"LONG; check carefully whether something not in the EN was added]")
    return ""


def make_batches(paragraphs: list[dict], max_context_tokens: int, reserved_tokens: int = 1500,
                 max_items: int = 8) -> list[list[dict]]:
    """Split paragraphs into batches that fit the model's context window AND hold at most
    `max_items` paragraphs: small batches keep the per-paragraph attention (recall) high, keep
    the JSON answer short enough not to hit max_tokens, and give the UI finer progress."""
    budget = max(500, max_context_tokens - reserved_tokens)
    batches: list[list[dict]] = []
    current: list[dict] = []
    used = 0
    for p in paragraphs:
        cost = estimate_tokens(p.get("en", "") + p.get("ru", "")) + 20
        if (used + cost > budget or len(current) >= max_items) and current:
            batches.append(current)
            current = []
            used = 0
        current.append(p)
        used += cost
    if current:
        batches.append(current)
    return batches


def parse_llm_json(text: str) -> list[dict]:
    """Robustly parse a JSON array of issues from raw LLM text, with fallbacks."""
    if not text:
        return []
    text = text.strip()
    # strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("issues", []) or []
    except json.JSONDecodeError:
        pass
    # fallback: find the first [...] block
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    # fallback: find individual {...} objects
    objs = []
    for m in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL):
        try:
            objs.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
    return objs


_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _norm_tokens(text: str) -> list[str]:
    return [w.lower().replace("ё", "е") for w in _WORD.findall(text or "")]


def quote_supported(quote: str, source: str, threshold: float = 0.7) -> bool:
    """True if (nearly) all words of `quote` occur in `source` -- tolerant to small quoting
    slips (punctuation, an ellipsis) but rejects fabricated 'evidence'. An empty quote is
    handled by the caller (it is only legitimate for ru_quote on pure omissions)."""
    q = _norm_tokens(quote)
    if not q:
        return False
    src = set(_norm_tokens(source))
    hit = sum(1 for w in q if w in src)
    return hit / len(q) >= threshold


def parse_verdict(text: str) -> tuple[bool, str]:
    """Parse the verification pass answer. Anything unparseable counts as 'rejected'
    (conservative: an issue we cannot confirm is not reported)."""
    m = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
    if isinstance(obj, dict):
        v = str(obj.get("verdict", "")).lower()
        return v.startswith("confirm"), str(obj.get("reason", ""))
    low = (text or "").lower()
    if "confirmed" in low and "rejected" not in low:
        return True, ""
    return False, "unparseable verification answer"


def _as_str(v) -> str:
    return "" if v is None else str(v)


def review_batch(llm, model: str, batch: list[dict], *, temperature: float = 0.1,
                 max_tokens: int = 2048, timeout: float = 180,
                 reasoning_effort: Optional[str] = "none", verify: bool = True,
                 system_prompt: Optional[str] = None, verify_prompt: Optional[str] = None,
                 verify_user_template: Optional[str] = None) -> dict:
    """One LLM review of a batch of paragraphs (+ optional verification pass for the
    reported substantial issues). Shared by TranslationAnalyzer and the benchmark so both
    measure exactly the same pipeline.

    Returns {"issues": [Issue], "dropped": [{"id","reason",...}], "stats": [GenerationStats],
             "raw": str}. Raises LMStudioError (never swallowed here).
    """
    by_id = {str(p["id"]): p for p in batch}
    messages = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user", "content": _format_batch(batch, hints=((system_prompt or SYSTEM_PROMPT) is not SYSTEM_PROMPT_V1))},
    ]
    raw, stats = llm.chat_with_stats(
        messages, model=model, temperature=temperature, max_tokens=max_tokens,
        timeout=timeout, reasoning_effort=reasoning_effort,
    )
    all_stats = [stats]
    user_message = messages[1]["content"]
    issues: list[Issue] = []
    dropped: list[dict] = []
    v2 = (system_prompt or SYSTEM_PROMPT) is not SYSTEM_PROMPT_V1
    for item in parse_llm_json(raw):
        if not isinstance(item, dict):
            continue
        pid = item.get("id")
        # single-paragraph batches: tolerate the model echoing a different id
        if (pid is None or str(pid) not in by_id) and len(batch) == 1:
            pid = batch[0]["id"]
        if pid is None or str(pid) not in by_id:
            dropped.append({"id": pid, "reason": "unknown_id"})
            continue
        para = by_id[str(pid)]
        sev = item.get("severity")
        en_q, ru_q = _as_str(item.get("en_quote")), _as_str(item.get("ru_quote"))
        if v2:
            # evidence check: the EN quote must really come from the EN paragraph; the
            # RU quote (if any) from the RU paragraph. Fabricated evidence => discard.
            # At least one quote is required (an omission has no RU fragment, an addition
            # has no EN fragment); every non-empty quote must really occur in its text.
            if not en_q.strip() and not ru_q.strip():
                dropped.append({"id": str(pid), "reason": "no_evidence"})
                continue
            if en_q.strip() and not quote_supported(en_q, para.get("en", "")):
                dropped.append({"id": str(pid), "reason": "en_quote_not_in_source", "en_quote": en_q})
                continue
            if ru_q.strip() and not quote_supported(ru_q, para.get("ru", "")):
                dropped.append({"id": str(pid), "reason": "ru_quote_not_in_source", "ru_quote": ru_q})
                continue
        issue = Issue(
            id=str(pid),
            severity=sev if sev in SEVERITIES else "minor",
            category=_as_str(item.get("category")) or "meaning_accuracy",
            problem=_as_str(item.get("problem")),
            why_significant=_as_str(item.get("why_significant")),
            suggestion=_as_str(item.get("suggestion")),
            en=para.get("en", ""), ru=para.get("ru", ""),
            en_quote=en_q, ru_quote=ru_q,
            paper_title=para.get("paper_title"), section_title=para.get("section_title"),
        )
        if verify and issue.severity in ("critical", "major"):
            vmsg = [
                {"role": "system", "content": verify_prompt or VERIFY_PROMPT},
                {"role": "user", "content": prompts_mod.render(
                    verify_user_template or VERIFY_USER_TEMPLATE,
                    en=issue.en, ru=issue.ru, category=issue.category, severity=issue.severity,
                    en_quote=issue.en_quote, ru_quote=issue.ru_quote or "(missing)", problem=issue.problem)},
            ]
            vraw, vstats = llm.chat_with_stats(
                vmsg, model=model, temperature=0.0, max_tokens=300, timeout=timeout,
                reasoning_effort=reasoning_effort,
            )
            all_stats.append(vstats)
            ok, reason = parse_verdict(vraw)
            issue.verified, issue.verify_reason = ok, reason
            if not ok:
                dropped.append({"id": issue.id, "reason": "rejected_by_verification",
                                "detail": reason, "problem": issue.problem})
                continue
        issues.append(issue)
    truncated = stats.stop_reason == "length"
    if truncated:
        logger.warning("review answer hit max_tokens (%s completion tokens) - JSON may be cut off; "
                       "salvaged %d issue(s)", stats.completion_tokens, len(issues))
    return {"issues": issues, "dropped": dropped, "stats": all_stats, "raw": raw, "truncated": truncated,
            "user_message": user_message}


class TranslationAnalyzer:
    def __init__(self, conn, llm_client: LMStudioClient, model: str,
                 temperature: float = 0.1, max_tokens: int = 2048,
                 timeout: float = 180, max_context_length: int = 8192,
                 reasoning_effort: Optional[str] = "none", verify: Optional[bool] = None,
                 options: Optional[dict] = None, prompts=None, meta: Optional[dict] = None):
        o = dict(opts_mod.DEFAULTS)
        o.update(options or {})
        if verify is not None:
            o["translation_verify"] = bool(verify)
        self.o = o
        self.verify = bool(o["translation_verify"])
        self.batch_size = int(o["translation_batch_size"])
        self.min_severity = o["translation_min_severity"]
        self.prompts = prompts or prompts_mod.ResolvedPrompts()
        self.meta = meta or {}
        self.conn = conn
        self.llm = llm_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_context_length = max_context_length
        # See docs/04_benchmark_translation.md: reasoning ON was benchmarked and,
        # at equal max_tokens, left too little budget for the actual JSON answer
        # (0/5 detections, truncated output) and was ~3x slower per pair; reasoning
        # OFF detected the same corruptions reliably and fast. Default OFF here too,
        # still overridable (e.g. "auto"/"low"/"high") per job request.
        self.reasoning_effort = reasoning_effort

    def analyze(self, paper: int, section: Optional[int] = None,
                para_from: Optional[int] = None, para_to: Optional[int] = None,
                progress: Optional[Callable[[str, int, int, Optional[int], str], None]] = None) -> dict:
        from app.core import db as core_db  # deferred import; provided by the core agent

        def report(stage, current, total, current_id=None, message=""):
            if progress:
                progress(stage, current, total, current_id, message)

        report("Сравнение EN/RU", 0, 1, None, "выборка абзацев из БД")
        paragraphs = core_db.get_range(self.conn, paper, section, para_from, para_to)
        total_paras = len(paragraphs)
        if not paragraphs:
            return {"issues": [], "summary": "Нет абзацев в указанном диапазоне.", "paragraphs_analyzed": 0,
                    "prompt_warnings": self.prompts.warnings, "warnings": list(self.prompts.warnings)}

        t_total = time.monotonic()
        review_prompt = self.prompts.text("translation_review")
        verify_prompt = self.prompts.text("translation_verify")
        verify_user = self.prompts.text("translation_verify_user")
        batches = make_batches(paragraphs, self.max_context_length,
                                reserved_tokens=self.max_tokens + estimate_tokens(review_prompt) + 300,
                                max_items=self.batch_size)
        first_user: Optional[str] = None
        batch_prompts: list[dict] = []  # per batch: size + hash of the user message (not the full text)
        report("Сравнение EN/RU", 0, total_paras, paragraphs[0]["id"],
               f"Paper {paper} / Section {section or '-'}: 0/{total_paras}")

        all_issues: list[Issue] = []
        batch_stats: list[dict] = []
        dropped_all: list[dict] = []
        truncated_batches = 0
        done = 0
        for bi, batch in enumerate(batches):
            report("Анализ смысловых расхождений", done, total_paras, batch[0]["id"],
                   f"Paper {paper} / Section {section or '-'}: батч {bi+1}/{len(batches)} · "
                   f"отправка в {llm_title(self.llm)} · {self.model}")
            try:
                res = review_batch(
                    self.llm, self.model, batch, temperature=self.temperature,
                    max_tokens=self.max_tokens, timeout=self.timeout,
                    reasoning_effort=self.reasoning_effort, verify=self.verify,
                    system_prompt=review_prompt, verify_prompt=verify_prompt, verify_user_template=verify_user,
                )
                if first_user is None:
                    first_user = res.get("user_message")
                um = res.get("user_message") or ""
                batch_prompts.append({"batch": bi + 1, "paragraphs": len(batch), "chars": len(um),
                                      "hash": prompts_mod.text_hash(um)})
                for st in res["stats"]:
                    batch_stats.append(st.to_dict())
                if res["dropped"]:
                    dropped_all.extend(res["dropped"])
                    logger.info("batch %d/%d: dropped %d candidate issue(s): %s", bi + 1, len(batches),
                                len(res["dropped"]), [(d["id"], d["reason"]) for d in res["dropped"]])
                logger.info("batch %d/%d: %s", bi + 1, len(batches), res["stats"][0].summary_line())
                report("Анализ смысловых расхождений", done, total_paras, batch[0]["id"],
                       f"Paper {paper} / Section {section or '-'}: батч {bi+1}/{len(batches)} — "
                       f"{res['stats'][0].summary_line()}")
                all_issues.extend(res["issues"])
                truncated_batches += int(res["truncated"])
            except LMStudioError as exc:
                # Surface this to the caller as a real error rather than silently
                # reporting "no issues found" -- a missing/unreachable model must
                # never look like a clean bill of health for the translation.
                logger.error("translation batch failed: %s", exc)
                report("Анализ смысловых расхождений", done, total_paras, batch[0]["id"],
                       f"ошибка: {exc}")
                return {
                    "issues": [], "summary": None, "error": str(exc),
                    "paragraphs_analyzed": total_paras, "batch_stats": batch_stats,
                    "stats_summary": None,
                    "prompt_warnings": self.prompts.warnings, "warnings": list(self.prompts.warnings),
                }
            done += len(batch)
            report("Анализ смысловых расхождений", done, total_paras, batch[-1]["id"],
                   f"Paper {paper} / Section {section or '-'}: {done}/{total_paras}")

        report("Формирование отчёта", total_paras, total_paras, None, "агрегация результатов")
        rank = {"minor": 0, "major": 1, "critical": 2}
        n_before = len(all_issues)
        all_issues = [i for i in all_issues if rank.get(i.severity, 0) >= rank.get(self.min_severity, 0)]
        below_threshold = n_before - len(all_issues)
        summary = self._summarize(all_issues, total_paras)
        if below_threshold:
            summary += f" Скрыто замечаний ниже выбранного порога: {below_threshold}."

        agg = self._aggregate_stats(batch_stats)
        if agg:
            logger.info("aggregate: %s", agg["summary"])
        report("Формирование отчёта", total_paras, total_paras, None, "готово")

        return {
            "issues": [i.to_dict() for i in all_issues],
            "summary": summary,
            "paragraphs_analyzed": total_paras,
            "batch_stats": batch_stats,
            "stats_summary": agg["summary"] if agg else None,
            "dropped_candidates": dropped_all,
            "truncated_batches": truncated_batches,
            "below_threshold": below_threshold,
            "experiment": self._experiment(review_prompt, verify_prompt, verify_user, first_user, total_paras,
                                           len(batches), len(dropped_all), truncated_batches, below_threshold,
                                           agg, time.monotonic() - t_total, batch_prompts),
            "prompt_warnings": self.prompts.warnings,
            "warnings": list(self.prompts.warnings),
            "verify": self.verify,
        }

    def _experiment(self, review_prompt, verify_prompt, verify_user, first_user, total_paras, n_batches,
                    n_dropped, truncated, below_threshold, agg, total_s,
                    batch_prompts=None) -> dict:
        used = ["translation_review"] + (["translation_verify", "translation_verify_user"] if self.verify else [])
        return {
            "kind": "translation",
            "app": {"version": __version__, "date": _dt.date.today().isoformat()},
            "model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
            "reasoning": {"setting": self.meta.get("reasoning_setting"), "effort": self.reasoning_effort},
            "context_length": self.max_context_length,
            "preset": self.prompts.preset,
            "prompt_versions": self.prompts.versions(used),
            "prompt_texts": self.prompts.texts(used),
            "prompts": {"system": review_prompt, "user": (first_user or "")[:8000],
                        "user_note": "показан полный запрос только для первой порции абзацев; "
                                     "для остальных — размер и хэш в batches",
                        "batches": (batch_prompts or [])[:MAX_BATCH_PROMPTS],
                        "batches_total": len(batch_prompts or []),
                        "verify_system": verify_prompt if self.verify else None,
                        "verify_user": verify_user if self.verify else None},
            "options": dict(self.o),
            "search_settings": {},
            "retrieval": {"paragraphs": total_paras, "batches": n_batches, "batch_size": self.batch_size,
                          "verify": self.verify, "dropped_candidates": n_dropped,
                          "truncated_batches": truncated, "below_threshold": below_threshold,
                          "min_severity": self.min_severity},
            "plan": None,
            "timings": {"total_s": round(total_s, 2)},
            "stats": agg,
        }

    @staticmethod
    def _aggregate_stats(batch_stats: list[dict]) -> Optional[dict]:
        if not batch_stats:
            return None
        model = batch_stats[0].get("model")
        total_elapsed = sum(b.get("elapsed_s") or 0 for b in batch_stats)
        total_completion = sum(b.get("completion_tokens") or 0 for b in batch_stats)
        total_prompt = sum(b.get("prompt_tokens") or 0 for b in batch_stats)
        overall_tps = round(total_completion / total_elapsed, 1) if total_elapsed > 0 else None
        # generation speed: tokens / time spent generating (without waiting for the first token of each call)
        timed = [b for b in batch_stats if b.get("gen_time_s") and b.get("completion_tokens")]
        gen_time = sum(b["gen_time_s"] for b in timed)
        avg_tps = round(sum(b["completion_tokens"] for b in timed) / gen_time, 1) if gen_time > 0 else overall_tps
        costs = [b["cost"] for b in batch_stats if b.get("cost") is not None]
        total_cost = sum(costs) if costs else None
        summary = (f"{model} · {len(batch_stats)} батч(ей) · генерация {avg_tps} ток/с (в среднем) · "
                   f"всего {round(total_elapsed)} с · {total_prompt} → {total_completion} ток.")
        if total_cost is not None:
            summary += " · бесплатно" if total_cost == 0 else f" · стоимость ≈ ${total_cost:.4f}"
        return {"summary": summary, "total_elapsed_s": round(total_elapsed, 1),
                "total_prompt_tokens": total_prompt, "total_completion_tokens": total_completion,
                "avg_tokens_per_sec": avg_tps, "overall_tokens_per_sec": overall_tps, "cost": total_cost}

    @staticmethod
    def _summarize(issues: list[Issue], total_paras: int) -> str:
        if not issues:
            return f"Проверено абзацев: {total_paras}. Существенных расхождений не найдено."
        counts = {s: 0 for s in SEVERITIES}
        for i in issues:
            counts[i.severity] = counts.get(i.severity, 0) + 1
        return (f"Проверено абзацев: {total_paras}. Найдено проблем: {len(issues)} "
                f"(критичных: {counts['critical']}, значительных: {counts['major']}, "
                f"незначительных: {counts['minor']}).")


def export_markdown(report: dict, title: str = "Отчёт об анализе перевода") -> str:
    lines = [f"# {title}", "", report.get("summary", "")]
    if report.get("stats_summary"):
        lines.append(f"\n_Производительность:_ {report['stats_summary']}")
    lines.append("")
    issues = report.get("issues", [])
    if not issues:
        lines.append("Проблем не обнаружено.")
        return "\n".join(lines)
    sev_order = {"critical": 0, "major": 1, "minor": 2}
    for issue in sorted(issues, key=lambda i: sev_order.get(i["severity"], 3)):
        lines.append(f"## [{issue['id']}] {issue['severity'].upper()} — {issue['category']}")
        pt, st = issue.get("paper_title"), issue.get("section_title")
        if pt or st:
            title_bits = []
            if pt:
                title_bits.append(f"«{pt.get('ru') or pt.get('en') or ''}»")
            if st:
                title_bits.append(f"— {st.get('ru') or st.get('en') or ''}")
            lines.append(" ".join(title_bits))
        lines.append("")
        lines.append(f"**Проблема:** {issue['problem']}")
        lines.append("")
        if issue.get("why_significant"):
            lines.append(f"**Почему это важно:** {issue['why_significant']}")
            lines.append("")
        if issue.get("suggestion"):
            lines.append(f"**Предложение:** {issue['suggestion']}")
            lines.append("")
        lines.append(f"> EN: {issue.get('en','')}")
        lines.append(">")
        lines.append(f"> RU: {issue.get('ru','')}")
        lines.append("")
    return "\n".join(lines)
