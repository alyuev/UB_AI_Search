"""Translation-analysis benchmark against the model CURRENTLY LOADED in LM Studio.

Never loads or switches models (uses resolve_model(None) = the loaded one; if none is
loaded it reports and exits). Runs the exact production pipeline
(app.llm.translation.review_batch) on bench/translation/dataset.json (Paper 0 slice).

Configurations (--configs, comma-separated, run in order):
  v1    original prompt, no verification   (the "before")
  v2    evidence-based prompt, quote check in code, no verification pass
  v2v   v2 + LLM verification pass for critical/major candidates  (production default)

Metrics: a case counts as "flagged" iff at least one critical/major issue survives the
pipeline (minor findings are reported separately as `any_flag`). Precision/recall are over
that flag; per-type detection and false-positive rates are listed; times/tokens include the
verification calls.

Usage:
  python bench/translation/run_translation_bench.py [--configs v1,v2,v2v] [--reasoning off|on]
Results are merged into bench/translation/results.json and docs/04_benchmark_translation.md
is regenerated from everything stored there.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from app.llm.lmstudio import LMStudioClient, LMStudioError  # noqa: E402
from app.llm.rag import resolve_reasoning_effort  # noqa: E402
from app.llm.translation import SYSTEM_PROMPT_V1, SYSTEM_PROMPT_V2, review_batch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(HERE, "dataset.json")
RESULTS_PATH = os.path.join(HERE, "results.json")
REPORT_PATH = os.path.join(ROOT, "docs", "04_benchmark_translation.md")
LMSTUDIO_URL = "http://localhost:1234"

CONFIGS = {
    "v1": dict(system_prompt=SYSTEM_PROMPT_V1, verify=False, label="v1: исходный промпт, без проверки"),
    "v2": dict(system_prompt=SYSTEM_PROMPT_V2, verify=False, label="v2: промпт с доказательствами + проверка цитат"),
    "v2v": dict(system_prompt=SYSTEM_PROMPT_V2, verify=True, label="v2v: v2 + верификационный проход"),
}


def run_config(client, model, cases, cfg_name, reasoning_effort, max_tokens):
    cfg = CONFIGS[cfg_name]
    rows = []
    t_all = time.time()
    for case in cases:
        batch = [{"id": case["para_id"], "paper": 0, "section": None, "paragraph": None,
                  "en": case["en"], "ru": case["ru_test"]}]
        t0 = time.time()
        try:
            res = review_batch(client, model, batch, temperature=0.1, max_tokens=max_tokens,
                               timeout=300, reasoning_effort=reasoning_effort,
                               verify=cfg["verify"], system_prompt=cfg["system_prompt"])
        except LMStudioError as exc:
            rows.append({**case, "error": str(exc)})
            print(f"  {case['id']}: ERROR {exc}")
            continue
        issues = res["issues"]
        substantial = [i for i in issues if i.severity in ("critical", "major")]
        comp = sum((s.completion_tokens or 0) for s in res["stats"])
        prompt = sum((s.prompt_tokens or 0) for s in res["stats"])
        reasoning = sum((s.reasoning_tokens or 0) for s in res["stats"])
        elapsed = sum(s.elapsed_s for s in res["stats"])
        row = {
            "id": case["id"], "issue_type": case["issue_type"], "expect_flagged": case["expect_flagged"],
            "flagged": bool(substantial), "any_flag": bool(issues),
            "issues": [{"severity": i.severity, "category": i.category, "problem": i.problem,
                        "en_quote": i.en_quote, "ru_quote": i.ru_quote, "verified": i.verified}
                       for i in issues],
            "dropped": res["dropped"], "elapsed_s": round(elapsed, 1), "wall_s": round(time.time() - t0, 1),
            "prompt_tokens": prompt, "completion_tokens": comp, "reasoning_tokens": reasoning,
            "calls": len(res["stats"]),
            "ttft_s": res["stats"][0].ttft_s, "tok_s": res["stats"][0].tokens_per_sec,
        }
        rows.append(row)
        ok = "OK " if row["flagged"] == row["expect_flagged"] else "MISS"
        print(f"  [{ok}] {case['id']:<34} flagged={row['flagged']!s:<5} any={row['any_flag']!s:<5} "
              f"{row['wall_s']}s calls={row['calls']}")
    return {"rows": rows, "wall_total_s": round(time.time() - t_all, 1)}


def metrics(rows):
    valid = [r for r in rows if "error" not in r]
    tp = sum(1 for r in valid if r["flagged"] and r["expect_flagged"])
    fp = sum(1 for r in valid if r["flagged"] and not r["expect_flagged"])
    tn = sum(1 for r in valid if not r["flagged"] and not r["expect_flagged"])
    fn = sum(1 for r in valid if not r["flagged"] and r["expect_flagged"])
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    per_type = {}
    for r in valid:
        t = per_type.setdefault(r["issue_type"], {"n": 0, "flagged": 0, "any_flag": 0})
        t["n"] += 1
        t["flagged"] += int(r["flagged"])
        t["any_flag"] += int(r["any_flag"])
    n = max(1, len(valid))
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
        "per_type": per_type,
        "avg_wall_s": round(sum(r["wall_s"] for r in valid) / n, 1),
        "avg_calls": round(sum(r["calls"] for r in valid) / n, 2),
        "tokens_prompt": sum(r["prompt_tokens"] for r in valid),
        "tokens_completion": sum(r["completion_tokens"] for r in valid),
        "tokens_reasoning": sum(r["reasoning_tokens"] for r in valid),
        "errors": len(rows) - len(valid),
    }


def fmt(x):
    return "n/a" if x is None else f"{x:.2f}"


def write_report(store):
    lines = ["# Бенчмарк анализа перевода", ""]
    lines += [
        "Датасет: `bench/translation/dataset.json` — 17 реальных абзацев **Paper 0 (Предисловие)**: "
        "7 намеренных искажений (пропуск, добавление, искажение смысла, неверный термин, инверсия "
        "отрицания, неверное число, частичный пропуск), 4 допустимых перефразирования (синонимы, "
        "перестановка, залог, слияние предложений) и 6 официальных переводов без изменений. Все "
        "варианты написаны вручную (`build_dataset.py`), чтобы каждое искажение было реальным, а каждое "
        "перефразирование — смысл-сохраняющим.", "",
        "«Отмечено» = после всего конвейера остался хотя бы один **critical/major**-пункт "
        "(minor учитывается отдельно как `any`). Это тот же код, что и в рабочем анализе "
        "(`app.llm.translation.review_batch`).", ""]
    # group by model+reasoning
    groups = {}
    for key, entry in store.items():
        groups.setdefault((entry["model"], entry["reasoning"]), []).append((key, entry))
    order = {"v1": 0, "v2": 1, "v2v": 2}
    for (model, reasoning), entries in groups.items():
        lines += [f"## Модель `{model}`, рассуждение: {reasoning}", ""]
        entries.sort(key=lambda e: order.get(e[1]["config"], 9))
        lines += ["| Конфигурация | Precision | Recall | F1 | TP/FP/TN/FN | с/абзац | вызовов | ток. (prompt→compl.) |",
                  "|---|---|---|---|---|---|---|---|"]
        for key, e in entries:
            m = e["metrics"]
            lines.append(f"| {CONFIGS[e['config']]['label']} | **{fmt(m['precision'])}** | **{fmt(m['recall'])}** | "
                         f"{fmt(m['f1'])} | {m['tp']}/{m['fp']}/{m['tn']}/{m['fn']} | {m['avg_wall_s']} | "
                         f"{m['avg_calls']} | {m['tokens_prompt']}→{m['tokens_completion']} |")
        lines.append("")
        for key, e in entries:
            m = e["metrics"]
            lines += [f"### {CONFIGS[e['config']]['label']}", "",
                      "| Тип | Кейсов | Отмечено (critical/major) | Любой пункт (в т.ч. minor) |", "|---|---|---|---|"]
            for t, c in m["per_type"].items():
                lines.append(f"| {t} | {c['n']} | {c['flagged']} | {c['any_flag']} |")
            lines += ["", "<details><summary>Построчно</summary>", "",
                      "| Кейс | Ожидалось | Отмечено | Пункты | с | вызовов |", "|---|---|---|---|---|---|"]
            for r in e["rows"]:
                if "error" in r:
                    lines.append(f"| {r['id']} | {r['expect_flagged']} | ОШИБКА | {r['error'][:60]} | | |")
                    continue
                probs = "; ".join(f"{i['severity']}: {i['problem'][:70]}" for i in r["issues"]) or "—"
                lines.append(f"| {r['id']} | {r['expect_flagged']} | {r['flagged']} | {probs} | {r['wall_s']} | {r['calls']} |")
            lines += ["", "</details>", ""]
    lines += [
        "## История и выводы", "",
        "- **Первый прогон** (gemma-4-26b-a4b-it-qat, рассуждение выкл., старый промпт v1, предыдущий "
        "датасет из случайных абзацев): Precision 0.45, Recall 1.00 (TP=5, FP=6): модель отмечала "
        "практически всё, включая перефразирования и неизменённые пары (но и «recall 1.00» был завышен: "
        "кейс wrong_term в том датасете совпадал с оригиналом). С включённым рассуждением (по умолчанию "
        "модели) при max_tokens=1500 всё бюджет уходил в reasoning_content (1497 ток.), JSON-ответа не "
        "оставалось: 0/5 обнаружений, ~58 с на пару вместо ~8 с.",
        "- Поэтому по умолчанию рассуждение для анализа перевода выключено (`reasoning_effort=none`), "
        "а точность повышается промптом v2 (доказательства-цитаты, список «не ошибок», калибровочные "
        "примеры, «[] — нормальный ответ»), проверкой цитат в коде и верификационным проходом.", ""]
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="v1,v2,v2v")
    ap.add_argument("--reasoning", choices=["auto", "on", "off"], default="off")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--limit", type=int, default=0, help="only first N cases (quick check)")
    args = ap.parse_args()

    client = LMStudioClient(LMSTUDIO_URL)
    try:
        resolved = client.resolve_model(None)
    except LMStudioError as exc:
        print(f"Бенчмарк пропущен: {exc}")
        return
    model = resolved["model"]
    print(f"model={model} context={resolved['context']} reasoning={args.reasoning}")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if args.limit:
        cases = cases[: args.limit]
    effort = resolve_reasoning_effort(args.reasoning)

    store = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            store = json.load(f)

    for cfg_name in args.configs.split(","):
        print(f"=== {cfg_name} ===")
        out = run_config(client, model, cases, cfg_name, effort, args.max_tokens)
        m = metrics(out["rows"])
        print(f"  precision={fmt(m['precision'])} recall={fmt(m['recall'])} f1={fmt(m['f1'])} "
              f"TP/FP/TN/FN={m['tp']}/{m['fp']}/{m['tn']}/{m['fn']} avg={m['avg_wall_s']}s")
        store[f"{model}|{args.reasoning}|{cfg_name}"] = {
            "model": model, "reasoning": args.reasoning, "config": cfg_name,
            "rows": out["rows"], "metrics": m, "wall_total_s": out["wall_total_s"],
            "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=1)
        write_report(store)
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
