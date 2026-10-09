r"""Eval FTS variants / semantic / hybrid on the Paper-0 scratch DB.
Usage: python bench/foreword/eval.py  (writes docs/05_fts_tuning_foreword.md)
Builds paper0.db via build_scratch.py if missing.
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from app.core.search import Searcher
from app.config import DEFAULT_SETTINGS

DB = HERE / "paper0.db"
W = DEFAULT_SETTINGS["fts_weights"]
ONES = {k: 1.0 for k in W}


def metrics(ranked, rel):
    r5 = len(set(ranked[:5]) & rel) / len(rel)
    r10 = len(set(ranked[:10]) & rel) / len(rel)
    mrr = 0.0
    for i, pid in enumerate(ranked[:10]):
        if pid in rel:
            mrr = 1 / (i + 1); break
    return r5, r10, mrr


def main():
    if not DB.exists():
        import build_scratch
        sys.argv = [sys.argv[0]]
        exec(open(HERE / "build_scratch.py").read())
    from app.core.embedder import create_embedder
    emb = create_embedder("bge-m3", runtime="llamacpp-vulkan", device="auto")
    s = Searcher(DB, embedder=emb)
    qs = json.load(open(HERE / "queries.json", encoding="utf-8"))["queries"]

    def cfg(weights, titles, mode="and"):
        def run(q):
            s.fts_weights, s.fts_titles, s.fts_mode_standalone, s.fts_mode_hybrid = weights, titles, mode, mode
            return s.fts(q, k=10)
        return run

    def sem(q): return s.semantic(q, k=10)

    def hyb(weights, titles, mode="and"):
        def run(q):
            s.fts_weights, s.fts_titles, s.fts_mode_standalone, s.fts_mode_hybrid = weights, titles, mode, mode
            return s.hybrid(q, k=10)
        return run

    systems = {
        "FTS old (unweighted, no titles)": cfg(ONES, False),
        "FTS weighted, no titles": cfg(W, False),
        "FTS unweighted + titles": cfg(ONES, True),
        "FTS weighted + titles": cfg(W, True),
        "FTS OR-mode, unweighted, no titles": cfg(ONES, False, "or"),
        "FTS OR-mode, weighted + titles": cfg(W, True, "or"),
        "Semantic (bge-m3)": sem,
        "Hybrid RRF (old FTS)": hyb(ONES, False),
        "Hybrid RRF (weighted + titles)": hyb(W, True),
        "Hybrid RRF (OR-mode, weighted + titles)": hyb(W, True, "or"),
        "NEW DEFAULT standalone FTS (and_fallback_or, weighted+titles)": cfg(W, True, "and_fallback_or"),
        "NEW DEFAULT hybrid (OR FTS, weighted+titles)": hyb(W, True, "or"),
    }
    res = {n: [] for n in systems}
    per_q = {}
    for q in qs:
        rel = {r["id"] for r in q["relevant"]}
        per_q[q["id"]] = {}
        for n, fn in systems.items():
            ranked = [r["id"] for r in fn(q["query"])]
            m = metrics(ranked, rel)
            res[n].append(m)
            per_q[q["id"]][n] = m
    lines = ["| System | Recall@5 | Recall@10 | MRR@10 |", "|---|---|---|---|"]
    for n, ms in res.items():
        a = [sum(x[i] for x in ms) / len(ms) for i in range(3)]
        lines.append(f"| {n} | {a[0]:.3f} | {a[1]:.3f} | {a[2]:.3f} |")
    table = "\n".join(lines)
    print(table)
    # by language
    langs = {}
    for q, in [(q,) for q in qs]:
        langs.setdefault(q["lang"], []).append(q["id"])
    lt = ["| System | RU Recall@10 | EN Recall@10 | RU MRR | EN MRR |", "|---|---|---|---|---|"]
    for n in systems:
        row = []
        for idx in (1, 2):
            for lg in ("ru", "en"):
                v = [per_q[i][n][idx] for i in langs[lg]]
                row.append(sum(v) / len(v))
        lt.append(f"| {n} | {row[0]:.3f} | {row[1]:.3f} | {row[2]:.3f} | {row[3]:.3f} |")
    ltable = "\n".join(lt)
    pq = ["| Query | " + " | ".join(n.split(" (")[0] for n in systems) + " |", "|---|" + "---|" * len(systems)]
    for q in qs:
        pq.append(f"| {q['id']} {q['query'][:40]} | " + " | ".join(f"{per_q[q['id']][n][2]:.2f}" for n in systems) + " |")
    doc = f"""# FTS tuning on Paper 0 (Foreword)

> **Caveat: the relevance judgments (`bench/foreword/queries.json`) are a DRAFT, status "draft - needs user review".**
> They were judged by one reader (an AI agent) on {len(qs)} queries over 171 paragraphs, so differences of a few
> hundredths are noise. Treat results as a smoke test of the mechanism, not as a tuning conclusion for the whole book.

Setup: scratch DB `bench/foreword/paper0.db` (Paper 0 only, bge-m3 via llama.cpp Vulkan). Weights (bm25, positional): `{json.dumps(W)}`.
"Titles" = the new `title_en`/`title_ru` FTS columns (paper + section titles) both indexed and matched.
Reproduce: `.venv/Scripts/python bench/foreword/eval.py`.

"AND" (current default) requires every query word to match, which is why long natural-language queries score ~0 in FTS;
"OR-mode" (`fts_mode_*` settings) matches any word and lets bm25 rank. Column weights and titles
barely move the numbers on Paper 0 (short titles, few title-driven queries); the big effect is AND vs OR.

## Overall (mean over {len(qs)} queries)

{table}

## By query language (Recall@10 / MRR@10)

{ltable}

## Per-query MRR@10

{chr(10).join(pq)}

Notes: queries have 2-8 relevant paragraphs (list-style paragraphs inflate counts); Foreword section titles are
short, so title weight mainly helps queries naming a section ("The Trinities", "God the Sevenfold").
"""
    (ROOT / "docs" / "05_fts_tuning_foreword.md").write_text(doc, encoding="utf-8")
    print("wrote docs/05_fts_tuning_foreword.md")


if __name__ == "__main__":
    main()
