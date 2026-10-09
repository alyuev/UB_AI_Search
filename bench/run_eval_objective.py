"""Objective, harder-to-game evaluations to cross-check the queries.json-based
eval (which is suspected biased toward FTS since relevant IDs were found by
grepping terms). Runs against the already-built subset DBs (bench/bge-m3.db,
bench/multilingual-e5-base.db, both llamacpp-vulkan, 1572 paragraphs).

1) Cross-lingual pair retrieval: RU text -> search EN-only vectors -> rank of
   the true paired EN paragraph (same id), and EN -> RU. Uses the parallel
   corpus itself as ground truth, no manual relevance judging involved.
2) Paraphrase/concept queries: skipped (would need an LLM to generate cheap,
   good paraphrases; noted rather than faked).
3) Pool bias check for the original 29 queries: pool top-10 of FTS +
   semantic(bge-m3) + semantic(e5-base), report how many of each system's
   top-10 are NOT in the existing (grep-derived) qrels ("unjudged").
4) Query latency breakdown: embed time vs matrix-search time, separately,
   using a persistent httpx.Client (keep-alive) talking to llama-server.
5) Sanity-check the actual server invocation used cls/mean pooling and the
   right prefixes.
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.embedder import create_embedder, LLAMACPP_REGISTRY
from app.core.search import Searcher

random.seed(42)

MODELS = ["bge-m3", "multilingual-e5-base"]
QUERIES = json.load(open(ROOT / "bench" / "queries.json", encoding="utf-8"))


def load_subset():
    return json.load(open(ROOT / "bench" / "subset.json", encoding="utf-8"))


# --------------------------------------------------------------------------
# 1) Cross-lingual pair retrieval
# --------------------------------------------------------------------------

def cross_lingual_pair_eval(model_name: str, n_sample=200, k=10):
    db_path = ROOT / "bench" / f"{model_name}.db"
    embedder = create_embedder(model_name, runtime="llamacpp-vulkan")
    searcher = Searcher(str(db_path), embedder=embedder)

    ids = searcher._ids
    id_to_idx = searcher._id_to_idx
    subset = {r["id"]: r for r in load_subset()}

    sample_ids = random.sample(ids, min(n_sample, len(ids)))

    def eval_direction(query_lang, query_text_fn, target_mat):
        ranks = []
        for pid in sample_ids:
            text = query_text_fn(subset[pid])
            if not text.strip():
                continue
            qvec = embedder.embed_query(text)
            sim = target_mat @ qvec
            order = np.argsort(-sim)
            true_idx = id_to_idx[pid]
            rank = int(np.where(order == true_idx)[0][0]) + 1
            ranks.append(rank)
        ranks = np.array(ranks)
        recall_at_1 = float((ranks <= 1).mean())
        recall_at_10 = float((ranks <= 10).mean())
        mrr = float((1.0 / ranks).mean())
        return {"recall@1": recall_at_1, "recall@10": recall_at_10, "mrr": mrr, "n": len(ranks)}

    def first_sentence(text):
        for sep in [". ", "? ", "! ", ".\n"]:
            if sep in text:
                return text.split(sep)[0] + sep.strip()[-1]
        return text[:200]

    results = {
        "ru_full_to_en": eval_direction("ru", lambda r: r["ru"], searcher._en_mat),
        "ru_first_sentence_to_en": eval_direction("ru", lambda r: first_sentence(r["ru"]), searcher._en_mat),
        "en_full_to_ru": eval_direction("en", lambda r: r["en"], searcher._ru_mat),
        "en_first_sentence_to_ru": eval_direction("en", lambda r: first_sentence(r["en"]), searcher._ru_mat),
    }
    if hasattr(embedder, "close"):
        embedder.close()
    return results


# --------------------------------------------------------------------------
# 3) Pool bias check
# --------------------------------------------------------------------------

def pool_bias_check():
    bge_embedder = create_embedder("bge-m3", runtime="llamacpp-vulkan")
    e5_embedder = create_embedder("multilingual-e5-base", runtime="llamacpp-vulkan")
    bge_searcher = Searcher(str(ROOT / "bench" / "bge-m3.db"), embedder=bge_embedder)
    e5_searcher = Searcher(str(ROOT / "bench" / "multilingual-e5-base.db"), embedder=e5_embedder)

    per_system_unjudged = {"fts": [], "semantic_bge": [], "semantic_e5": []}
    for q in QUERIES:
        qrels = set(q["relevant"])
        fts_ids = [r["id"] for r in bge_searcher.fts(q["q"], k=10, lang="both")]
        bge_ids = [r["id"] for r in bge_searcher.semantic(q["q"], k=10)]
        e5_ids = [r["id"] for r in e5_searcher.semantic(q["q"], k=10)]

        for name, ids in (("fts", fts_ids), ("semantic_bge", bge_ids), ("semantic_e5", e5_ids)):
            unjudged = sum(1 for i in ids if i not in qrels)
            per_system_unjudged[name].append(unjudged / max(1, len(ids)))

    bge_embedder.close()
    e5_embedder.close()
    return {k: float(np.mean(v)) for k, v in per_system_unjudged.items()}


# --------------------------------------------------------------------------
# 4) Latency breakdown (embed vs matrix search), persistent connection
# --------------------------------------------------------------------------

def latency_breakdown(model_name: str, n=20):
    db_path = ROOT / "bench" / f"{model_name}.db"
    embedder = create_embedder(model_name, runtime="llamacpp-vulkan")
    searcher = Searcher(str(db_path), embedder=embedder)

    sample_queries = [q["q"] for q in QUERIES[:n]] if len(QUERIES) >= n else [q["q"] for q in QUERIES]

    embed_times = []
    search_times = []
    for q in sample_queries:
        t0 = time.time()
        qvec = embedder.embed_query(q)
        t1 = time.time()
        sim_en = searcher._en_mat @ qvec
        sim_ru = searcher._ru_mat @ qvec
        sim = np.maximum(sim_en, sim_ru)
        _ = np.argsort(-sim)[:10]
        t2 = time.time()
        embed_times.append((t1 - t0) * 1000)
        search_times.append((t2 - t1) * 1000)

    embedder.close()
    return {
        "embed_ms_avg": float(np.mean(embed_times)),
        "matrix_search_ms_avg": float(np.mean(search_times)),
        "n": len(sample_queries),
    }


# --------------------------------------------------------------------------
# 5) Pooling/prefix sanity check
# --------------------------------------------------------------------------

def pooling_sanity_check():
    out = {}
    for name in MODELS:
        spec = LLAMACPP_REGISTRY[name]
        out[name] = {
            "pooling": spec["pooling"],
            "prefix_query": spec["prefix_query"],
            "prefix_passage": spec["prefix_passage"],
        }
    return out


def main():
    report = {}

    print("=== pooling/prefix sanity check ===")
    report["pooling_sanity"] = pooling_sanity_check()
    print(json.dumps(report["pooling_sanity"], indent=2))

    for m in MODELS:
        print(f"=== cross-lingual pair retrieval: {m} ===", flush=True)
        report.setdefault("cross_lingual", {})[m] = cross_lingual_pair_eval(m, n_sample=200)
        print(json.dumps(report["cross_lingual"][m], indent=2))

    print("=== pool bias check (29 original queries) ===", flush=True)
    report["pool_bias_unjudged_fraction"] = pool_bias_check()
    print(json.dumps(report["pool_bias_unjudged_fraction"], indent=2))

    for m in MODELS:
        print(f"=== latency breakdown: {m} ===", flush=True)
        report.setdefault("latency", {})[m] = latency_breakdown(m)
        print(json.dumps(report["latency"][m], indent=2))

    report["paraphrase_queries"] = "SKIPPED: generating good paraphrase queries cheaply (without an LLM) is not feasible; noted rather than faked with naive keyword removal."

    out = ROOT / "bench" / "eval_objective_results.json"
    json.dump(report, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
