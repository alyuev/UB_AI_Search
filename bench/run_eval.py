"""Evaluate FTS / semantic / hybrid search across models and schemes against
bench/queries.json, reporting Recall@10, MRR@10 and latency.

Also evaluates embedding schemes A (RU-only), B (EN-only), C (EN+RU
concatenated) vs D (separate EN/RU, max) — A/B reuse the embeddings already
stored for each model; C is computed in-memory only, for the single
best-performing model, to save time (per the "subset" benchmark plan).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.embedder import create_embedder
from app.core.search import Searcher

QUERIES = json.load(open(ROOT / "bench" / "queries.json", encoding="utf-8"))
# Only the two models that passed the ONNX-vs-GGUF quality gate (>=0.98 cosine)
# are benchmarked; both run via the llamacpp-vulkan runtime (fast, correct
# GPU compute-queue usage). e5-small and qwen3 were dropped — see
# app.core.embedder.LLAMACPP_REGISTRY[...]["not_recommended_reason"].
MODELS = ["bge-m3", "multilingual-e5-base"]
RUNTIME = "llamacpp-vulkan"


def recall_at_k(relevant, retrieved, k=10):
    top = set(retrieved[:k])
    rel = set(relevant)
    if not rel:
        return None
    return len(top & rel) / len(rel)


def mrr_at_k(relevant, retrieved, k=10):
    rel = set(relevant)
    for i, pid in enumerate(retrieved[:k]):
        if pid in rel:
            return 1.0 / (i + 1)
    return 0.0


def eval_ids_list(get_ids_fn, k=10):
    recalls, mrrs, lat = [], [], []
    for q in QUERIES:
        t0 = time.time()
        ids = get_ids_fn(q)
        dt = (time.time() - t0) * 1000
        recalls.append(recall_at_k(q["relevant"], ids, k) or 0.0)
        mrrs.append(mrr_at_k(q["relevant"], ids, k))
        lat.append(dt)
    n = len(QUERIES)
    return {
        "recall@10": sum(recalls) / n,
        "mrr@10": sum(mrrs) / n,
        "latency_ms_avg": sum(lat) / n,
    }


def eval_method(searcher, method, k=10, **kwargs):
    def get_ids(q):
        if method == "fts":
            res = searcher.fts(q["q"], k=k, **kwargs)
        elif method == "semantic":
            res = searcher.semantic(q["q"], k=k, **kwargs)
        else:
            res = searcher.hybrid(q["q"], k=k, **kwargs)
        return [r["id"] for r in res]

    return eval_ids_list(get_ids, k=k)


# --------------------------------------------------------------------------
# Scheme A/B/C evaluation (operate directly on the loaded en/ru matrices)
# --------------------------------------------------------------------------

def eval_scheme(searcher: Searcher, embedder, scheme: str, k=10):
    """scheme in {"A","B","D"}: A=ru-only, B=en-only, D=max(en,ru) (current).
    Computed directly against the matrices Searcher already loaded.
    """
    def get_ids(q):
        qvec = embedder.embed_query(q["q"])
        if scheme == "A":
            sim = searcher._ru_mat @ qvec
        elif scheme == "B":
            sim = searcher._en_mat @ qvec
        else:
            sim = np.maximum(searcher._en_mat @ qvec, searcher._ru_mat @ qvec)
        top_idx = np.argsort(-sim)[:k]
        return [searcher._ids[i] for i in top_idx]

    return eval_ids_list(get_ids, k=k)


def eval_scheme_c(db_path, embedder, k=10):
    """Scheme C: embed EN+RU concatenated per paragraph (in-memory only,
    not persisted to the DB) and evaluate semantic recall/MRR/latency.
    """
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, en, ru FROM paragraphs ORDER BY seq").fetchall()
    ids = [r["id"] for r in rows]
    # e5-base's native context is only 512 tokens; a few of our paragraphs'
    # EN+RU concatenation exceeds that in tokens (llama-server then rejects
    # the request with 400 rather than silently truncating). Character-cap
    # generously below the token budget — this is scheme C's real practical
    # limitation with a 512-context model, noted in docs/03_benchmark_search.md.
    concat_texts = [f"{r['en']}\n{r['ru']}"[:1400] for r in rows]

    t0 = time.time()
    mat = embedder.embed_passages(concat_texts, batch_size=16)
    embed_time = time.time() - t0

    def get_ids(q):
        qvec = embedder.embed_query(q["q"])
        sim = mat @ qvec
        top_idx = np.argsort(-sim)[:k]
        return [ids[i] for i in top_idx]

    result = eval_ids_list(get_ids, k=k)
    result["embed_all_passages_seconds"] = embed_time
    return result


def main():
    report = {}
    for m in MODELS:
        db_path = ROOT / "bench" / f"{m}.db"
        if not db_path.exists():
            print(f"skip {m}: {db_path} missing")
            continue
        embedder = create_embedder(m, runtime=RUNTIME)
        searcher = Searcher(str(db_path), embedder=embedder)
        report[m] = {
            "fts": eval_method(searcher, "fts", lang="both"),
            "semantic_scheme_D": eval_method(searcher, "semantic"),
            "scheme_A_ru_only": eval_scheme(searcher, embedder, "A"),
            "scheme_B_en_only": eval_scheme(searcher, embedder, "B"),
            "hybrid_rrf60_cand50": eval_method(searcher, "hybrid", rrf_k=60, cand=50),
            "hybrid_rrf30_cand30": eval_method(searcher, "hybrid", rrf_k=30, cand=30),
        }
        print(m, json.dumps(report[m], indent=2))

    # pick the best model by semantic scheme D recall@10 (fallback: hybrid)
    def score(m):
        r = report[m]
        return (r["semantic_scheme_D"]["recall@10"], r["hybrid_rrf60_cand50"]["recall@10"])

    best_model = max(report, key=score)
    print(f"\nBest model by semantic recall@10: {best_model}")

    # scheme C only for the best model, to save time
    best_db = ROOT / "bench" / f"{best_model}.db"
    best_embedder = create_embedder(best_model, runtime=RUNTIME)
    print(f"Computing scheme C (EN+RU concatenated) for {best_model} ...")
    report[best_model]["scheme_C_concat"] = eval_scheme_c(best_db, best_embedder)
    report["_best_model"] = best_model

    out = ROOT / "bench" / "eval_results.json"
    json.dump(report, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
