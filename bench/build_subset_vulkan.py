import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.core.indexer import build_index

ROOT = Path(__file__).resolve().parent.parent
SUBSET_JSON = ROOT / "bench" / "subset.json"
n_paragraphs = len(json.load(open(SUBSET_JSON, encoding="utf-8")))

models = ["bge-m3", "multilingual-e5-base"]
results = {}
for m in models:
    t0 = time.time()
    def prog(stage, cur, total, cid, msg, m=m):
        if total and (cur % 500 == 0 or cur == total):
            print(f"[{m}] {stage} {cur}/{total} {msg}", flush=True)
    db_path = ROOT / "bench" / f"{m}.db"
    build_index(str(SUBSET_JSON), str(db_path), m, progress=prog, book_id="urantia_subset",
                runtime="llamacpp-vulkan")
    elapsed = time.time() - t0
    size = db_path.stat().st_size
    results[m] = {
        "seconds": elapsed, "db_bytes": size, "paragraphs": n_paragraphs,
        "paras_per_sec": n_paragraphs / elapsed,
        "extrapolated_full_corpus_seconds": elapsed * (14585 / n_paragraphs),
    }
    print(f"=== {m} done in {elapsed:.1f}s, {n_paragraphs/elapsed:.1f} paras/sec, "
          f"db size {size/1e6:.1f} MB ===", flush=True)

json.dump(results, open(ROOT / "bench" / "build_times_vulkan.json", "w"), indent=2)
print("ALL DONE")
