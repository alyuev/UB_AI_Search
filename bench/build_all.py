import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.core.indexer import build_index

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "urantia_bilingual_rag.json"

models = ["multilingual-e5-small", "multilingual-e5-base", "bge-m3"]
results = {}
for m in models:
    t0 = time.time()
    def prog(stage, cur, total, cid, msg, m=m):
        if cur % 4000 == 0 or cur == total:
            print(f"[{m}] {stage} {cur}/{total} {msg}", flush=True)
    db_path = ROOT / "bench" / f"{m}.db"
    build_index(str(JSON_PATH), str(db_path), m, progress=prog)
    elapsed = time.time() - t0
    size = db_path.stat().st_size
    results[m] = {"seconds": elapsed, "db_bytes": size}
    print(f"=== {m} done in {elapsed:.1f}s, db size {size/1e6:.1f} MB ===", flush=True)

json.dump(results, open(ROOT / "bench" / "build_times.json", "w"), indent=2)
print("ALL DONE")
