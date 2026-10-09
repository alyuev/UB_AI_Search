import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.core.indexer import build_index

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "urantia_bilingual_rag.json"
DB_PATH = ROOT / "library" / "urantia" / "book.db"

def prog(stage, cur, total, cid, msg):
    if total and (cur % 2000 == 0 or cur == total):
        print(f"{stage} {cur}/{total} {msg}", flush=True)

t0 = time.time()
build_index(
    str(JSON_PATH), str(DB_PATH), "bge-m3", scheme="D",
    book_id="urantia", book_name="The Urantia Book", progress=prog,
    runtime="llamacpp-vulkan", lower_priority=True, embed_batch_size=16,
)
elapsed = time.time() - t0
size = DB_PATH.stat().st_size
print(f"=== FINAL BUILD done in {elapsed:.1f}s, db size {size/1e6:.1f} MB ===")
json.dump({"seconds": elapsed, "db_bytes": size}, open(ROOT/"bench"/"final_build_stats.json","w"), indent=2)
print("ALL DONE")
