"""Build a scratch DB with ONLY Paper 0 (Foreword). Never touches library/."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.core.indexer import build_index

SRC = ROOT / "urantia_bilingual_rag.json"
OUT_JSON = Path(__file__).parent / "paper0.json"
DB = Path(__file__).parent / "paper0.db"

if __name__ == "__main__":
    data = [r for r in json.load(open(SRC, encoding="utf-8")) if r["paper"] == 0]
    json.dump(data, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False)
    print(len(data), "paragraphs")
    build_index(OUT_JSON, DB, "bge-m3", book_id="urantia", runtime="llamacpp-vulkan")
    print("built", DB)
