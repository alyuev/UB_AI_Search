"""Profile CPU vs DirectML execution providers, and batch sizes, for each
embedding model on ~300 real paragraphs, to pick the fastest "auto" path.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.embedder import Embedder, MODEL_REGISTRY

SUBSET = json.load(open(ROOT / "bench" / "subset.json", encoding="utf-8"))
SAMPLE = [r["en"] for r in SUBSET[:300]]

import os

MODELS = ["multilingual-e5-small", "multilingual-e5-base", "bge-m3", "qwen3-embedding-0.6b"]
BATCH_SIZES = [8, 16, 32]
PHYSICAL_CORES = (os.cpu_count() or 24) // 2
# (device, intra_op_threads label) pairs to profile
DEVICE_CONFIGS = [
    ("cpu", PHYSICAL_CORES, f"cpu_all{PHYSICAL_CORES}"),
    ("cpu", max(1, PHYSICAL_CORES - 2), f"cpu_minus2_{max(1, PHYSICAL_CORES-2)}"),
    ("gpu", None, "gpu_dml"),
]


def main():
    results = {}
    for model_name in MODELS:
        results[model_name] = {}
        for device, threads, label in DEVICE_CONFIGS:
            try:
                embedder = Embedder(model_name, device=device, intra_op_threads=threads)
            except Exception as e:
                results[model_name][label] = {"error": str(e)}
                continue
            device = label  # reuse loop body below, keyed by our label
            for bs in BATCH_SIZES:
                t0 = time.time()
                embedder.embed_passages(SAMPLE, batch_size=bs)
                dt = time.time() - t0
                rate = len(SAMPLE) / dt
                key = f"{device}_bs{bs}"
                results[model_name][key] = {
                    "provider": embedder.provider,
                    "seconds": dt,
                    "texts_per_sec": rate,
                }
                print(f"{model_name:28s} {key:10s} provider={embedder.provider:22s} "
                      f"{dt:6.2f}s  {rate:6.2f} texts/sec", flush=True)

    out = ROOT / "bench" / "profile_results.json"
    json.dump(results, open(out, "w"), indent=2)
    print("wrote", out)

    # pick fastest device per model (best texts/sec across its batch sizes)
    best = {}
    for model_name, dev_results in results.items():
        best_key, best_rate = None, -1
        for key, r in dev_results.items():
            if "texts_per_sec" in r and r["texts_per_sec"] > best_rate:
                best_rate = r["texts_per_sec"]
                best_key = key
        best[model_name] = {"best_key": best_key, "texts_per_sec": best_rate}
    print(json.dumps(best, indent=2))
    json.dump(best, open(ROOT / "bench" / "profile_best.json", "w"), indent=2)


if __name__ == "__main__":
    main()
