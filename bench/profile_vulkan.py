"""Profile llama.cpp (Vulkan backend) speed for all 4 GGUF models on ~300
real paragraphs, at several batch sizes. Also used to sanity-check quality
(the actual ONNX-vs-GGUF cosine check itself is done ad hoc; this script
only measures throughput of an already-verified server).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.embedder import LLAMACPP_REGISTRY, GGUF_DIR, LLAMACPP_BIN_DIR

SUBSET = json.load(open(ROOT / "bench" / "subset.json", encoding="utf-8"))
SAMPLE = [r["en"] for r in SUBSET[:300]]
BATCH_SIZES = [8, 16, 32, 64]

MODELS = ["multilingual-e5-small", "multilingual-e5-base", "bge-m3", "qwen3-embedding-0.6b"]


def start_server(model_name: str, port: int):
    spec = LLAMACPP_REGISTRY[model_name]
    gguf_path = GGUF_DIR / spec["gguf_file"]
    server_bin = LLAMACPP_BIN_DIR / "llama-server.exe"
    cmd = [
        str(server_bin), "-m", str(gguf_path),
        "--embedding", "--pooling", spec["pooling"],
        "-ngl", "99", "-ub", "2048", "-b", "2048",
        "--port", str(port), "--host", "127.0.0.1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(ROOT))
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server for {model_name} exited early")
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
            if r.status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"server for {model_name} did not become ready")


def main():
    results = {}
    port = 8801
    for model_name in MODELS:
        results[model_name] = {}
        print(f"=== {model_name} ===", flush=True)
        proc = start_server(model_name, port)
        try:
            with httpx.Client(timeout=180) as client:
                for bs in BATCH_SIZES:
                    t0 = time.time()
                    for i in range(0, len(SAMPLE), bs):
                        batch = SAMPLE[i : i + bs]
                        r = client.post(f"http://127.0.0.1:{port}/v1/embeddings", json={"input": batch})
                        r.raise_for_status()
                    dt = time.time() - t0
                    rate = len(SAMPLE) / dt
                    results[model_name][f"vulkan_bs{bs}"] = {"seconds": dt, "texts_per_sec": rate}
                    print(f"  bs={bs:3d}  {dt:6.2f}s  {rate:6.2f} texts/sec", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        port += 1

    out = ROOT / "bench" / "profile_vulkan_results.json"
    json.dump(results, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
