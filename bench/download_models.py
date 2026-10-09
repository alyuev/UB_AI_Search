"""Download ONNX embedding models used by the app / benchmarks.

Run online, once: `python bench/download_models.py [model ...]`
With no args, downloads all models in the registry.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.embedder import MODEL_REGISTRY, download_model  # noqa: E402


def main():
    names = sys.argv[1:] or list(MODEL_REGISTRY.keys())
    for name in names:
        print(f"=== {name} ===")
        download_model(name)


if __name__ == "__main__":
    main()
