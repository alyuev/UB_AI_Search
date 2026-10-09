"""Fake app.core.embedder for tests."""
from __future__ import annotations

import hashlib
import struct


def list_available_models():
    return [
        {"name": "multilingual-e5-small", "dim": 384, "max_tokens": 512,
         "downloaded": True, "path": "/models/e5-small", "description": "fast default"},
        {"name": "multilingual-e5-base", "dim": 768, "max_tokens": 512,
         "downloaded": False, "path": "/models/e5-base", "description": "balanced"},
        {"name": "BGE-M3", "dim": 1024, "max_tokens": 8192,
         "downloaded": False, "path": "/models/bge-m3", "description": "best quality"},
    ]


class Embedder:
    def __init__(self, model_name: str = "multilingual-e5-small", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.dim = 384
        self.status = "ready"

    def _vec(self, text: str):
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vals = struct.unpack("64f", (h * 4)[:256])
        return list(vals[: self.dim]) + [0.0] * max(0, self.dim - len(vals))

    def embed_query(self, text: str):
        return self._vec(text or "")
