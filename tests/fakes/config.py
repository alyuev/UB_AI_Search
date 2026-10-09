"""Fake app.config module matching the contract, for tests only."""
import os
import tempfile

ROOT = os.path.join(tempfile.gettempdir(), "smartsearch_fake_root")
MODELS_DIR = os.path.join(ROOT, "models")
LIBRARY_DIR = os.path.join(ROOT, "library")
LOGS_DIR = os.path.join(ROOT, "logs")
DATA_DIR = os.path.join(ROOT, "data")

for d in (ROOT, MODELS_DIR, LIBRARY_DIR, LOGS_DIR, DATA_DIR):
    os.makedirs(d, exist_ok=True)

_SETTINGS = {
    "lmstudio_url": "http://localhost:1234",
    "llm_model": None,
    "temperature": 0.2,
    "max_tokens": 1024,
    "llm_timeout": 120,
    "embedding_model": "multilingual-e5-small",
    "device": "cpu",
    "active_book": "urantia",
}


def load_settings():
    return dict(_SETTINGS)


def save_settings(d):
    _SETTINGS.update(d)
    return dict(_SETTINGS)
