"""Global paths and persisted settings for SmartSearch."""
from __future__ import annotations

import json
from pathlib import Path

# ROOT is the project directory (parent of the `app` package), resolved
# robustly regardless of the current working directory, spaces or cyrillic
# characters in the path.
ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = ROOT / "models"
LIBRARY_DIR = ROOT / "library"
LOGS_DIR = ROOT / "logs"
DATA_DIR = ROOT / "data"

SETTINGS_PATH = DATA_DIR / "settings.json"

DEFAULT_SETTINGS = {
    "lmstudio_url": "http://localhost:1234",
    "llm_model": "",  # model of the ACTIVE provider
    # LLM providers (profiles WITHOUT api keys; keys live in data/secrets.json). None = not materialised yet:
    # the old lmstudio_url / llm_model then act as the profile "LM Studio (локально)".
    "llm_providers": None,
    "active_llm_provider": "",
    "allow_cloud_llm": False,  # offline by default; cloud providers are unusable until explicitly allowed
    "llm_provider_rag": "", "llm_provider_translation": "", "llm_provider_rewrite": "",   # "" = the active one
    "llm_model_rag": "", "llm_model_translation": "", "llm_model_rewrite": "",            # "" = provider's model
    "openrouter_referer": "", "openrouter_title": "",
    "temperature": 0.2,
    "max_tokens": 4000,
    "llm_timeout": 300,
    "embedding_model": "bge-m3",
    "embedding_runtime": "llamacpp-vulkan",
    "device": "auto",
    "active_book": "urantia",
    "fts_mode_hybrid": "or",  # "and" | "or" | "and_fallback_or"
    "fts_stopwords": True,  # drop stop words from unquoted FTS terms
    "fts_mode_standalone": "or",  # quoted phrases stay exact; AND+fallback lost recall on Paper 0 eval
    # bm25 column weights for FTS5 search (by column name; missing = 1.0).
    "fts_weights": {"en": 1.0, "ru": 1.0, "ru_lemma": 0.5, "title_en": 2.0, "title_ru": 2.0},
}


def _ensure_dirs() -> None:
    for d in (MODELS_DIR, LIBRARY_DIR, LOGS_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_settings() -> dict:
    """Load settings from data/settings.json, creating defaults if missing."""
    _ensure_dirs()
    if not SETTINGS_PATH.exists():
        save_settings(dict(DEFAULT_SETTINGS))
        return dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        data = {}
    merged = dict(DEFAULT_SETTINGS)
    merged.update(data or {})
    return merged


def save_settings(settings: dict) -> None:
    _ensure_dirs()
    merged = dict(DEFAULT_SETTINGS)
    merged.update(settings or {})
    tmp = SETTINGS_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    tmp.replace(SETTINGS_PATH)
