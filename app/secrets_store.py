"""API keys of LLM providers: a separate file data/secrets.json, never settings.json / history / logs.

On Windows the keys are encrypted with DPAPI (CryptProtectData, bound to the current Windows user;
no extra dependencies). Where DPAPI is unavailable the keys are only base64-encoded ("plain") and the
UI shows a visible notice. Environment variables are a read-only fallback source.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from app import config

_force_plain = False  # tests: pretend DPAPI is unavailable

ENV_VARS = {"openrouter": "OPENROUTER_API_KEY", "huggingface": "HF_TOKEN", "openai": "OPENAI_API_KEY"}


# --------------------------------------------------------------- DPAPI ---

def dpapi_available() -> bool:
    return sys.platform == "win32" and not _force_plain


def _blob_type():
    import ctypes
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]
    return DATA_BLOB


def _dpapi(data: bytes, protect: bool) -> bytes:
    import ctypes
    DATA_BLOB = _blob_type()
    crypt32, kernel32 = ctypes.windll.crypt32, ctypes.windll.kernel32
    buf = ctypes.create_string_buffer(data, len(data))
    blob_in = DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    blob_out = DATA_BLOB()
    fn = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    ok = fn(ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out))
    if not ok:
        raise OSError("DPAPI: операция не удалась")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        kernel32.LocalFree(blob_out.pbData)


def _encrypt(plain: str) -> tuple[str, str]:
    raw = plain.encode("utf-8")
    if dpapi_available():
        try:
            return "dpapi", base64.b64encode(_dpapi(raw, True)).decode("ascii")
        except Exception:  # noqa: BLE001
            pass
    return "plain", base64.b64encode(raw).decode("ascii")


def _decrypt(method: str, blob: str) -> Optional[str]:
    try:
        raw = base64.b64decode(blob)
        if method == "dpapi":
            if sys.platform != "win32":
                return None
            raw = _dpapi(raw, False)
        return raw.decode("utf-8")
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- file ---

def _path() -> Path:
    return Path(config.DATA_DIR) / "secrets.json"


def _load() -> dict:
    try:
        d = json.loads(_path().read_text(encoding="utf-8"))
        if isinstance(d, dict) and isinstance(d.get("keys"), dict):
            return d
    except (OSError, ValueError):
        pass
    return {"version": 1, "keys": {}}


def _save(d: dict) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
    tmp.replace(p)


def set_key(provider_id: str, key: str) -> None:
    key = (key or "").strip()
    if not key:
        raise ValueError("Ключ пустой")
    if any(ch.isspace() for ch in key) or not key.isascii():
        raise ValueError("Ключ содержит пробелы или не-латинские символы — проверьте, что скопирован только ключ")
    method, blob = _encrypt(key)
    d = _load()
    d["keys"][provider_id] = {"method": method, "data": blob}
    _save(d)


def delete_key(provider_id: str) -> None:
    d = _load()
    if d["keys"].pop(provider_id, None) is not None:
        _save(d)


def _stored(provider_id: str) -> Optional[str]:
    e = _load()["keys"].get(provider_id)
    return _decrypt(e.get("method", "plain"), e.get("data", "")) if isinstance(e, dict) else None


def _env_key(kind: str, base_url: str) -> Optional[str]:
    """Env fallback only where the key belongs to that service (never send an OpenAI key elsewhere)."""
    host = (urlparse(base_url or "").hostname or "").lower()
    if kind in ("openrouter", "huggingface"):
        name = ENV_VARS[kind]
    elif kind == "openai_compatible" and host == "api.openai.com":
        name = ENV_VARS["openai"]
    else:
        return None
    return (os.environ.get(name) or "").strip() or None


def get_key(provider_id: str, kind: str = "", base_url: str = "") -> Optional[str]:
    return _stored(provider_id) or _env_key(kind, base_url)


def key_info(provider_id: str, kind: str = "", base_url: str = "") -> dict:
    """What the API may reveal about a key: never the key itself."""
    k = _stored(provider_id)
    src = "stored" if k else None
    if not k:
        k = _env_key(kind, base_url)
        src = "env" if k else "none"
    return {"has_key": bool(k), "last4": k[-4:] if k else None, "key_source": src}


def protection() -> dict:
    ok = dpapi_available()
    return {"method": "dpapi" if ok else "plain",
            "notice": None if ok else "Ключи хранятся без шифрования (шифрование Windows недоступно). "
                                      "Не копируйте папку data другим людям."}


_KEY_RE = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9._\-]{8,})")


def redact(text: str, *keys: Optional[str]) -> str:
    """Remove API keys (known ones and sk-… lookalikes) from any text before logging / showing it."""
    out = text or ""
    for k in keys:
        if k and len(k) >= 6:
            out = out.replace(k, "***")
    return _KEY_RE.sub("***", out)
