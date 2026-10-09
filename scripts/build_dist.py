"""Build the distributable "Bilingual Text Analyzer" (docs/00_architecture_report.md, section 11).

    .venv\\Scripts\\python scripts\\build_dist.py --mode user            # app + prebuilt DB + runtime
    .venv\\Scripts\\python scripts\\build_dist.py --mode builder         # + source corpus & DB-building tools
    .venv\\Scripts\\python scripts\\build_dist.py --mode both --no-zip

Output (default --out dist):
    dist/<mode>/BilingualTextAnalyzer/        the folder (run start.bat inside)
    dist/BilingualTextAnalyzer-<mode>.zip     the same, zipped (top-level folder BilingualTextAnalyzer/)
    dist/sizes_<mode>.json                    size breakdown (used by docs/06_packaging.md)

What goes in:
  * python/            official python.org *embeddable* CPython (same minor version as .venv), with
                       site-packages copied OFFLINE from .venv using each distribution's RECORD
                       (exactly the versions that were tested): only the runtime dependency closure
                       of the packages the app imports -- no pytest/pip/setuptools/huggingface_hub/
                       sympy/mpmath/tqdm/yaml.
  * app/, main.py, start.bat, README.md, docs/new_book_howto.md
  * library/<book_id>/book.db      prebuilt database
  * models/gguf/<model>.gguf + models/<model>/tokenizer.json (+config.json)   (Vulkan runtime)
  * tools/llama.cpp/bin/           llama-server + its DLLs only (no bench/cli/quantize tools)
  The ONNX weights (hundreds of MB) are NOT shipped: if Vulkan is unavailable on a machine, the
  semantic part of search is disabled (full-text search keeps working) until the ONNX model is
  added -- documented in README.md / docs/06_packaging.md.
  builder mode additionally: urantia_bilingual_rag.json/.md, data/titles/, tools/extract_epub_titles.py,
  tools/convert_vectors.py.
Excluded always: .venv, tests/, bench/, logs/, other models, scratch DBs, the EPUB sources, __pycache__,
data/secrets.json (API keys of LLM providers), data/settings.json, data/history.db, .env files.

The embeddable zip is cached in dist/_cache (downloaded once from python.org if missing).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DIST_NAME = "BilingualTextAnalyzer"

# Top-level distributions the app imports at runtime (directly or lazily).
RUNTIME_TOP = [
    "fastapi", "uvicorn", "httpx", "numpy", "pymorphy3", "pymorphy3-dicts-ru", "psutil",
    "tokenizers",
]
# Dependencies that are installed in .venv but are NOT needed at runtime for this app
# (verified by the dist smoke test): hub client (offline), progress bars, symbolic math that
# onnxruntime only imports in its optional transformers tools, test tooling, packaging tools.
RUNTIME_EXCLUDE = {
    "huggingface-hub", "hf-xet", "tqdm", "pyyaml", "fsspec", "filelock", "sympy", "mpmath",
    "pytest", "pluggy", "iniconfig", "pygments", "setuptools", "pip", "wheel", "packaging",
    # onnxruntime(-directml): ~67 MB, useless in dist since ONNX weights are not shipped;
    # embeddings run via llama.cpp/Vulkan. Kept in .venv for benchmarks/quality reference.
    "onnxruntime", "onnxruntime-directml", "protobuf", "flatbuffers", "coloredlogs", "humanfriendly",
}

LLAMA_KEEP_EXE = {"llama-server.exe"}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# ---------------------------------------------------------------- python + packages ---

def fetch_embeddable_python(out: Path) -> Path:
    ver = "%d.%d.%d" % sys.version_info[:3]
    name = f"python-{ver}-embed-amd64.zip"
    cache = out / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    zpath = cache / name
    if not zpath.exists():
        url = f"https://www.python.org/ftp/python/{ver}/{name}"
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"corrupt embeddable zip ({bad}); delete {zpath} and retry")
    return zpath


def find_secret_files(folder: Path) -> list[Path]:
    """Files that must never ship: API keys (secrets.json anywhere, .env in the root) and the personal
    data/settings.json + data/history.db. Used as a guard before zipping."""
    bad = []
    for f in folder.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(folder).parts
        name = f.name.lower()
        if name == "secrets.json" or (len(rel) == 1 and name.startswith(".env")) or                 (len(rel) == 2 and rel[0] == "data" and name in ("settings.json", "history.db", "prompt_presets.json")):
            bad.append(f)
    return bad


def runtime_closure() -> list[metadata.Distribution]:
    """Resolve the dependency closure of RUNTIME_TOP from what is installed in the running env."""
    from packaging.requirements import Requirement

    seen: dict[str, metadata.Distribution] = {}
    stack = [norm(n) for n in RUNTIME_TOP]
    while stack:
        n = stack.pop()
        if n in seen or n in RUNTIME_EXCLUDE:
            continue
        try:
            dist = metadata.distribution(n)
        except metadata.PackageNotFoundError:
            print(f"  warning: {n} not installed in this environment, skipped")
            continue
        seen[n] = dist
        for r in dist.requires or []:
            req = Requirement(r)
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            stack.append(norm(req.name))
    return sorted(seen.values(), key=lambda d: norm(d.metadata["Name"]))


def copy_site_packages(dest: Path) -> list[str]:
    """Copy each distribution's files (per RECORD) into dest; returns 'name==version' lines."""
    dest.mkdir(parents=True, exist_ok=True)
    pins = []
    for dist in runtime_closure():
        name, ver = dist.metadata["Name"], dist.version
        pins.append(f"{name}=={ver}")
        base = Path(str(dist.locate_file("")))
        for pp in dist.files or []:
            parts = pp.parts
            if parts and parts[0] == "..":
                continue  # console-script .exe etc. living outside site-packages
            if "__pycache__" in parts or pp.suffix == ".pyc" or "tests" in parts[:-1]:
                continue
            src = base / pp
            if not src.is_file():
                continue
            tgt = dest / pp
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, tgt)
    return pins


def setup_python(target: Path, zpath: Path) -> list[str]:
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(target)
    # The embeddable distribution ignores PYTHONPATH and (being in isolated mode) does not add the
    # script directory; add the app root ("..") and site-packages explicitly.
    pth = next(target.glob("python*._pth"))
    stem = pth.stem  # python314._pth -> "python314"
    pth.write_text(
        f"{stem}.zip\n.\n..\nLib\\site-packages\nimport site\n", encoding="utf-8"
    )
    return copy_site_packages(target / "Lib" / "site-packages")


# ---------------------------------------------------------------- app files ---

def copy_tree(src: Path, dst: Path, skip_dirs=("__pycache__",)):
    for f in src.rglob("*"):
        if f.is_dir() or any(p in skip_dirs for p in f.parts) or f.suffix == ".pyc":
            continue
        t = dst / f.relative_to(src)
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, t)


def copy_file(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(f"required file missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def llama_files(bin_dir: Path) -> list[Path]:
    keep = []
    for f in sorted(bin_dir.iterdir()):
        if not f.is_file():
            continue
        n = f.name.lower()
        if f.suffix.lower() == ".exe":
            if n in LLAMA_KEEP_EXE:
                keep.append(f)
        elif f.suffix.lower() == ".dll":
            # other tools' private "<tool>-impl.dll" and the RPC backend are not needed by llama-server
            if n.endswith("-impl.dll") and n != "llama-server-impl.dll":
                continue
            if n.startswith("ggml-rpc"):
                continue
            keep.append(f)
        else:
            keep.append(f)  # licenses
    return keep


START_BAT = r"""@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "HF_HUB_OFFLINE=1"
title Bilingual Text Analyzer

set "PY="
if exist "python\python.exe" set "PY=python\python.exe"
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
    echo [ОШИБКА] Не найден python\python.exe рядом с start.bat.
    echo Распакуйте архив целиком в обычную папку и запускайте start.bat из неё.
    pause
    exit /b 1
)
if not exist "app\server.py" (
    echo [ОШИБКА] Не найдена папка app. Архив распакован не полностью.
    pause
    exit /b 1
)

set "HASDB="
for /d %%D in ("library\*") do if exist "%%~D\book.db" set "HASDB=1"
if not defined HASDB (
    echo [ВНИМАНИЕ] В папке library нет готовой базы книги.
    echo Откройте вкладку "База данных" и постройте её из исходного файла.
)
if not exist "tools\llama.cpp\bin\llama-server.exe" (
    echo [ВНИМАНИЕ] Нет tools\llama.cpp\bin\llama-server.exe: поиск по смыслу недоступен,
    echo работает только полнотекстовый поиск.
)

echo Запуск... окно браузера откроется автоматически. Чтобы выйти, закройте это окно.
"%PY%" main.py %*
if errorlevel 1 (
    echo.
    echo Программа завершилась с ошибкой. Подробности в папке logs.
    pause
)
"""


def write_bat(path: Path):
    path.write_bytes(START_BAT.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))


# ---------------------------------------------------------------- build ---

def build(mode: str, out: Path, book_id: str, zpath: Path, make_zip: bool, model: str | None, runtime: str | None):
    from app import config
    from app.core import embedder as emb

    settings = config.load_settings()
    model = model or settings.get("embedding_model", "bge-m3")
    runtime = runtime or settings.get("embedding_runtime", "llamacpp-vulkan")
    folder = out / mode / DIST_NAME
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    sizes: dict[str, int] = {}
    t0 = time.time()
    print(f"=== build {mode}: {folder}")

    # python + packages
    pins = setup_python(folder / "python", zpath)
    (folder / "requirements-runtime.txt").write_text("\n".join(pins) + "\n", encoding="utf-8")
    sizes["python_total"] = dir_size(folder / "python")
    sizes["python_site_packages"] = dir_size(folder / "python" / "Lib" / "site-packages")
    sizes["python_interpreter"] = sizes["python_total"] - sizes["python_site_packages"]

    # app
    copy_tree(ROOT / "app", folder / "app")
    copy_file(ROOT / "main.py", folder / "main.py")
    write_bat(folder / "start.bat")
    copy_file(ROOT / "README.md", folder / "README.md")
    copy_file(ROOT / "docs" / "new_book_howto.md", folder / "docs" / "new_book_howto.md")
    sizes["app_code"] = dir_size(folder / "app")

    # library
    db = config.LIBRARY_DIR / book_id / "book.db"
    copy_file(db, folder / "library" / book_id / "book.db")
    sizes["book_db"] = db.stat().st_size

    # embedding model files (Vulkan runtime) -- GGUF + tokenizer
    entry = emb.LLAMACPP_REGISTRY.get(model)
    if entry is None:
        raise SystemExit(f"model {model!r} has no GGUF entry; cannot build the Vulkan package")
    gguf = emb.GGUF_DIR / entry["gguf_file"]
    copy_file(gguf, folder / "models" / "gguf" / gguf.name)
    for fn in ("tokenizer.json", "config.json"):
        src = emb.MODELS_DIR / model / fn
        if src.exists():
            copy_file(src, folder / "models" / model / fn)
    sizes["model_gguf"] = gguf.stat().st_size
    sizes["model_tokenizer_cfg"] = dir_size(folder / "models" / model)

    # llama.cpp
    for f in llama_files(emb.LLAMACPP_BIN_DIR):
        copy_file(f, folder / "tools" / "llama.cpp" / "bin" / f.name)
    sizes["llama_cpp_bin"] = dir_size(folder / "tools" / "llama.cpp")
    sizes["llama_cpp_bin_full_original"] = dir_size(emb.LLAMACPP_BIN_DIR)

    (folder / "data").mkdir(exist_ok=True)

    if mode == "builder":
        for name in ("urantia_bilingual_rag.json", "urantia_bilingual_rag.md"):
            copy_file(ROOT / name, folder / name)
        copy_tree(ROOT / "data" / "titles", folder / "data" / "titles")
        copy_file(ROOT / "tools" / "extract_epub_titles.py", folder / "tools" / "extract_epub_titles.py")
        copy_file(ROOT / "tools" / "convert_vectors.py", folder / "tools" / "convert_vectors.py")
        sizes["source_json"] = (folder / "urantia_bilingual_rag.json").stat().st_size
        sizes["source_md"] = (folder / "urantia_bilingual_rag.md").stat().st_size

    leaked = find_secret_files(folder)
    if leaked:
        raise SystemExit("refusing to package secrets / personal data: " + ", ".join(str(f) for f in leaked))
    sizes["folder_total"] = dir_size(folder)
    if make_zip:
        zp = out / f"{DIST_NAME}-{mode}.zip"
        zp.unlink(missing_ok=True)
        print(f"zipping -> {zp} ...")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for f in sorted(folder.rglob("*")):
                if f.is_file():
                    zf.write(f, Path(DIST_NAME) / f.relative_to(folder))
        sizes["zip"] = zp.stat().st_size
    sizes["build_seconds"] = round(time.time() - t0)
    (out / f"sizes_{mode}.json").write_text(json.dumps(sizes, indent=1), encoding="utf-8")
    print(f"=== {mode} done in {sizes['build_seconds']}s")
    for k, v in sizes.items():
        print(f"  {k:<30} {human(v) if k != 'build_seconds' else str(v) + ' s'}")
    return folder, sizes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["user", "builder", "both"], default="user")
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    ap.add_argument("--book-id", default="urantia")
    ap.add_argument("--model", default=None, help="default: embedding_model from settings.json")
    ap.add_argument("--runtime", default=None)
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    zpath = fetch_embeddable_python(args.out)
    modes = ["user", "builder"] if args.mode == "both" else [args.mode]
    for m in modes:
        build(m, args.out, args.book_id, zpath, not args.no_zip, args.model, args.runtime)


if __name__ == "__main__":
    main()
