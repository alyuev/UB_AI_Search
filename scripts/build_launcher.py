"""Compile SmartSearch.exe (scripts/launcher/SmartSearch.cs) with the C# compiler built into Windows.

Run: .venv\\Scripts\\python scripts\\build_launcher.py [output.exe]   (default: SmartSearch.exe in the repo root)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts" / "launcher" / "SmartSearch.cs"
ICON = ROOT / "app" / "static" / "icon.ico"


def find_csc() -> Path:
    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    for fw in ("Framework64", "Framework"):
        p = windir / "Microsoft.NET" / fw / "v4.0.30319" / "csc.exe"
        if p.exists():
            return p
    raise RuntimeError("csc.exe (.NET Framework 4) not found")


def build(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(find_csc()), "/nologo", "/target:winexe", "/optimize+", "/platform:anycpu",
           f"/win32icon:{ICON}", "/r:System.Windows.Forms.dll", f"/out:{out}", str(SRC)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="cp866", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"csc failed:\n{r.stdout}\n{r.stderr}")
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "SmartSearch.exe"
    print(f"built {build(target)} ({target.stat().st_size} bytes)")
