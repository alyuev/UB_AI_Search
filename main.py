"""Entry point: find a free port (default start 8765), start uvicorn, open the browser.

Options:
  --port N        start looking for a free port at N (env SMARTSEARCH_PORT works too)
  --no-browser    do not open the browser (env SMARTSEARCH_NO_BROWSER=1 works too)
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser

# Fully offline at runtime: models are local files, nothing may be fetched from the Hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTHONUTF8", "1")

import uvicorn  # noqa: E402


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def find_free_port(start: int = 8765, attempts: int = 50) -> int:
    if os.environ.pop("SMARTSEARCH_RESTARTED", None):
        # restarted from the settings page: wait for the old process to free its port, keep the same address
        for _ in range(40):
            if _port_free(start):
                return start
            time.sleep(0.5)
    port = start
    for _ in range(attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    raise RuntimeError(
        f"Не удалось найти свободный порт в диапазоне {start}-{start + attempts - 1}. "
        f"Закройте лишние программы или запустите с другим номером: main.py --port 9000"
    )


def _arg_value(name: str):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def main():
    try:
        start = int(_arg_value("--port") or os.environ.get("SMARTSEARCH_PORT") or 8765)
    except ValueError:
        print("Некорректный номер порта (--port).")
        sys.exit(2)
    try:
        port = find_free_port(start)
    except RuntimeError as exc:
        print(exc)
        sys.exit(3)

    from app.server import app  # imported here so config/log side effects run after cwd setup

    url = f"http://127.0.0.1:{port}"

    def open_browser():
        time.sleep(1.0)
        webbrowser.open(url)

    if "--no-browser" not in sys.argv and not os.environ.get("SMARTSEARCH_NO_BROWSER"):
        threading.Thread(target=open_browser, daemon=True).start()
    print(f"SmartSearch запущен: {url}")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info"))
    app.state.uvicorn_server = server   # POST /api/restart stops it and sets restart_requested
    server.run()
    if getattr(app.state, "restart_requested", False):
        print("Перезапуск…")
        argv = [a for a in sys.argv]
        if "--port" not in argv:
            argv += ["--port", str(port)]
        env = dict(os.environ, SMARTSEARCH_RESTARTED="1", SMARTSEARCH_NO_BROWSER="1")
        subprocess.Popen([sys.executable] + argv, env=env, cwd=os.getcwd())


if __name__ == "__main__":
    main()
