"""Entry point: find a free port (default start 8765), start uvicorn, open the browser.

Options:
  --port N        start looking for a free port at N (env SMARTSEARCH_PORT works too)
  --no-browser    do not open the browser (env SMARTSEARCH_NO_BROWSER=1 works too)
  --desktop       show the UI in its own app window (pywebview / WebView2) instead of the browser;
                  closing the window stops the program (env SMARTSEARCH_DESKTOP=1 works too)
  --host ADDR     listen on ADDR, e.g. 0.0.0.0 so AI agents on other machines (WSL, LAN) can reach
                  /mcp and /api/agent/* (env SMARTSEARCH_HOST); needs SMARTSEARCH_AGENT_TOKEN
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


HOST = "127.0.0.1"


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((HOST, port))
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
                s.bind((HOST, port))
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


def _redirect_missing_streams():
    """Under pythonw.exe there is no console: stdout/stderr are None and uvicorn's logging would fail."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    os.makedirs("logs", exist_ok=True)
    f = open(os.path.join("logs", "console.log"), "a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = f
    if sys.stderr is None:
        sys.stderr = f


def _webview2_available() -> bool:
    """True when pywebview can show the window with Edge WebView2 (Windows 10/11 normally have it)."""
    if os.environ.get("SMARTSEARCH_FORCE_BROWSER"):
        return False
    try:
        if sys.platform != "win32":
            import webview  # noqa: F401  (GTK/Qt/Cocoa: let pywebview decide)
            return True
        import webview.platforms.winforms as wf
        return getattr(wf, "renderer", "") == "edgechromium"
    except Exception as exc:  # pythonnet/.NET missing, broken install...
        print(f"pywebview unavailable: {exc!r}", flush=True)
        return False


def _message_box(text: str, title: str = "Bilingual Text Analyzer"):
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, title, 0x40 | 0x40000)  # MB_ICONINFORMATION | MB_TOPMOST
    else:
        print(text, flush=True)


def _signal_ready():
    """Tell SmartSearch.exe that the window is up, so it closes its «запускается…» splash."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenEventW(0x0002, False, r"Local\SmartSearchReady")   # EVENT_MODIFY_STATE
        if h:
            k.SetEvent(h)
            k.CloseHandle(h)
    except Exception:
        pass


def run_browser_fallback(server, url: str):
    """No WebView2: serve as usual, open the browser, and keep a small dialog as the «quit» button."""
    t = threading.Thread(target=server.run, daemon=True, name="uvicorn")
    t.start()
    while not server.started and t.is_alive():
        time.sleep(0.1)
    if not server.started:
        _message_box("Программа не смогла запуститься. Подробности в папке logs.")
        return
    if not os.environ.get("SMARTSEARCH_NO_BROWSER"):
        webbrowser.open(url)
    _signal_ready()

    def dialog():
        _message_box("В этой Windows нет компонента Microsoft Edge WebView2, поэтому программа открыта "
                     f"в браузере:\n\n{url}\n\nЕго можно установить бесплатно: "
                     "https://go.microsoft.com/fwlink/p/?LinkId=2124703\n\n"
                     "Нажмите «ОК», чтобы завершить работу программы.")
        server.should_exit = True

    threading.Thread(target=dialog, daemon=True).start()
    t.join()


def _set_window_icon(window):
    ico = os.path.abspath(os.path.join("app", "static", "icon.ico"))
    if sys.platform != "win32" or not os.path.exists(ico):
        return
    try:
        from System.Drawing import Icon  # pythonnet, loaded by pywebview's WinForms backend
        form = window.native
        form.Invoke(__import__("System").Action(lambda: setattr(form, "Icon", Icon(ico))))
    except Exception as exc:
        print(f"window icon not set: {exc!r}", flush=True)


def run_desktop(server, url: str):
    """Server in a background thread, the UI in a native window on the main thread."""
    if not _webview2_available():
        return run_browser_fallback(server, url)
    import webview

    webview.settings["ALLOW_DOWNLOADS"] = True           # «Экспорт в Markdown», prompt presets
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True

    t = threading.Thread(target=server.run, daemon=True, name="uvicorn")
    t.start()
    while not server.started and t.is_alive():
        time.sleep(0.1)
    if not server.started:
        return  # server failed to start; the error is already logged

    window = webview.create_window("Bilingual Text Analyzer", url, width=1400, height=900,
                                   min_size=(800, 500), text_select=True)

    def watch_server():
        # restart from the settings page stops uvicorn: close the window, main() spawns the new process
        t.join()
        try:
            window.destroy()
        except Exception:
            pass

    threading.Thread(target=watch_server, daemon=True).start()
    window.events.shown += lambda: _set_window_icon(window)
    window.events.loaded += _signal_ready
    webview.start(private_mode=False, storage_path=os.path.abspath(os.path.join("data", "webview")))
    server.should_exit = True   # window closed by the user
    t.join(timeout=15)


def main():
    global HOST
    desktop = "--desktop" in sys.argv or bool(os.environ.get("SMARTSEARCH_DESKTOP"))
    if desktop:
        _redirect_missing_streams()
    HOST = _arg_value("--host") or os.environ.get("SMARTSEARCH_HOST") or "127.0.0.1"
    if HOST not in ("127.0.0.1", "localhost", "::1"):
        from app.agent_api import agent_token
        if not agent_token():
            print("Для --host нужен ключ доступа агентов: задайте переменную SMARTSEARCH_AGENT_TOKEN "
                  "(или «agent_token» в data/settings.json).")
            sys.exit(4)
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

    if not desktop and "--no-browser" not in sys.argv and not os.environ.get("SMARTSEARCH_NO_BROWSER"):
        threading.Thread(target=open_browser, daemon=True).start()
    print(f"SmartSearch запускается: {url}", flush=True)
    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=port, log_level="info"))

    def print_banner():
        # after uvicorn's own startup lines, so the address is the last thing visible in the window
        while not server.started and not server.should_exit:
            time.sleep(0.2)
        if server.started:
            line = "=" * 60
            print("", line, "  SmartSearch работает.", f"  Откройте в браузере:  {url}",
                  f"  Для ИИ-агентов (MCP):  http://{'<адрес этого компьютера>' if HOST != '127.0.0.1' else '127.0.0.1'}:{port}/mcp",
                  "  Чтобы остановить программу, закройте это окно.", line, "",
                  sep="\n", flush=True)

    app.state.uvicorn_server = server   # POST /api/restart stops it and sets restart_requested
    if desktop:
        run_desktop(server, url)
    else:
        threading.Thread(target=print_banner, daemon=True).start()
        server.run()
    if getattr(app.state, "restart_requested", False):
        print("Перезапуск…")
        argv = [a for a in sys.argv]
        if HOST != "127.0.0.1" and "--host" not in argv:
            argv += ["--host", HOST]
        if "--port" not in argv:
            argv += ["--port", str(port)]
        env = dict(os.environ, SMARTSEARCH_RESTARTED="1", SMARTSEARCH_NO_BROWSER="1")
        subprocess.Popen([sys.executable] + argv, env=env, cwd=os.getcwd())


if __name__ == "__main__":
    main()
