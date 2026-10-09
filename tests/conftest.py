"""Test bootstrap: makes `app.config`, `app.log`, `app.core.*` resolve to the fakes
under tests/fakes when the real core modules (built by the parallel agent) are not
yet present. Once the real app.config/app.core exist, this becomes a no-op.
"""
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _install_fake(module_name: str, fake_path: str):
    try:
        importlib.import_module(module_name)
        return  # real module present, don't shadow it
    except ImportError:
        pass
    fake = importlib.import_module(fake_path)
    sys.modules[module_name] = fake


_install_fake("app.config", "tests.fakes.config")
_install_fake("app.log", "tests.fakes.log")
_install_fake("app.core", "tests.fakes.core")
_install_fake("app.core.db", "tests.fakes.core.db")
_install_fake("app.core.embedder", "tests.fakes.core.embedder")
_install_fake("app.core.indexer", "tests.fakes.core.indexer")
_install_fake("app.core.search", "tests.fakes.core.search")


import pytest  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _isolate_app_paths(tmp_path_factory):
    """Session-wide safety net: redirect every path the app might WRITE to
    (settings.json, the library/ directory) into a tmp dir for the whole test
    run, regardless of what any individual test module does.

    Without this, a real app.config module's save_settings()/book.db writes
    land in the real project's data/settings.json and library/ -- which has
    already happened once (a test run corrupted the live app's active_book
    while `main.py` was running against the real data). Individual fixtures
    (e.g. tests/test_server.py's `client`) may still additionally repoint
    config.LIBRARY_DIR etc. to their own tmp_path; that's fine, this fixture
    just ensures nothing ever falls back to the real paths by omission.

    MODELS_DIR / ROOT are deliberately left untouched: tests read (never
    write) already-downloaded embedding models from there.
    """
    try:
        from app import config
    except ImportError:
        return  # fakes-only environment (real app.config not present yet)

    tmp_root = tmp_path_factory.mktemp("app_data_isolated")
    tmp_data_dir = tmp_root / "data"
    tmp_library_dir = tmp_root / "library"
    tmp_logs_dir = tmp_root / "logs"
    for d in (tmp_data_dir, tmp_library_dir, tmp_logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    config.DATA_DIR = tmp_data_dir
    config.LIBRARY_DIR = tmp_library_dir
    config.LOGS_DIR = tmp_logs_dir
    if hasattr(config, "SETTINGS_PATH"):
        config.SETTINGS_PATH = tmp_data_dir / "settings.json"


@pytest.fixture
def api(tmp_path, monkeypatch):
    """TestClient on app.server with EVERYTHING isolated in tmp_path (settings, history.db, presets,
    library) and a fake LM Studio client (`api.llm`). No embedder/llama-server, no real model."""
    import importlib

    from fastapi.testclient import TestClient

    from app import config
    from tests.helpers import create_fake_book_db
    from tests.server_helpers import ServerFakeLLM

    data, lib = tmp_path / "data", tmp_path / "library"
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "SETTINGS_PATH", data / "settings.json")
    monkeypatch.setattr(config, "LIBRARY_DIR", lib)
    (lib / "test_book").mkdir(parents=True)
    create_fake_book_db(str(lib / "test_book" / "book.db"), book_id="test_book")
    config.save_settings({**config.load_settings(), "active_book": "test_book", "llm_model": None})

    server = importlib.import_module("app.server")

    def _no_embedder(*a, **k):
        raise RuntimeError("no embedder in tests")

    monkeypatch.setattr(server.core_embedder, "create_embedder", _no_embedder)
    server.state.embedder, server.state.embedder_key = None, None
    fake = ServerFakeLLM()
    monkeypatch.setattr(server, "get_llm_client", lambda: fake)
    server.state.load_book("test_book")
    client = TestClient(server.app)  # no `with`: startup hooks (real active book) are not run
    client.llm, client.server = fake, server
    yield client
    server.state.shutdown()
    server.state.conn, server.state.book_id = None, None


@pytest.fixture(autouse=True)
def _clean_provider_state(monkeypatch):
    """No real keys from the environment, no cached model lists / fake transport leaking between tests."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from app.llm import providers
    providers.clear_model_cache()
    monkeypatch.setattr(providers, "_TEST_TRANSPORT", None)
    monkeypatch.setattr(providers, "_sleep", lambda s: None)
    yield
    providers.clear_model_cache()
