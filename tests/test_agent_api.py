"""Agent API: REST + MCP (search_book / ask_book / get_answer / get_paragraph), access rules, chat formatting."""
from app import agent_api


def rpc(api, method, params=None, mid=1, headers=None):
    msg = {"jsonrpc": "2.0", "method": method, **({"id": mid} if mid is not None else {}),
           **({"params": params} if params is not None else {})}
    return api.post("/mcp", json=msg, headers=headers or {})


def test_rest_search_returns_chat_text(api):
    r = api.get("/api/agent/search", params={"q": "любви", "mode": "fts"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["results"][0]["id"] == "147.2.2"
    assert "Иисус говорил о любви Отца" in d["text"] and "`147.2.2`" in d["text"]


def test_rest_ask_waits_for_answer(api):
    r = api.post("/api/agent/ask", json={"question": "Что говорил Иисус о любви?", "wait": 30})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["status"] == "done" and "любви Отца" in d["answer"]
    assert d["text"].startswith("❓ *Что говорил Иисус о любви?*")
    # an ordinary job: visible in the job list and in the history
    assert any(j["job_id"] == d["job_id"] for j in api.get("/api/jobs").json()["jobs"])
    assert api.get("/api/history").json()["items"][0]["query"] == "Что говорил Иисус о любви?"


def test_rest_ask_timeout_then_get_answer(api):
    d = api.post("/api/agent/ask", json={"question": "Что говорил Иисус о любви?", "wait": 0}).json()
    assert d["status"] in ("queued", "running", "done")
    d2 = api.get(f"/api/agent/answer/{d['job_id']}", params={"wait": 30}).json()
    assert d2["status"] == "done" and d2["question"] == "Что говорил Иисус о любви?"
    assert api.get("/api/agent/answer/nope").status_code == 404


def test_mcp_handshake_and_tools(api):
    r = rpc(api, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                "clientInfo": {"name": "hermes", "version": "1"}})
    res = r.json()["result"]
    assert res["protocolVersion"] == "2025-03-26" and "tools" in res["capabilities"]
    assert rpc(api, "notifications/initialized", mid=None).status_code == 202
    names = {t["name"] for t in rpc(api, "tools/list").json()["result"]["tools"]}
    assert names == {"ask_book", "search_book", "get_answer", "get_paragraph"}
    assert rpc(api, "nope").json()["error"]["code"] == -32601


def test_mcp_tool_calls(api):
    res = rpc(api, "tools/call", {"name": "search_book", "arguments": {"query": "Слово", "mode": "fts"}}).json()["result"]
    assert not res["isError"] and "В начале было Слово" in res["content"][0]["text"]
    assert "structuredContent" not in res   # only the ready text: no JSON copy for the agent's model to read

    res = rpc(api, "tools/call", {"name": "get_paragraph", "arguments": {"id": "147.2.2", "context": 1}}).json()["result"]
    assert all(f"`{i}`" in res["content"][0]["text"] for i in ("147.2.1", "147.2.2", "147.2.3"))

    res = rpc(api, "tools/call", {"name": "ask_book", "arguments": {"question": "О любви"}}).json()["result"]
    assert not res["isError"] and "любви Отца" in res["content"][0]["text"]

    res = rpc(api, "tools/call", {"name": "get_paragraph", "arguments": {"id": "999.9.9"}}).json()["result"]
    assert res["isError"] and "не найден" in res["content"][0]["text"]


def test_token_required_when_set(api, monkeypatch):
    monkeypatch.setenv("SMARTSEARCH_AGENT_TOKEN", "s3cret")
    assert rpc(api, "ping").status_code == 401
    assert rpc(api, "ping", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert rpc(api, "ping", headers={"Authorization": "Bearer s3cret"}).json()["result"] == {}
    assert api.get("/api/agent/search", params={"q": "x"}).status_code == 401


def test_outside_clients_reach_only_agent_api(api, monkeypatch):
    from fastapi.testclient import TestClient
    remote = TestClient(api.server.app, client=("192.168.1.50", 5000))
    assert remote.get("/api/settings").status_code == 403
    assert remote.get("/").status_code == 403
    assert remote.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}).status_code == 403  # no key set
    monkeypatch.setenv("SMARTSEARCH_AGENT_TOKEN", "s3cret")
    ok = remote.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
                     headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


def test_format_answer_lists_cited_sources_and_fits_telegram():
    src = {"id": "147:2.4", "paper": 147, "en": "It was evening.", "ru": "Был вечер в Иерусалиме.",
           "paper_title": {"ru": "Интерлюдия в Иерусалиме", "en": "x"}, "section_title": {"ru": "Вечер"}}
    text = agent_api.format_answer("Когда?", {"answer": "Вечером [147:2.4]. " + "слово " * 2000,
                                              "sources": [src], "cited_ids": ["147:2.4"]})
    assert len(text) <= agent_api.TEXT_LIMIT
    assert "📚 *Источники*" in text and "Документ 147. Интерлюдия в Иерусалиме · Вечер" in text
