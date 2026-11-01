"""API for AI agents (Hermes, Claude, etc.): search and questions over the book in one call.

Two ways in, same three tools:
  * REST  — GET /api/agent/search, POST /api/agent/ask, GET /api/agent/answer/{job_id}
  * MCP   — POST /mcp (Streamable HTTP, JSON responses; tools search_book, ask_book, get_answer, get_paragraph)

Every reply carries `text`: ready Markdown for a chat (Telegram), so the agent can forward it as is.
A question runs as an ordinary RAG job: it shows in the UI and lands in the history like one asked on the page.

Access: from this computer (127.0.0.1) no key is needed unless one is set; from other machines a key is
mandatory (SMARTSEARCH_AGENT_TOKEN, or «agent_token» in data/settings.json), sent as
`Authorization: Bearer <key>`.
"""
from __future__ import annotations

import hmac
import json
import os
import re
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from app import config
from app.version import __version__

router = APIRouter()

MCP_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
ASK_WAIT_DEFAULT = 240      # seconds ask_book waits before handing back a job id to poll
ASK_WAIT_MAX = 900
TEXT_LIMIT = 3800           # a Telegram message holds 4096 characters
QUOTE_LEN = 220

_LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}


# ------------------------------------------------------------------ access ---

def agent_token() -> str:
    return (os.environ.get("SMARTSEARCH_AGENT_TOKEN") or config.load_settings().get("agent_token") or "").strip()


def _check_access(request: Request) -> None:
    token = agent_token()
    host = request.client.host if request.client else ""
    if not token:
        if host in _LOOPBACK:
            return
        raise HTTPException(403, "Доступ агентов с других компьютеров закрыт: задайте ключ SMARTSEARCH_AGENT_TOKEN")
    auth = request.headers.get("authorization", "")
    given = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
    if not hmac.compare_digest(given.encode(), token.encode()):
        raise HTTPException(401, "Неверный или отсутствующий ключ (Authorization: Bearer …)")


# -------------------------------------------------------------- formatting ---

def _srv():
    from app import server   # late: app.server includes this router at import time
    return server


def _clean(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _quote(text: Optional[str], n: int = QUOTE_LEN) -> str:
    t = _clean(text)
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "…"


def _where(p: dict) -> str:
    """«Документ 147. Название · Раздел 2. Название» from the paragraph's titles (Russian first)."""
    parts = []
    pt, st = p.get("paper_title") or {}, p.get("section_title") or {}
    title = pt.get("ru") or pt.get("en")
    parts.append(f"Документ {p.get('paper')}" + (f". {_clean(title)}" if title else ""))
    sec = st.get("ru") or st.get("en")
    if sec:
        parts.append(_clean(sec))
    return " · ".join(parts)


def _para_out(p: dict) -> dict:
    return {"id": p["id"], "where": _where(p), "ru": _clean(p.get("ru")), "en": _clean(p.get("en"))}


def _fit(text: str) -> str:
    return text if len(text) <= TEXT_LIMIT else text[:TEXT_LIMIT].rsplit("\n", 1)[0] + "\n…"


def format_search(query: str, results: list[dict]) -> str:
    if not results:
        return f"🔎 *{query}*\n\nВ книге ничего не найдено."
    lines = [f"🔎 *{query}* — найдено абзацев: {len(results)}", ""]
    for i, p in enumerate(results, 1):
        lines.append(f"{i}. `{p['id']}` — _{_where(p)}_")
        lines.append(f"   {_quote(p.get('ru') or p.get('en'))}")
        lines.append("")
    return _fit("\n".join(lines).rstrip())


def format_answer(question: str, result: dict) -> str:
    answer = (result.get("answer") or "").strip()
    by_id = {p["id"]: p for p in result.get("sources") or []}
    cited = [c for c in result.get("cited_ids") or [] if c in by_id]
    head = f"❓ *{question}*\n\n"
    tail = ""
    if cited:
        refs = [f"• `{c}` — _{_where(by_id[c])}_\n  «{_quote(by_id[c].get('ru') or by_id[c].get('en'), 160)}»"
                for c in dict.fromkeys(cited)]
        tail = "\n\n📚 *Источники*\n" + "\n".join(refs)
    if result.get("truncated"):
        tail += "\n\n⚠️ Ответ оборван на середине."
    body = answer or "Модель не дала ответа."
    room = TEXT_LIMIT - len(head) - len(tail)
    if len(body) > room:      # keep the sources visible, shorten the answer instead
        body = body[:max(room, 200)].rsplit(" ", 1)[0] + "…"
    return _fit(head + body + tail)


# ----------------------------------------------------------------- actions ---

def do_search(query: str, k: int = 5, mode: str = "hybrid") -> dict:
    query = (query or "").strip()
    if not query:
        raise HTTPException(400, "Пустой запрос")
    k = max(1, min(int(k or 5), 30))
    data = _srv().api_search(q=query, mode=mode, k=k, lang=None, history=1)
    results = data["results"]
    return {"query": query, "results": [_para_out(p) for p in results], "text": format_search(query, results)}


def _job_reply(job, question: str) -> dict:
    d = {"job_id": job.job_id, "status": job.status, "question": question}
    if job.status == "done" or (job.status == "error" and (job.result or {}).get("answer")):
        r = job.result or {}
        by_id = {p["id"]: p for p in r.get("sources") or []}
        d.update({"answer": r.get("answer"), "no_answer": bool(r.get("no_answer")),
                  "sources": [_para_out(by_id[c]) for c in dict.fromkeys(r.get("cited_ids") or []) if c in by_id],
                  "model": r.get("answered_model"), "text": format_answer(question, r)})
        if job.status == "error":
            d["error"] = job.error
    elif job.status == "error":
        d.update({"error": job.error, "text": f"⚠️ Не удалось ответить: {job.error}"})
    elif job.status == "cancelled":
        d["text"] = "Вопрос отменён."
    else:
        d["text"] = (f"⏳ Ответ ещё готовится ({job.stage or 'в очереди'}, {job.to_dict()['elapsed']:.0f} с). "
                     f"Запросите его снова через get_answer с job_id={job.job_id}.")
    return d


_QUESTIONS: dict[str, str] = {}   # job_id -> question, for get_answer


def _wait(job, wait: float) -> None:
    end = time.monotonic() + wait
    while job.status in ("queued", "running") and time.monotonic() < end:
        time.sleep(0.5)


def do_ask(question: str, wait: float = ASK_WAIT_DEFAULT, overview: Optional[bool] = None,
           k: Optional[int] = None) -> dict:
    question = (question or "").strip()
    if not question:
        raise HTTPException(400, "Пустой вопрос")
    srv = _srv()
    started = srv.api_job_rag(srv.RagJobRequest(question=question, overview=overview, k=k))
    job = srv.job_manager.get(started["job_id"])
    _QUESTIONS[job.job_id] = question
    _wait(job, max(0.0, min(float(wait), ASK_WAIT_MAX)))
    return _job_reply(job, question)


def do_get_answer(job_id: str, wait: float = 60) -> dict:
    job = _srv().job_manager.get(job_id)
    if job is None or job.type != "rag":
        raise HTTPException(404, "Задача не найдена (после перезапуска программы старые ответы не хранятся)")
    _wait(job, max(0.0, min(float(wait), ASK_WAIT_MAX)))
    return _job_reply(job, _QUESTIONS.get(job_id, ""))


def do_get_paragraph(para_id: str, context: int = 0) -> dict:
    srv = _srv()
    if context:
        n = max(0, min(int(context), 5))
        paras = srv.api_paragraph_context(para_id, before=n, after=n)["context"]
    else:
        paras = [srv.api_paragraph(para_id)]
    lines = []
    for p in paras:
        lines += [f"`{p['id']}` — _{_where(p)}_", _clean(p.get("ru")), f"_{_clean(p.get('en'))}_", ""]
    return {"paragraphs": [_para_out(p) for p in paras], "text": _fit("\n".join(lines).rstrip())}


# -------------------------------------------------------------------- REST ---

class AskRequest(BaseModel):
    question: str
    wait: float = ASK_WAIT_DEFAULT
    overview: Optional[bool] = None
    k: Optional[int] = None


@router.get("/api/agent/search")
def rest_search(request: Request, q: str, k: int = 5, mode: str = "hybrid"):
    _check_access(request)
    return do_search(q, k, mode)


@router.post("/api/agent/ask")
def rest_ask(request: Request, req: AskRequest):
    _check_access(request)
    return do_ask(req.question, req.wait, req.overview, req.k)


@router.get("/api/agent/answer/{job_id}")
def rest_answer(request: Request, job_id: str, wait: float = 60):
    _check_access(request)
    return do_get_answer(job_id, wait)


# --------------------------------------------------------------------- MCP ---

_FORWARD = ("В ответе есть поле text — готовый Markdown для чата: перешли его пользователю как есть, "
            "не пересказывай и не добавляй ничего от себя.")

TOOLS = [
    {"name": "ask_book",
     "description": ("Задать вопрос по «Книге Урантии» (двуязычная база EN/RU). Программа сама найдёт абзацы и "
                     "ответит строго по ним, со ссылками вида [147:2.4]. Вопрос можно задавать по-русски или "
                     "по-английски. Ответ занимает от 10 секунд до нескольких минут. Если ответ не успел — "
                     "придёт job_id, тогда вызови get_answer. " + _FORWARD),
     "inputSchema": {"type": "object", "properties": {
         "question": {"type": "string", "description": "Вопрос пользователя дословно"},
         "overview": {"type": "boolean",
                      "description": "true — обзор по всей книге (вопросы «перечисли все…», «где упоминается…»); "
                                     "не указывай, программа решит сама"}},
         "required": ["question"]}},
    {"name": "search_book",
     "description": ("Найти абзацы «Книги Урантии» по словам и смыслу, без ответа модели (быстро, ~1 с). "
                     "Подходит для «найди, где говорится о…», «цитата про…». " + _FORWARD),
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "Что искать (русский или английский)"},
         "k": {"type": "integer", "minimum": 1, "maximum": 30, "default": 5, "description": "Сколько абзацев"},
         "mode": {"type": "string", "enum": ["hybrid", "fts", "semantic"], "default": "hybrid",
                  "description": "hybrid — слова+смысл; fts — точные слова (фраза в кавычках); semantic — смысл"}},
         "required": ["query"]}},
    {"name": "get_answer",
     "description": "Получить ответ на вопрос, который ask_book не успел дождаться (по job_id). " + _FORWARD,
     "inputSchema": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}},
    {"name": "get_paragraph",
     "description": ("Полный текст абзаца по номеру (например 147:2.4 = Документ 147, раздел 2, абзац 4) "
                     "на русском и английском, при желании с соседними абзацами. " + _FORWARD),
     "inputSchema": {"type": "object", "properties": {
         "id": {"type": "string", "description": "Номер абзаца, например 147:2.4"},
         "context": {"type": "integer", "minimum": 0, "maximum": 5, "default": 0,
                     "description": "Сколько соседних абзацев показать до и после"}},
         "required": ["id"]}},
]


def _call_tool(name: str, args: dict) -> dict:
    if name == "ask_book":
        return do_ask(args.get("question", ""), overview=args.get("overview"))
    if name == "search_book":
        return do_search(args.get("query", ""), args.get("k", 5), args.get("mode", "hybrid"))
    if name == "get_answer":
        return do_get_answer(args.get("job_id", ""))
    if name == "get_paragraph":
        return do_get_paragraph(args.get("id", ""), args.get("context", 0))
    raise KeyError(name)


def _rpc_result(mid, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _rpc_error(mid, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def _handle_rpc(msg: dict) -> Optional[dict]:
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
        return _rpc_error(msg.get("id") if isinstance(msg, dict) else None, -32600, "Invalid Request")
    method, mid, params = msg["method"], msg.get("id"), msg.get("params") or {}
    if "id" not in msg:          # notification (notifications/initialized, cancelled…): nothing to answer
        return None
    if method == "initialize":
        asked = params.get("protocolVersion")
        return _rpc_result(mid, {
            "protocolVersion": asked if asked in MCP_PROTOCOL_VERSIONS else MCP_PROTOCOL_VERSIONS[0],
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "smartsearch", "title": "SmartSearch — Книга Урантии", "version": __version__},
            "instructions": ("Поиск и ответы по двуязычной «Книге Урантии». Для вопросов — ask_book, для поиска "
                             "цитат — search_book. Поле text в результате — готовое оформленное сообщение: "
                             "передавай его пользователю без изменений.")})
    if method == "ping":
        return _rpc_result(mid, {})
    if method == "tools/list":
        return _rpc_result(mid, {"tools": TOOLS})
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        try:
            out = _call_tool(name, args)
        except KeyError:
            return _rpc_error(mid, -32602, f"Unknown tool: {name}")
        except HTTPException as exc:
            return _rpc_result(mid, {"content": [{"type": "text", "text": f"⚠️ {exc.detail}"}], "isError": True})
        except Exception as exc:  # noqa: BLE001
            return _rpc_result(mid, {"content": [{"type": "text", "text": f"⚠️ Ошибка: {exc}"}], "isError": True})
        data = {k: v for k, v in out.items() if k != "text"}
        return _rpc_result(mid, {"content": [{"type": "text", "text": out.get("text", "")}],
                                 "structuredContent": data, "isError": False})
    return _rpc_error(mid, -32601, f"Method not found: {method}")


@router.post("/mcp")
async def mcp_post(request: Request):
    _check_access(request)
    try:
        body = json.loads(await request.body())
    except ValueError:
        return JSONResponse(_rpc_error(None, -32700, "Parse error"), status_code=400)
    from starlette.concurrency import run_in_threadpool   # tools block (ask_book waits for the model)
    if isinstance(body, list):
        replies = [r for r in [await run_in_threadpool(_handle_rpc, m) for m in body] if r is not None]
        return JSONResponse(replies) if replies else Response(status_code=202)
    reply = await run_in_threadpool(_handle_rpc, body)
    return JSONResponse(reply) if reply is not None else Response(status_code=202)


@router.get("/mcp")
def mcp_get(request: Request):
    _check_access(request)
    return Response(status_code=405, headers={"Allow": "POST"})   # no server-initiated stream


@router.delete("/mcp")
def mcp_delete(request: Request):
    _check_access(request)
    return Response(status_code=405, headers={"Allow": "POST"})   # stateless: no sessions to end
