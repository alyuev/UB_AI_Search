"""A fake OpenAI-compatible HTTP server (httpx.MockTransport): no network, no real keys."""
from __future__ import annotations

import json

import httpx

OPENROUTER_MODELS = {"data": [
    {"id": "meta/llama-free:free", "name": "Llama Free", "context_length": 65536,
     "pricing": {"prompt": "0", "completion": "0"}, "supported_parameters": ["temperature"]},
    {"id": "acme/think-1", "name": "Think 1", "context_length": 200000,
     "pricing": {"prompt": "0.000003", "completion": "0.000015"}, "supported_parameters": ["reasoning", "temperature"]},
    {"id": "acme/embed-3", "name": "Embed", "context_length": 8000, "pricing": {"prompt": "0.0000001", "completion": "0"}},
]}


def sse_events(events: list, done: bool = True) -> bytes:
    """A raw SSE body from already prepared chunk dicts (reasoning deltas, error events, ...)."""
    lines = ["data: " + json.dumps(e) for e in events]
    if done:
        lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode()


def sse(text: str, usage: dict | None = None, finish: str = "stop", model: str | None = None) -> bytes:
    lines = []
    extra = {"model": model} if model else {}
    for i in range(0, len(text), 7):
        lines.append("data: " + json.dumps({**extra, "choices": [{"delta": {"content": text[i:i + 7]}}]}))
    lines.append("data: " + json.dumps({**extra, "choices": [{"delta": {}, "finish_reason": finish}]}))
    if usage is not None:
        lines.append("data: " + json.dumps({"choices": [], "usage": usage}))
    lines.append("data: [DONE]")
    return (": PROCESSING\n\n" + "\n\n".join(lines) + "\n\n").encode()


class FakeOpenAI:
    """`script` is a list of responses/exceptions consumed by successive chat requests; once empty, `reply` is used."""

    def __init__(self, reply="Ответ [147.2.2].", usage=None, models=None, native=None):
        self.reply = reply
        self.usage = usage if usage is not None else {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        self.models = models if models is not None else OPENROUTER_MODELS
        self.native = native or {}      # path -> json for /api/tags, /api/v0/models, /api/show, /props ...
        self.script: list = []
        self.requests: list[httpx.Request] = []
        self.chats: list[dict] = []
        self.model_requests = 0

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path in self.native:
            val = self.native[path]
            return val if isinstance(val, httpx.Response) else httpx.Response(200, json=val)
        if path.endswith("/chat/completions"):
            body = json.loads(request.content)
            self.chats.append({"body": body, "headers": dict(request.headers)})
            if self.script:
                item = self.script.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item
            return httpx.Response(200, content=sse(self.reply, self.usage), headers={"content-type": "text/event-stream"})
        if path in self.native:
            val = self.native[path]
            return val if isinstance(val, httpx.Response) else httpx.Response(200, json=val)
        if path.endswith("/key"):
            return httpx.Response(200, json={"data": {"label": "test"}})
        if path.endswith("/models"):
            self.model_requests += 1
            return httpx.Response(200, json=self.models)
        return httpx.Response(404, json={"error": {"message": "not found"}})

    def last_chat(self) -> dict:
        return self.chats[-1]
