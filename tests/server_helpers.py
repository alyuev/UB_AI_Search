"""Helpers for server-level tests: a fake LM Studio client (no network, no real model) and a job runner."""
from __future__ import annotations

import json
import time

from app.llm.lmstudio import GenerationStats

PLAN = json.dumps({"keywords_ru": ["любовь"], "keywords_en": ["love", "Jesus"],
                   "semantic_query": "Иисус говорил о любви Отца"})
ANSWER = "Иисус говорил о любви Отца [147.2.2]."
NO_ISSUES = "[]"
ONE_ISSUE = json.dumps([{"id": "147.2.2", "severity": "major", "category": "omissions", "problem": "потеряно слово",
                         "en_quote": "Jesus spoke", "ru_quote": "", "why_significant": "x", "suggestion": "y"}])


def fake_stats() -> GenerationStats:
    return GenerationStats(model="fake-model", elapsed_s=1.0, ttft_s=0.1, prompt_tokens=100, completion_tokens=20,
                           reasoning_tokens=0, total_tokens=120, tokens_per_sec=20.0, stop_reason="stop")


class ServerFakeLLM:
    """Stands in for LMStudioClient. Answers by message shape:
    search plan -> chat(); translation review -> `review`; verification (max_tokens=300) -> confirmed;
    everything else (a RAG question) -> `answer`. Every call is recorded in `calls`."""

    def __init__(self, plan=PLAN, answer=ANSWER, review=NO_ISSUES, context=16000):
        self.plan, self.answer, self.review, self.context = plan, answer, review, context
        self.calls: list[dict] = []

    def resolve_model(self, configured):
        return {"model": configured or "fake-model", "context": self.context, "note": None}

    def chat(self, messages, model, temperature=0.2, max_tokens=1024, timeout=None, stream=False,
             reasoning_effort=None):
        self.calls.append({"kind": "plan", "messages": messages, "model": model, "temperature": temperature,
                           "max_tokens": max_tokens, "reasoning_effort": reasoning_effort})
        return self.plan

    def chat_with_stats(self, messages, model, temperature=0.2, max_tokens=1024, timeout=None,
                        reasoning_effort=None, on_delta=None):
        user = messages[-1]["content"]
        if max_tokens == 300:
            kind, out = "verify", '{"verdict": "confirmed", "reason": "ok"}'
        elif "### ID" in user:
            kind, out = "review", self.review
        else:
            kind, out = "answer", self.answer
        self.calls.append({"kind": kind, "messages": messages, "model": model, "temperature": temperature,
                           "max_tokens": max_tokens, "reasoning_effort": reasoning_effort})
        return out, fake_stats()

    def of(self, kind: str) -> list[dict]:
        return [c for c in self.calls if c["kind"] == kind]


def wait_job(client, job_id: str, timeout: float = 20.0) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def run_rag(client, question="Что говорил Иисус о любви?", **extra) -> dict:
    r = client.post("/api/jobs/rag", json={"question": question, **extra})
    assert r.status_code == 200, r.text
    return wait_job(client, r.json()["job_id"])


def run_translation(client, reference="147:2", **extra) -> dict:
    r = client.post("/api/jobs/translation-analysis", json={"reference": reference, **extra})
    assert r.status_code == 200, r.text
    return wait_job(client, r.json()["job_id"])


def last_history_id(client) -> int:
    return client.get("/api/history").json()["items"][0]["id"]
