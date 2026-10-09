"""In-process background job manager for long-running tasks (indexing, RAG, translation analysis).

No external queue: jobs run in daemon threads, progress kept in memory, and can be observed via
polling (GET /api/jobs/{id}) or SSE (GET /api/jobs/{id}/events).
"""
from __future__ import annotations

import itertools
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    from app.log import get_logger
except ImportError:  # pragma: no cover - fallback for isolated testing
    from tests.fakes.log import get_logger

logger = get_logger("app")

_STATUS_QUEUED = "queued"
_STATUS_RUNNING = "running"
_STATUS_DONE = "done"
_STATUS_ERROR = "error"
_STATUS_CANCELLED = "cancelled"


@dataclass
class Stage:
    name: str
    state: str = "pending"  # pending|active|done


@dataclass
class Job:
    job_id: str
    type: str
    stages: list = field(default_factory=list)
    status: str = _STATUS_QUEUED
    stage: Optional[str] = None
    progress: float = 0.0
    current: int = 0
    total: int = 0
    current_id: Optional[int] = None
    message: str = ""
    started_at: Optional[float] = None
    elapsed: float = 0.0
    eta: Optional[float] = None
    rate: Optional[float] = None  # items/sec, from the same rate window used for eta
    result: Optional[dict] = None
    error: Optional[str] = None
    log: list = field(default_factory=list)
    _cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _rate_window: list = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        if self.status == _STATUS_RUNNING and self.started_at:
            self.elapsed = time.time() - self.started_at
        return {
            "job_id": self.job_id,
            "type": self.type,
            "status": self.status,
            "stage": self.stage,
            "stages": [{"name": s.name, "state": s.state} for s in self.stages],
            "progress": round(self.progress, 4),
            "current": self.current,
            "total": self.total,
            "current_id": self.current_id,
            "message": self.message,
            "elapsed": round(self.elapsed, 1),
            "eta": round(self.eta, 1) if self.eta is not None else None,
            "rate": round(self.rate, 1) if self.rate is not None else None,
            "result": self.result,
            "error": self.error,
            "log": self.log[-200:],
        }

    def add_log(self, line: str):
        self.log.append(f"[{time.strftime('%H:%M:%S')}] {line}")


class JobManager:
    """Registry + runner for background jobs. One thread per job, cooperative cancellation."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._subscribers: dict[str, list] = {}

    def create(self, job_type: str, stage_names: list[str], worker: Callable[[Job], None]) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id=job_id, type=job_type,
                   stages=[Stage(name=n) for n in stage_names])
        with self._lock:
            self._jobs[job_id] = job

        def run():
            job.status = _STATUS_RUNNING
            job.started_at = time.time()
            job.add_log(f"job {job_type} started")
            try:
                worker(job)
                if job._cancel_event.is_set():
                    job.status = _STATUS_CANCELLED
                    job.add_log("job cancelled")
                else:
                    job.status = _STATUS_DONE
                    job.progress = 1.0
                    for s in job.stages:
                        s.state = "done"
                    job.add_log("job finished")
            except Exception as exc:  # noqa: BLE001
                job.status = _STATUS_ERROR
                job.error = str(exc)
                job.add_log(f"job error: {exc}")
                logger.exception("job %s failed", job_id)
            finally:
                self._notify(job_id)

        threading.Thread(target=run, daemon=True, name=f"job-{job_id}").start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.started_at or 0, reverse=True)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        job._cancel_event.set()
        job.add_log("cancel requested")
        return True

    def make_progress_callback(self, job: Job):
        """Build a progress(stage, current, total, current_id, message) callback for core workers."""

        def progress(stage: str, current: int, total: int, current_id=None, message: str = ""):
            now = time.time()
            job._rate_window.append((now, current))
            job._rate_window[:] = job._rate_window[-20:]
            for s in job.stages:
                if s.name == stage:
                    s.state = "active"
                elif s.state == "active":
                    s.state = "done"
            job.stage = stage
            job.current = current
            job.total = total
            job.current_id = current_id
            job.message = message
            job.progress = (current / total) if total else 0.0
            if len(job._rate_window) >= 2:
                t0, c0 = job._rate_window[0]
                t1, c1 = job._rate_window[-1]
                dt, dc = t1 - t0, c1 - c0
                if dt > 0 and dc > 0 and total:
                    rate = dc / dt
                    remaining = max(0, total - current)
                    job.eta = remaining / rate
            job.add_log(f"{stage}: {current}/{total} {message}".strip())
            self._notify(job.job_id)

        return progress

    def cancel_event(self, job: Job) -> threading.Event:
        return job._cancel_event

    # --- SSE support -------------------------------------------------
    def subscribe(self, job_id: str):
        import queue
        q = queue.Queue()
        with self._lock:
            self._subscribers.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q):
        with self._lock:
            subs = self._subscribers.get(job_id, [])
            if q in subs:
                subs.remove(q)

    def _notify(self, job_id: str):
        job = self._jobs.get(job_id)
        if not job:
            return
        with self._lock:
            subs = list(self._subscribers.get(job_id, []))
        for q in subs:
            try:
                q.put_nowait(job.to_dict())
            except Exception:  # noqa: BLE001
                pass


manager = JobManager()
