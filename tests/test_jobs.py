
import time

from app.jobs import JobManager


def test_job_lifecycle_success():
    mgr = JobManager()

    def worker(job):
        progress = mgr.make_progress_callback(job)
        for i in range(1, 4):
            progress("step", i, 3, None, f"step {i}")
        job.result = {"ok": True}

    job = mgr.create("demo", ["step"], worker)
    for _ in range(50):
        if mgr.get(job.job_id).status in ("done", "error"):
            break
        time.sleep(0.05)
    d = mgr.get(job.job_id).to_dict()
    assert d["status"] == "done"
    assert d["progress"] == 1.0
    assert d["result"] == {"ok": True}


def test_job_cancel():
    mgr = JobManager()

    def worker(job):
        cancel = mgr.cancel_event(job)
        progress = mgr.make_progress_callback(job)
        for i in range(1, 100):
            if cancel.is_set():
                return
            progress("step", i, 100, None, "")
            time.sleep(0.01)

    job = mgr.create("demo", ["step"], worker)
    time.sleep(0.05)
    mgr.cancel(job.job_id)
    for _ in range(100):
        if mgr.get(job.job_id).status in ("done", "error", "cancelled"):
            break
        time.sleep(0.05)
    assert mgr.get(job.job_id).status == "cancelled"


def test_job_error_captured():
    mgr = JobManager()

    def worker(job):
        raise ValueError("boom")

    job = mgr.create("demo", [], worker)
    for _ in range(50):
        if mgr.get(job.job_id).status in ("done", "error"):
            break
        time.sleep(0.05)
    d = mgr.get(job.job_id).to_dict()
    assert d["status"] == "error"
    assert "boom" in d["error"]
