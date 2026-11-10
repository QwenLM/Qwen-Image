"""JobStore 状态机单测：用一个假 pool 顶替 GPU，覆盖全部终态。

真实 pool 的 spawn + CUDA 加载无法在没有显卡的机器上跑，
但状态机的正确性完全可以在这里保证 —— 它只依赖
submit / wait_started / discard 三个原语。
"""
import threading
from concurrent.futures import Future

import pytest

from blobs import BlobStore
from errors import QueueFull
from jobs import JobStore
from protocol import Task


class FakePool:
    """可编排的假 pool。行为由 mode 决定。"""

    def __init__(self, mode="ok", delay=0.0):
        self.mode = mode
        self.delay = delay
        self.discarded = []
        self.cancelled = threading.Event()

    def submit(self, task):
        if self.mode == "queue_full":
            raise QueueFull("full")
        task.task_id = "t1"
        fut = Future()

        def run():
            if self.mode == "never_started":
                return
            if self.delay:
                threading.Event().wait(self.delay)
            if self.mode == "fail":
                from protocol import Result

                fut.set_result(Result(ok=False, task_id="t1", error="CUDA OOM"))
            elif self.mode == "hang":
                return  # 永不完成，触发 infer timeout
            else:
                from protocol import Result

                fut.set_result(Result(ok=True, task_id="t1", images=[b"PNG1", b"PNG2"]))

        threading.Thread(target=run, daemon=True).start()
        return fut

    def wait_started(self, task_id, timeout):
        """只有 never_started 才会在这里卡满 timeout 然后返回 False，
        与真 pool 的阻塞语义一致；立刻返回 False 会等同于
        「一进队列就超时」。"""
        if self.mode == "never_started":
            threading.Event().wait(timeout)
            return False
        return True

    def discard(self, task_id):
        self.discarded.append(task_id)


def make_task(n=1, seed=7):
    return Task(
        op="t2i", prompt="p", negative_prompt=None, width=1328, height=1328,
        seed=seed, steps=10, guidance=4.0, n=n,
    )


@pytest.fixture
def store(tmp_path):
    blob = BlobStore(str(tmp_path), ttl_seconds=3600)
    s = JobStore(
        FakePool(), blob, queue_deadline=1.0, infer_timeout=2.0, ttl_seconds=3600
    )
    yield s
    s.close()
    blob.close()


def _settle(job, store, timeout=4.0):
    import time

    end = time.time() + timeout
    while time.time() < end:
        if job.status in ("succeeded", "failed", "cancelled", "expired"):
            return job
        time.sleep(0.02)
    return job


def test_success_writes_blobs_and_seeds(tmp_path):
    blob = BlobStore(str(tmp_path), ttl_seconds=3600)
    s = JobStore(FakePool(), blob, queue_deadline=1.0, infer_timeout=2.0)
    job = s.submit(make_task(n=2, seed=100), "b64_json")
    job = _settle(job, s)
    assert job.status == "succeeded"
    assert len(job.blobs) == 2
    assert job.seeds == [100, 101]
    s.close()
    blob.close()


def test_worker_failure_becomes_failed(tmp_path):
    blob = BlobStore(str(tmp_path), ttl_seconds=3600)
    s = JobStore(FakePool("fail"), blob, queue_deadline=1.0, infer_timeout=2.0)
    job = _settle(s.submit(make_task(), "b64_json"), s)
    assert job.status == "failed"
    assert "CUDA OOM" in job.error
    s.close()
    blob.close()


def test_queue_deadline_expires(tmp_path):
    blob = BlobStore(str(tmp_path), ttl_seconds=3600)
    pool = FakePool("never_started")
    s = JobStore(pool, blob, queue_deadline=0.2, infer_timeout=2.0)
    job = _settle(s.submit(make_task(), "b64_json"), s)
    assert job.status == "expired"
    assert pool.discarded == ["t1"]
    s.close()
    blob.close()


def test_infer_timeout_kills_job(tmp_path):
    blob = BlobStore(str(tmp_path), ttl_seconds=3600)
    s = JobStore(FakePool("hang"), blob, queue_deadline=1.0, infer_timeout=0.3)
    job = _settle(s.submit(make_task(), "b64_json"), s)
    assert job.status == "failed"
    assert "Timeout" in job.error or "timeout" in job.error
    s.close()
    blob.close()


def test_cancel_queued_marks_cancelled(tmp_path):
    blob = BlobStore(str(tmp_path), ttl_seconds=3600)
    pool = FakePool("never_started")
    s = JobStore(pool, blob, queue_deadline=0.5, infer_timeout=2.0)
    job = s.submit(make_task(), "b64_json")
    job = s.cancel(job.job_id)
    assert job.status == "cancelled"
    assert pool.discarded == ["t1"]
    s.close()
    blob.close()


def test_queue_full_propagates(tmp_path):
    blob = BlobStore(str(tmp_path), ttl_seconds=3600)
    s = JobStore(FakePool("queue_full"), blob)
    with pytest.raises(QueueFull):
        s.submit(make_task(), "b64_json")
    s.close()
    blob.close()