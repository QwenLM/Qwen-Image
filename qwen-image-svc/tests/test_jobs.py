"""JobStore 状态机单测：用假 pool 覆盖所有终态，不需要 GPU。"""
import threading
from concurrent.futures import Future

import pytest

from blobs import BlobStore
from errors import QueueFull
from jobs import JobStore
from protocol import Result, Task


class FakePool:
    def __init__(self, mode="ok", delay=0.0):
        self.mode = mode
        self.delay = delay
        self.discarded = []

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
                fut.set_result(Result(ok=False, task_id="t1", error="CUDA OOM"))
            elif self.mode == "hang":
                return
            else:
                fut.set_result(
                    Result(
                        ok=True,
                        task_id="t1",
                        images=[b"PNG1", b"PNG2"],
                        elapsed_ms=123,
                    )
                )

        threading.Thread(target=run, daemon=True).start()
        return fut

    def wait_started(self, task_id, timeout):
        if self.mode == "never_started":
            threading.Event().wait(timeout)
            return False
        return True

    def discard(self, task_id):
        self.discarded.append(task_id)


def make_task(num_images_per_prompt=1, seed=7):
    return Task(
        op="t2i",
        prompt="p",
        negative_prompt=" ",
        width=1664,
        height=928,
        seed=seed,
        true_cfg_scale=4.0,
        guidance_scale=1.0,
        num_inference_steps=50,
        num_images_per_prompt=num_images_per_prompt,
    )


def settle(job, timeout=4.0):
    import time

    end = time.time() + timeout
    while time.time() < end:
        if job.status in ("succeeded", "failed", "cancelled", "expired"):
            return job
        time.sleep(0.02)
    return job


def test_success_writes_blobs_seeds_and_elapsed(tmp_path):
    blob = BlobStore(str(tmp_path), ttl_seconds=3600)
    store = JobStore(FakePool(), blob, queue_deadline=1.0, infer_timeout=2.0)

    job = settle(store.submit(make_task(num_images_per_prompt=2)))

    assert job.status == "succeeded"
    assert len(job.blobs) == 2
    assert job.seeds == [7, 8]
    assert job.elapsed_ms == 123
    store.close()
    blob.close()


def test_worker_failure_becomes_failed(tmp_path):
    blob = BlobStore(str(tmp_path), ttl_seconds=3600)
    store = JobStore(FakePool("fail"), blob, queue_deadline=1.0, infer_timeout=2.0)

    job = settle(store.submit(make_task()))

    assert job.status == "failed"
    assert "CUDA OOM" in job.error
    store.close()
    blob.close()


def test_queue_deadline_expires(tmp_path):
    blob = BlobStore(str(tmp_path), ttl_seconds=3600)
    pool = FakePool("never_started")
    store = JobStore(pool, blob, queue_deadline=0.2, infer_timeout=2.0)

    job = settle(store.submit(make_task()))

    assert job.status == "expired"
    assert pool.discarded == ["t1"]
    store.close()
    blob.close()


def test_infer_timeout_fails_job(tmp_path):
    blob = BlobStore(str(tmp_path), ttl_seconds=3600)
    store = JobStore(FakePool("hang"), blob, queue_deadline=1.0, infer_timeout=0.3)

    job = settle(store.submit(make_task()))

    assert job.status == "failed"
    assert "Timeout" in job.error or "timeout" in job.error
    store.close()
    blob.close()


def test_cancel_queued_marks_cancelled(tmp_path):
    blob = BlobStore(str(tmp_path), ttl_seconds=3600)
    pool = FakePool("never_started")
    store = JobStore(pool, blob, queue_deadline=0.5, infer_timeout=2.0)

    job = store.submit(make_task())
    job = store.cancel(job.job_id)

    assert job.status == "cancelled"
    assert pool.discarded == ["t1"]
    store.close()
    blob.close()


def test_queue_full_propagates(tmp_path):
    blob = BlobStore(str(tmp_path), ttl_seconds=3600)
    store = JobStore(FakePool("queue_full"), blob)

    with pytest.raises(QueueFull):
        store.submit(make_task())

    store.close()
    blob.close()