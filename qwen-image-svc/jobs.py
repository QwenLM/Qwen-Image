"""任务状态机：唯一知道「一个请求现在处于什么阶段」的地方。

同步接口不是另一条路径，而是这个状态机上的 submit + 等待。
这样取消、超时、错误码全都只在这里实现一次。

只在 succeeded 之后才落盘（blobs），job 表里只存 blob 文件名，
绝不存 base64 —— 否则 job 表会随 TTL 变成内存黑洞。
"""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Optional

from blobs import BlobStore
from errors import NotFound, Overloaded, QueueFull
from pool import WorkerPool
from protocol import Result, Task

TERMINAL = {"succeeded", "failed", "cancelled", "expired"}


@dataclass
class Job:
    job_id: str
    task: Task
    status: str = "queued"          # queued|running|succeeded|failed|cancelled|expired
    created: int = field(default_factory=lambda: int(time.time()))
    finished: Optional[int] = None
    blobs: list[str] = field(default_factory=list)   # 文件名，不是路径
    seeds: list[int] = field(default_factory=list)
    elapsed_ms: Optional[int] = None
    error: Optional[str] = None
    # watcher 信号量的一次性释放标记。cancel() 与 _watch 的 finally
    # 都会走到这里，靠它保证只释放一次；否则取消与完成撞上会多释放，
    # max_watchers 就越界了。
    released: bool = False


class JobStore:
    def __init__(
        self,
        pool: WorkerPool,
        blobs: BlobStore,
        *,
        queue_deadline: float = 480.0,
        infer_timeout: float = 300.0,
        ttl_seconds: int = 24 * 3600,
        max_watchers: int = 64,
    ) -> None:
        self.pool = pool
        self.blobs = blobs
        self.queue_deadline = queue_deadline
        self.infer_timeout = infer_timeout
        self.ttl = ttl_seconds
        self.max_watchers = max_watchers

        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._watchers = threading.Semaphore(max_watchers)
        self._stop = threading.Event()
        threading.Thread(target=self._gc_loop, daemon=True).start()

    # ---------------------------------------------------------------- 提交
    def submit(self, task: Task) -> Job:
        """同步与异步共用。异步路径拿 job 立即返回，watcher 在后台推进状态。"""
        if not self._watchers.acquire(blocking=False):
            raise Overloaded("too many in-flight jobs", retry_after=15)
        job = Job(job_id=uuid.uuid4().hex, task=task)
        try:
            fut = self.pool.submit(task)
        except QueueFull:
            self._watchers.release()
            raise

        with self._lock:
            self._jobs[job.job_id] = job

        t = threading.Thread(
            target=self._watch, args=(job, fut), daemon=True
        )
        t.start()
        return job

    def wait(self, job: Job) -> Job:
        """同步路径专用：等 job 进入终态。"""
        deadline = time.monotonic() + self.queue_deadline + self.infer_timeout + 30
        while time.monotonic() < deadline:
            if self.get(job.job_id).status in TERMINAL:
                return self.get(job.job_id)
            time.sleep(0.2)
        return self.get(job.job_id)

    # ---------------------------------------------------------------- 查询/取消
    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise NotFound(f"job {job_id} not found or expired")
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status == "running":
            # 进了 GPU 就是不可中断的：pipeline 没有 mid-step 取消点。
            # 如实返回当前状态，比假装取消成功好。
            return job
        if not self._finish(job, "cancelled"):
            return job  # 已经处于别的终态
        self.pool.discard(job.task.task_id)
        self._release(job)
        return job

    # ---------------------------------------------------------------- 内部
    def _release(self, job: Job) -> None:
        """watcher 信号量的唯一释放点，幂等。"""
        with self._lock:
            if job.released:
                return
            job.released = True
        self._watchers.release()

    def _finish(self, job: Job, status: str, error: Optional[str] = None) -> bool:
        """写入终态。已经是终态则不动并返回 False。

        cancel() 与 watcher 是两条线程，谁先到谁说了算；
        没有这个守卫，被取消的任务会在排队超时后被改写成 expired。
        """
        with self._lock:
            if job.status in TERMINAL:
                return False
            job.status = status
            job.error = error
            job.finished = int(time.time())
            return True

    def _watch(self, job: Job, fut: Future) -> None:
        try:
            # 阶段一：排队。wait_started 只代表被 worker 取走，
            # 不代表任务没被取消 —— 两条线程竞态要在每次改状态前检查。
            if not self.pool.wait_started(job.task.task_id, self.queue_deadline):
                if self._finish(job, "expired", "queue wait exceeded deadline"):
                    self.pool.discard(job.task.task_id)
                return
            if job.status == "cancelled":
                return
            with self._lock:
                if job.status in TERMINAL:
                    return
                job.status = "running"

            # 阶段二：推理
            res: Result = fut.result(timeout=self.infer_timeout)
            if not res.ok:
                self._finish(job, "failed", res.error)
                return

            job.blobs = [self.blobs.put(b) for b in res.images]
            job.seeds = [job.task.seed + i for i in range(len(res.images))]
            job.elapsed_ms = res.elapsed_ms
            self._finish(job, "succeeded")
        except Exception as exc:  # noqa: BLE001
            self._finish(job, "failed", f"{type(exc).__name__}: {exc}")
        finally:
            self._release(job)

    def _gc_loop(self) -> None:
        while not self._stop.wait(300):
            cutoff = time.time() - self.ttl
            with self._lock:
                dead = [
                    jid
                    for jid, j in self._jobs.items()
                    if j.status in TERMINAL and (j.finished or j.created) < cutoff
                ]
                for jid in dead:
                    del self._jobs[jid]

    def close(self) -> None:
        self._stop.set()