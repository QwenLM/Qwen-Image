"""进程池与路由：op -> 专职 GPU。

核心约束：worker 按 op 固定分配，不共享。
两个 pipeline 各约 60 GiB，一张 80 GiB 卡装不下两份，所以卡是
"专职"的：GPU0 -> t2i only，GPU1 -> edit only。副作用是好的：
一张卡被编辑任务占满时，文生图完全不受影响，不需要任何抢占逻辑。

背压模型：每个 op 的「在跑 + 排队」名额固定为 QUEUE_SLOTS_PER_OP。
一张卡同一时刻仍只跑 1 个推理（worker 逐个取队列）；多余请求先排队。
队列满了才 429，避免「第二请求立刻 saturated」。
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from multiprocessing import get_context

from errors import QueueFull, ServiceError
from protocol import Result, Task, WorkerEvent


def _parse_gpus(env_name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(env_name)
    if raw is None:
        return list(default)
    raw = raw.strip()
    if not raw:
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


# op -> 分配到的 GPU 编号。可用环境变量覆盖。
GPU_ASSIGNMENT: dict[str, list[int]] = {
    "t2i": _parse_gpus("T2I_GPUS", [0, 1]),
    "edit": _parse_gpus("EDIT_GPUS", [2, 3, 6]),
    "z_image": _parse_gpus("Z_IMAGE_GPUS", [4, 5]),
    "z_image_turbo": _parse_gpus("Z_IMAGE_TURBO_GPUS", [7]),
}

# 每个 op「在跑 + 排队」的任务上限，与卡数无关。
# 2 张卡时相当于每卡 6 个，3 张卡时相当于每卡 4 个，1 张卡时 12 个都堆在这一张上。
QUEUE_SLOTS_PER_OP = 12


@dataclass
class _Pending:
    future: Future
    started: threading.Event
    submitted_at: float


class WorkerPool:
    def __init__(self, assignment: dict[str, list[int]] | None = None) -> None:
        self.assignment = assignment or GPU_ASSIGNMENT
        # spawn 而非 fork：CUDA 上下文不能跨 fork 继承。
        # 注意用标准库 multiprocessing 而非 torch.multiprocessing ——
        # 我们把 Task/Result 都设计成基础类型 + bytes，不需要 torch 的
        # 共享内存机制。这样 API 进程永远不会 import torch。
        self._ctx = get_context("spawn")
        self._stop = self._ctx.Event()
        self._event_q = self._ctx.Queue()

        self._task_q = {
            op: self._ctx.Queue(maxsize=QUEUE_SLOTS_PER_OP)
            for op, gpus in self.assignment.items()
        }
        # 与队列容量对齐：满员前允许入队等待，而不是按卡数瞬间拒掉。
        self._slots = {
            op: threading.Semaphore(QUEUE_SLOTS_PER_OP)
            for op in self.assignment
        }
        self._ready = {op: threading.Event() for op in self.assignment}
        self._init_error: dict[str, str | None] = {op: None for op in self.assignment}

        self._pending: dict[str, _Pending] = {}
        self._lock = threading.Lock()
        self._procs: list = []

        self._spawn_all()
        self._dispatcher = threading.Thread(target=self._drain, daemon=True)
        self._dispatcher.start()
        self._supervisor = threading.Thread(target=self._supervise, daemon=True)
        self._supervisor.start()

    # ---------------------------------------------------------------- 生命周期
    def _spawn_all(self) -> None:
        from worker import worker_main  # 子进程内才会有 import diffusers

        for op, gpus in self.assignment.items():
            for gpu_id in gpus:
                p = self._ctx.Process(
                    target=worker_main,
                    args=(op, gpu_id, self._task_q[op], self._event_q, self._stop),
                    daemon=True,
                )
                p.start()
                self._procs.append(p)

    def wait_ready(self, op: str, timeout: float = 900.0) -> bool:
        return self._ready[op].wait(timeout)

    def ready_snapshot(self) -> dict[str, str]:
        out = {}
        for op in self.assignment:
            if self._init_error[op]:
                out[op] = f"init_failed: {self._init_error[op]}"
            elif self._ready[op].is_set():
                out[op] = "ready"
            else:
                out[op] = "loading"
        return out

    def queue_depth(self, op: str) -> int:
        return self._task_q[op].qsize()

    # ---------------------------------------------------------------- 投递
    def submit(self, task: Task) -> Future:
        op = task.op
        if op not in self._task_q:
            raise ServiceError(f"no worker for op {op!r}")

        task.task_id = f"{op}-{uuid.uuid4().hex[:12]}"
        pending = _Pending(Future(), threading.Event(), time.time())

        if not self._slots[op].acquire(blocking=False):
            # 队列（在跑+等待）已满：让客户端退避；未满则入队等待空闲 GPU
            raise QueueFull(f"{op} is saturated", retry_after=10)

        with self._lock:
            self._pending[task.task_id] = pending
        try:
            self._task_q[op].put(task, timeout=5)
        except Exception:
            # 入了信号量但队列满（理论上不会，两者容量一致）
            self._slots[op].release()
            with self._lock:
                self._pending.pop(task.task_id, None)
            raise QueueFull(f"{op} queue is full", retry_after=10)
        return pending.future

    def wait_started(self, task_id: str, timeout: float) -> bool:
        with self._lock:
            pending = self._pending.get(task_id)
        if pending is None:
            return False
        return pending.started.wait(timeout)

    def discard(self, task_id: str) -> None:
        """请求方放弃。只摘掉 Future；如果任务已在 GPU 上，绝不释放信号量
        ——槽位必须由结果回收时释放，否则会算成「空出来一张卡」。"""
        with self._lock:
            self._pending.pop(task_id, None)

    # ---------------------------------------------------------------- 结果回收
    def _drain(self) -> None:
        while not self._stop.is_set():
            try:
                evt: WorkerEvent = self._event_q.get(timeout=1.0)
            except Exception:
                continue

            if evt.kind == "ready":
                self._ready[evt.op].set()
                self._init_error[evt.op] = None
                continue
            if evt.kind == "init_failed":
                self._init_error[evt.op] = evt.error
                self._ready[evt.op].clear()
                continue
            if evt.kind == "started":
                with self._lock:
                    pending = self._pending.get(evt.task_id)
                if pending:
                    pending.started.set()
                continue

            # kind == "result"
            res: Result = evt.result  # type: ignore[assignment]
            with self._lock:
                pending = self._pending.pop(res.task_id, None)
            if pending and not pending.future.done():
                pending.future.set_result(res)
            # 槽位只在这里释放：任务真正离开 GPU 的时刻
            self._slots[evt.op].release()

    # ---------------------------------------------------------------- 监督
    def _supervise(self) -> None:
        """worker 进程死了就重启。没有它，一次 CUDA OOM 会让该 op 永久不可用。"""
        from worker import worker_main

        backoff = 5.0
        while not self._stop.wait(5.0):
            for idx, p in enumerate(self._procs):
                if p.is_alive():
                    continue
                op = self._op_of(idx)
                gpu_id = self._gpu_of(idx)
                self._ready[op].clear()
                self._init_error[op] = f"worker on cuda:{gpu_id} exited, restarting"
                np = self._ctx.Process(
                    target=worker_main,
                    args=(op, gpu_id, self._task_q[op], self._event_q, self._stop),
                    daemon=True,
                )
                np.start()
                self._procs[idx] = np
                time.sleep(min(backoff, 60.0))
                backoff = min(backoff * 2, 60.0)

    def _op_of(self, idx: int) -> str:
        return self._index[idx][0]

    def _gpu_of(self, idx: int) -> int:
        return self._index[idx][1]

    @property
    def _index(self) -> list[tuple[str, int]]:
        return [(op, g) for op, gs in self.assignment.items() for g in gs]

    # ---------------------------------------------------------------- 关闭
    def close(self) -> None:
        self._stop.set()
        for q in self._task_q.values():
            try:
                q.put(None, timeout=1)
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()