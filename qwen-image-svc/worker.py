"""每个进程独占一张卡，持有且仅持有一个 pipeline。

这个文件是全项目唯一 import diffusers 的地方。
输入 WorkerEvent/Task，输出 WorkerEvent，除此之外不依赖任何东西
——既不认识 HTTP，也不认识 OpenAI 协议，更不认识任务状态机。

启动顺序刻意是「先加载，再报 ready」：/readyz 依赖这个信号判断
能不能接流量。加载失败不报 ready，API 会一直显示该 op 不可用，
这比假装健康、让请求进来再超时好。
"""
from __future__ import annotations

import base64
import io
import os
import time
from typing import Any

from protocol import Result, Task, WorkerEvent

# op -> (repo_id, pipeline 类名)。用类名字符串而非直接 import，
# 这样拿注册表的代码不需要把整个 diffusers 拖进来。
# 同时支持环境变量覆盖，部署机换权重路径不用改代码。
MODEL_REGISTRY = {
    "t2i": (os.environ.get("T2I_REPO", "Qwen/Qwen-Image"), "QwenImagePipeline"),
    "edit": (
        os.environ.get("EDIT_REPO", "Qwen/Qwen-Image-Edit"),
        "QwenImageEditPlusPipeline",
    ),
    "z_image": (
        os.environ.get("Z_IMAGE_REPO", "Tongyi-MAI/Z-Image"),
        "ZImagePipeline",
    ),
    "z_image_turbo": (
        os.environ.get("Z_IMAGE_TURBO_REPO", "Tongyi-MAI/Z-Image-Turbo"),
        "ZImagePipeline",
    ),
}


def _decode_images(images_b64: list[str]) -> list[Any]:
    from PIL import Image

    out = []
    for item in images_b64:
        raw = base64.b64decode(item)
        out.append(Image.open(io.BytesIO(raw)).convert("RGB"))
    return out


def _encode_outputs(images) -> list[bytes]:
    out: list[bytes] = []
    for im in images:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


class ModelHost:
    """一个 GPU 上的单模型宿主。

    op 在构造时固定，且只接受该 op 的任务。这是「一卡一模型」的
    硬约束在代码层的体现：edit 进程不可能因为收到 t2i 任务而
    偷偷加载第二份 60 GiB 权重。

    SVC_FAKE=1 时用 PIL 画占位图，不加载任何权重。用途是让整条
    HTTP -> pool -> worker -> blobs 链路能在没有显卡的机器上、
    或权重下载之前先验证一遍 —— 接口层的 bug 和「显存不够」
    必须能被分开定位。
    """

    def __init__(self, op: str, gpu_id: int, dtype: str = "bfloat16") -> None:
        self.op = op
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.dtype_name = dtype
        self._pipe: Any = None
        self._torch: Any = None
        self.fake = os.environ.get("SVC_FAKE") == "1"

    def load(self) -> None:
        if self.fake:
            return
        import importlib

        import torch

        self._torch = torch
        diffusers = importlib.import_module("diffusers")
        repo, cls_name = MODEL_REGISTRY[self.op]
        cls = getattr(diffusers, cls_name)
        pipe = cls.from_pretrained(
            repo, torch_dtype=getattr(torch, self.dtype_name)
        ).to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe

    def run(self, task: Task) -> Result:
        started = time.monotonic()
        try:
            if task.op != self.op:
                # 路由层错投，应当被当作 bug 而不是被吞掉的运行时错误
                raise ValueError(
                    f"worker bound to {self.op!r} received {task.op!r} task"
                )
            if self.fake:
                images = self._fake_generate(task)
            else:
                with self._torch.inference_mode():
                    images = self._generate(task)
            return Result(
                ok=True,
                task_id=task.task_id,
                images=_encode_outputs(images),
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
        except Exception as exc:  # noqa: BLE001 —— worker 绝不能因单任务异常退出
            return Result(
                ok=False,
                task_id=task.task_id,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )

    def _generate(self, task: Task) -> list[Any]:
        # num_images_per_prompt 仍按请求保留；为控制显存，实际逐张执行。
        base_seed = task.seed
        out: list[Any] = []
        for i in range(task.num_images_per_prompt):
            seed = base_seed + i
            gen = self._torch.Generator(device=self.device).manual_seed(seed)
            common = dict(
                prompt=task.prompt,
                negative_prompt=task.negative_prompt,
                num_inference_steps=task.num_inference_steps,
                true_cfg_scale=task.true_cfg_scale,
                guidance_scale=task.guidance_scale,
                num_images_per_prompt=1,
                generator=gen,
            )
            if self.op in {"z_image", "z_image_turbo"}:
                z_kwargs = dict(
                    prompt=task.prompt,
                    width=task.width,
                    height=task.height,
                    num_inference_steps=task.num_inference_steps,
                    guidance_scale=task.guidance_scale,
                    num_images_per_prompt=1,
                    generator=gen,
                )
                if self.op == "z_image" and task.negative_prompt.strip():
                    z_kwargs["negative_prompt"] = task.negative_prompt
                image = self._pipe(**z_kwargs).images[0]
            elif self.op == "t2i":
                image = self._pipe(
                    width=task.width,
                    height=task.height,
                    **common,
                ).images[0]
            else:
                if not task.images_b64:
                    raise ValueError("edit task requires at least one input image")
                image = self._pipe(
                    image=_decode_images(task.images_b64),
                    **common,
                ).images[0]
            out.append(image)
        return out

    def _fake_generate(self, task: Task) -> list[Any]:
        """画一张占位图，把关键参数印在图上用于链路验证。"""
        from PIL import Image, ImageDraw

        width = task.width or 1024
        height = task.height or 1024
        out = []
        for i in range(task.num_images_per_prompt):
            seed = task.seed + i
            im = Image.new("RGB", (width // 4, height // 4), (30, 30, 40))
            draw = ImageDraw.Draw(im)
            lines = [
                f"FAKE {self.op}",
                f"{width}x{height}",
                f"seed={seed}",
                f"gpu={self.gpu_id}",
                (task.prompt[:40] + "...")
                if len(task.prompt) > 40
                else task.prompt,
            ]
            for index, line in enumerate(lines):
                draw.text((10, 10 + index * 14), line, fill=(240, 240, 240))
            out.append(im)
        return out


def worker_main(op: str, gpu_id: int, task_q, event_q, stop_evt) -> None:
    """子进程入口。必须是模块级函数，spawn 才能 pickle 到它。"""
    host = ModelHost(op, gpu_id)
    try:
        host.load()
    except Exception as exc:  # noqa: BLE001
        event_q.put(
            WorkerEvent(kind="init_failed", op=op, error=f"{type(exc).__name__}: {exc}")
        )
        return

    event_q.put(WorkerEvent(kind="ready", op=op))

    while not stop_evt.is_set():
        try:
            task = task_q.get(timeout=1.0)
        except Exception:  # queue.Empty
            continue
        if task is None:  # 毒丸
            break
        # started 事件让 pool 能区分「排队中」和「已在 GPU 上跑」，
        # 两段超时预算靠它切分。
        event_q.put(WorkerEvent(kind="started", op=op, task_id=task.task_id))
        result = host.run(task)
        event_q.put(WorkerEvent(kind="result", op=op, result=result))