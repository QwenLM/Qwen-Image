"""协议层：定义"外部长什么样"与"内部长什么样"之间的唯一映射。

这一层不 import torch / diffusers / fastapi，可以被打包进任何进程，
也是全项目唯一能在没有 GPU 的机器上完整单测的模块。

设计约束：
- 对外请求只暴露我们真正支持的参数子集。每多一个透传参数，
  就多一条协议层到 worker 层的映射路径。
- Task / Result 刻意用 dataclass + 基础类型而非 Pydantic：
  multiprocessing 的 pickle 面对 Pydantic v2 对象有跨版本边界问题，
  基础类型永远安全。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

# ------------------------------------------------------------------ 常量

# 官方 preset 尺寸。非 preset 尺寸画质不稳定，直接拒绝比默默降级好。
IMAGE_SIZES: Dict[str, tuple] = {
    "1328x1328": (1328, 1328),
    "1664x928": (1664, 928),
    "928x1664": (928, 1664),
    "1472x1140": (1472, 1140),
    "1140x1472": (1140, 1472),
}

ImageSize = Literal[
    "1328x1328", "1664x928", "928x1664", "1472x1140", "1140x1472"
]

# 模型别名 -> 内部 op。别名隔绝了 HF repo id，调用方不会
# 误以为可以填任意模型名。
MODEL_TO_OP: Dict[str, str] = {
    "qwen-image": "t2i",
    "qwen-image-edit": "edit",
}

ModelAlias = Literal["qwen-image", "qwen-image-edit"]
Op = Literal["t2i", "edit"]


# ------------------------------------------------------------------ 对外请求

class ImageRequest(BaseModel):
    """同步生图与异步提交共用的请求体。"""

    model: ModelAlias = "qwen-image"
    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: Optional[str] = None
    n: int = Field(default=1, ge=1, le=4)
    size: ImageSize = "1328x1328"
    seed: Optional[int] = Field(default=None, ge=0, le=2**31 - 1)
    steps: int = Field(default=40, ge=1, le=50)
    guidance: float = Field(default=4.0, ge=0.0, le=10.0)
    response_format: Literal["b64_json", "url"] = "b64_json"
    # 仅编辑模型使用；文生图带了图直接拒绝，免得调用方以为它生效了。
    image_b64: Optional[List[str]] = Field(default=None, max_length=3)

    @model_validator(mode="after")
    def _check_op_payload(self) -> "ImageRequest":
        op = MODEL_TO_OP[self.model]
        if op == "edit" and not self.image_b64:
            raise ValueError("model=qwen-image-edit requires image_b64")
        if op == "t2i" and self.image_b64:
            raise ValueError("image_b64 is only accepted by qwen-image-edit")
        return self

    def to_task(self) -> "Task":
        """协议对象 -> 内部任务。这是唯一一次字段映射，别处不要再映射。"""
        width, height = IMAGE_SIZES[self.size]
        return Task(
            op=MODEL_TO_OP[self.model],
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            width=width,
            height=height,
            seed=self.seed,
            steps=self.steps,
            guidance=self.guidance,
            n=self.n,
            image_b64=self.image_b64 or [],
        )


# ------------------------------------------------------------------ 对外响应

class ImageDatum(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    seed: Optional[int] = None


class ImageResponse(BaseModel):
    created: int
    data: List[ImageDatum]


class JobAccepted(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"
    poll_url: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled", "expired"]
    created: int
    # 字段名与 ImageResponse.data 保持一致：同一个东西只该有一种键名，
    # 否则客户端拿同一个解析函数处理两个接口就会踩空。
    data: List[ImageDatum] = Field(default_factory=list)
    error: Optional[str] = None


class ErrorBody(BaseModel):
    message: str
    type: str = "invalid_request_error"
    code: Optional[str] = None


# ------------------------------------------------------------------ 进程间

# 刻意不用 @dataclass(slots=True)：那是 3.10+ 才有的参数，
# 而这三类对象每请求只创建一两个，slots 的内存收益可以忽略。
# 保留 3.9 兼容性让本机也能跑测试，比这点内存重要。


@dataclass
class Task:
    """跨越队列的对象。

    注意这里没有 progress_callback —— 回调不可跨进程，
    进度只能靠 pool 侧的 started 事件近似。
    """

    op: Op
    prompt: str
    negative_prompt: Optional[str]
    width: int
    height: int
    seed: Optional[int]
    steps: int
    guidance: float
    n: int
    image_b64: List[str] = field(default_factory=list)
    task_id: str = ""  # 由 pool 投递时填充，worker 只读


@dataclass
class Result:
    """worker -> pool 的回传。成功时 images 是 PNG bytes 列表。"""

    ok: bool
    task_id: str
    images: List[bytes] = field(default_factory=list)
    error: Optional[str] = None
    elapsed_ms: int = 0


@dataclass
class WorkerEvent:
    """worker -> pool 的控制/数据事件。

    显式 kind 而非用 task_id 前缀编码语义：旧版本把
    "__ready__:op" 塞进 task_id，加载失败的信号没人处理，
    导致 /readyz 永远不就绪且请求挂到超时。
    """

    kind: Literal["ready", "init_failed", "started", "result"]
    op: Op
    task_id: str = ""
    error: Optional[str] = None
    result: Optional[Result] = None
    ts: float = field(default_factory=time.time)


def now() -> int:
    return int(time.time())