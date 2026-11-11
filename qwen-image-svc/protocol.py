"""Qwen Image 原生协议层。

这里的字段直接对应 Qwen diffusers pipeline 参数，不模拟 OpenAI 请求或响应。
协议层只负责校验、模型分派和进程间基础数据结构；不 import torch、diffusers 或 FastAPI。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Annotated, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

# ------------------------------------------------------------------ 模型与参数

ModelAlias = Literal["qwen-image", "qwen-image-edit"]
Op = Literal["t2i", "edit", "z_image", "z_image_turbo"]


class QwenParameters(BaseModel):
    """两个 Qwen pipeline 共用的原生推理参数。"""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str = " "
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    true_cfg_scale: float = Field(default=4.0, ge=0.0, le=20.0)
    guidance_scale: float = Field(default=1.0, ge=0.0, le=20.0)
    num_inference_steps: int = Field(default=25, ge=1, le=100)
    num_images_per_prompt: int = Field(default=1, ge=1, le=4)


class GenerateRequest(QwenParameters):
    """Qwen-Image 文生图请求。"""

    model: Literal["qwen-image"] = "qwen-image"
    width: int = Field(default=1664, ge=64, le=2048)
    height: int = Field(default=928, ge=64, le=2048)

    def to_task(self) -> "Task":
        return Task(
            op="t2i",
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            width=self.width,
            height=self.height,
            seed=self.seed,
            true_cfg_scale=self.true_cfg_scale,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            num_images_per_prompt=self.num_images_per_prompt,
        )


class EditRequest(QwenParameters):
    """Qwen-Image-Edit 请求。输入图字段直接命名为 images。"""

    model: Literal["qwen-image-edit"] = "qwen-image-edit"
    images: List[str] = Field(min_length=1, max_length=3)

    def to_task(self) -> "Task":
        return Task(
            op="edit",
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            width=None,
            height=None,
            seed=self.seed,
            true_cfg_scale=self.true_cfg_scale,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            num_images_per_prompt=self.num_images_per_prompt,
            images_b64=self.images,
        )


ImageRequest = Annotated[
    Union[GenerateRequest, EditRequest], Field(discriminator="model")
]


# ------------------------------------------------------------------ 原生响应

class GeneratedImage(BaseModel):
    url: str
    seed: int


class GenerateResponse(BaseModel):
    images: List[GeneratedImage]
    elapsed_ms: int


class JobAccepted(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"
    status_url: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled", "expired"]
    created: int
    images: List[GeneratedImage] = Field(default_factory=list)
    elapsed_ms: Optional[int] = None
    error: Optional[str] = None


class ErrorBody(BaseModel):
    message: str
    type: str = "request_error"
    code: Optional[str] = None


# ------------------------------------------------------------------ 进程间协议

@dataclass
class Task:
    """跨越 multiprocessing 队列的基础数据对象。"""

    op: Op
    prompt: str
    negative_prompt: str
    width: Optional[int]
    height: Optional[int]
    seed: int
    true_cfg_scale: float
    guidance_scale: float
    num_inference_steps: int
    num_images_per_prompt: int
    images_b64: List[str] = field(default_factory=list)
    task_id: str = ""


@dataclass
class Result:
    """worker 返回的 PNG bytes。"""

    ok: bool
    task_id: str
    images: List[bytes] = field(default_factory=list)
    error: Optional[str] = None
    elapsed_ms: int = 0


@dataclass
class WorkerEvent:
    """worker 到 pool 的控制或结果事件。"""

    kind: Literal["ready", "init_failed", "started", "result"]
    op: Op
    task_id: str = ""
    error: Optional[str] = None
    result: Optional[Result] = None
    ts: float = field(default_factory=time.time)


def now() -> int:
    return int(time.time())