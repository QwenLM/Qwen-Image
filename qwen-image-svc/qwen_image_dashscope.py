"""DashScope multimodal-generation compatibility adapter.

The adapter translates the public DashScope envelope into the local Task
structure. It never calls DashScope; inference remains on the local workers.
"""
from __future__ import annotations

import base64
import ipaddress
import re
import secrets
import socket
from typing import Annotated, Literal, Union
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import BaseModel, ConfigDict, Field

from errors import InvalidRequest
from protocol import Task

DASHSCOPE_T2I_MODELS = frozenset({"Qwen-Image-2512"})
DASHSCOPE_EDIT_MODELS = frozenset({"Qwen-Image-Edit-2511"})
DASHSCOPE_Z_IMAGE_MODELS = frozenset({"Z-Image", "Z-Image-Turbo"})
DEFAULT_SIZE = (1664, 928)
Z_IMAGE_DEFAULT_SIZE = (1024, 1536)
Z_IMAGE_MIN_PIXELS = 512 * 512
Z_IMAGE_MAX_PIXELS = 2048 * 2048
Z_IMAGE_PROMPT_MAX_CHARS = 800
MAX_REMOTE_IMAGE_BYTES = 10 * 1024 * 1024
REMOTE_IMAGE_TIMEOUT = 10.0


class TextContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=4000)


class ImageContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: str = Field(min_length=1)


MessageContent = Annotated[Union[TextContent, ImageContent], Field(union_mode="smart")]


class UserMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user"]
    content: list[MessageContent] = Field(min_length=1, max_length=16)


class DashScopeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[UserMessage] = Field(min_length=1, max_length=16)


class DashScopeParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    size: str | None = None
    n: int = Field(default=1, ge=1, le=4)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    negative_prompt: str = " "
    true_cfg_scale: float = Field(default=4.0, ge=0.0, le=20.0)
    guidance_scale: float = Field(default=1.0, ge=0.0, le=20.0)
    num_inference_steps: int = Field(default=25, ge=1, le=100)
    prompt_extend: bool | None = None
    watermark: bool | None = None


class DashScopeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=100)
    input: DashScopeInput
    parameters: DashScopeParameters = Field(default_factory=DashScopeParameters)


class DashScopeImage(BaseModel):
    image: str


class DashScopeAssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[DashScopeImage]


class DashScopeChoice(BaseModel):
    message: DashScopeAssistantMessage


class DashScopeOutput(BaseModel):
    choices: list[DashScopeChoice]


class DashScopeUsage(BaseModel):
    image_count: int


class DashScopeResponse(BaseModel):
    output: DashScopeOutput
    usage: DashScopeUsage
    request_id: str


class DashScopeError(BaseModel):
    code: str
    message: str
    request_id: str


def _parse_size(size: str | None, *, default: tuple[int, int] = DEFAULT_SIZE) -> tuple[int, int]:
    if size is None:
        return default
    match = re.fullmatch(r"(\d{2,4})[x*](\d{2,4})", size.strip())
    if not match:
        raise InvalidRequest("parameters.size must use WIDTH*HEIGHT format")
    width, height = (int(value) for value in match.groups())
    if not 64 <= width <= 2048 or not 64 <= height <= 2048:
        raise InvalidRequest("parameters.size must be between 64 and 2048 pixels")
    return width, height


def _parse_z_image_size(size: str | None) -> tuple[int, int]:
    width, height = _parse_size(size, default=Z_IMAGE_DEFAULT_SIZE)
    pixels = width * height
    if not Z_IMAGE_MIN_PIXELS <= pixels <= Z_IMAGE_MAX_PIXELS:
        raise InvalidRequest(
            "parameters.size total pixels must be between 512*512 and 2048*2048"
        )
    return width, height


def _public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise InvalidRequest("image must be an HTTP(S) URL or data URL")
    try:
        addresses = {
            ipaddress.ip_address(info[4][0])
            for info in socket.getaddrinfo(parsed.hostname, None)
        }
    except (OSError, ValueError) as exc:
        raise InvalidRequest("image URL host cannot be resolved") from exc
    if any(not address.is_global for address in addresses):
        raise InvalidRequest("image URL must resolve to a public address")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _image_to_base64(source: str) -> str:
    if source.startswith("data:"):
        header, separator, payload = source.partition(",")
        if separator == "" or ";base64" not in header:
            raise InvalidRequest("data image must be Base64 encoded")
        try:
            raw = base64.b64decode(payload, validate=True)
        except ValueError as exc:
            raise InvalidRequest("data image is not valid Base64") from exc
        if not raw or len(raw) > MAX_REMOTE_IMAGE_BYTES:
            raise InvalidRequest("image exceeds the 10 MiB limit")
        return base64.b64encode(raw).decode("ascii")

    _public_url(source)
    request = Request(source, headers={"User-Agent": "qwen-image-svc/1.0"})
    opener = build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=REMOTE_IMAGE_TIMEOUT) as response:
            raw = response.read(MAX_REMOTE_IMAGE_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise InvalidRequest("image URL could not be downloaded") from exc
    if not raw or len(raw) > MAX_REMOTE_IMAGE_BYTES:
        raise InvalidRequest("image exceeds the 10 MiB limit")
    return base64.b64encode(raw).decode("ascii")


def _request_parts(request: DashScopeRequest) -> tuple[str, list[str]]:
    texts: list[str] = []
    images: list[str] = []
    for message in request.input.messages:
        for item in message.content:
            if isinstance(item, TextContent):
                texts.append(item.text)
            else:
                images.append(_image_to_base64(item.image))
    if not texts:
        raise InvalidRequest("input.messages must contain text content")
    return "\n".join(texts), images


def _z_image_prompt(request: DashScopeRequest) -> str:
    texts: list[str] = []
    images = 0
    for message in request.input.messages:
        for item in message.content:
            if isinstance(item, TextContent):
                texts.append(item.text)
            else:
                images += 1
    if images:
        raise InvalidRequest("text-to-image requests cannot contain image content")
    if len(request.input.messages) != 1 or len(texts) != 1:
        raise InvalidRequest("Z-Image requests require exactly one text content")
    prompt = texts[0]
    if len(prompt) > Z_IMAGE_PROMPT_MAX_CHARS:
        prompt = prompt[:Z_IMAGE_PROMPT_MAX_CHARS]
    return prompt


def _z_image_task(request: DashScopeRequest) -> Task:
    parameters = request.parameters
    if parameters.n != 1:
        raise InvalidRequest("Z-Image requires parameters.n to be 1")
    if parameters.prompt_extend:
        raise InvalidRequest("prompt_extend is not supported in local mode")

    width, height = _parse_z_image_size(parameters.size)
    if request.model == "Z-Image-Turbo":
        op = "z_image_turbo"
        steps, guidance = 9, 0.0
    else:
        op = "z_image"
        steps, guidance = 25, 4.0

    return Task(
        op=op,
        prompt=_z_image_prompt(request),
        negative_prompt=" ",
        width=width,
        height=height,
        seed=parameters.seed,
        true_cfg_scale=0.0,
        guidance_scale=guidance,
        num_inference_steps=steps,
        num_images_per_prompt=1,
    )


def to_task(request: DashScopeRequest) -> Task:
    if request.model in DASHSCOPE_Z_IMAGE_MODELS:
        return _z_image_task(request)

    prompt, images = _request_parts(request)
    parameters = request.parameters
    width, height = _parse_size(parameters.size)

    if request.model in DASHSCOPE_T2I_MODELS:
        if images:
            raise InvalidRequest("text-to-image requests cannot contain image content")
        op = "t2i"
        task_width, task_height = width, height
    elif request.model in DASHSCOPE_EDIT_MODELS:
        if not images:
            raise InvalidRequest("image-edit requests require image content")
        if len(images) > 3:
            raise InvalidRequest("image-edit requests support at most three images")
        op = "edit"
        task_width, task_height = None, None
    else:
        raise InvalidRequest(f"unsupported DashScope model: {request.model}")

    return Task(
        op=op,
        prompt=prompt,
        negative_prompt=parameters.negative_prompt,
        width=task_width,
        height=task_height,
        seed=parameters.seed,
        true_cfg_scale=parameters.true_cfg_scale,
        guidance_scale=parameters.guidance_scale,
        num_inference_steps=parameters.num_inference_steps,
        num_images_per_prompt=parameters.n,
        images_b64=images,
    )


def from_job(job, *, public_base_url: str = "") -> DashScopeResponse:
    base = public_base_url.rstrip("/")
    content = [
        DashScopeImage(image=f"{base}/files/{blob}" if base else f"/files/{blob}")
        for blob in job.blobs
    ]
    return DashScopeResponse(
        output=DashScopeOutput(
            choices=[
                DashScopeChoice(
                    message=DashScopeAssistantMessage(content=content),
                )
            ]
        ),
        usage=DashScopeUsage(image_count=len(content)),
        request_id=job.job_id or secrets.token_hex(16),
    )
