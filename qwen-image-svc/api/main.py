"""Qwen Image 原生 HTTP 服务。

这里负责鉴权、路由、协议映射和结果组装，不 import torch。
同步与异步接口共享同一个 JobStore；model 只用于选择 Qwen pipeline，
不模拟 OpenAI 的 URL、字段或响应格式。
"""
from __future__ import annotations

import os
import re
import secrets
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from blobs import BlobStore
from errors import ServiceError, Unauthorized
from jobs import JobStore
from pool import WorkerPool
from protocol import (
    ErrorBody,
    GeneratedImage,
    GenerateResponse,
    ImageRequest,
    JobAccepted,
    JobStatus,
)
from qwen_image_dashscope import (
    DashScopeError,
    DashScopeRequest,
    DashScopeResponse,
    from_job as dashscope_response,
    to_task as dashscope_task,
)

AUTH_TOKEN = os.environ.get("SVC_TOKEN", "")
DATA_DIR = os.environ.get("SVC_DATA_DIR", "./data")
PUBLIC_BASE_URL = os.environ.get("SVC_PUBLIC_BASE_URL", "").rstrip("/")
QUEUE_DEADLINE = float(os.environ.get("SVC_QUEUE_DEADLINE", "120"))
INFER_TIMEOUT = float(os.environ.get("SVC_INFER_TIMEOUT", "300"))
JOB_TTL = int(os.environ.get("SVC_JOB_TTL", str(24 * 3600)))
DASHSCOPE_PATH = "/api/v1/services/aigc/multimodal-generation/generation"

_BLOB_NAME = re.compile(r"^[0-9a-f]{64}\.png$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not AUTH_TOKEN:
        raise RuntimeError("SVC_TOKEN is required")

    blobs = BlobStore(f"{DATA_DIR}/blobs")
    pool = WorkerPool()
    app.state.blobs = blobs
    app.state.pool = pool
    app.state.jobs = JobStore(
        pool,
        blobs,
        queue_deadline=QUEUE_DEADLINE,
        infer_timeout=INFER_TIMEOUT,
        ttl_seconds=JOB_TTL,
    )
    # 模型在 worker 子进程中异步加载，/readyz 负责表达是否可以接流量。
    yield
    app.state.jobs.close()
    pool.close()
    blobs.close()


app = FastAPI(title="qwen-image-svc", lifespan=lifespan)


def require_token(authorization: Optional[str] = Header(default=None)) -> None:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token, AUTH_TOKEN):
        raise Unauthorized("missing or invalid bearer token")


def _is_dashscope(request: Request) -> bool:
    return request.url.path == DASHSCOPE_PATH


def _dashscope_error(status: int, code: str, message: str) -> JSONResponse:
    body = DashScopeError(
        code=code,
        message=message,
        request_id=secrets.token_hex(16),
    )
    return JSONResponse(body.model_dump(), status_code=status)


def _dashscope_error_code(status: int) -> str:
    return {
        400: "InvalidParameter",
        401: "InvalidApiKey",
        404: "NotFound",
        429: "Throttling",
        503: "ServiceUnavailable",
        504: "Timeout",
    }.get(status, "InternalError")


@app.exception_handler(ServiceError)
async def _service_error(request: Request, exc: ServiceError) -> JSONResponse:
    headers = {}
    if exc.retry_after:
        headers["Retry-After"] = str(exc.retry_after)
    if _is_dashscope(request):
        response = _dashscope_error(
            exc.status,
            _dashscope_error_code(exc.status),
            exc.message,
        )
        response.headers.update(headers)
        return response
    body = ErrorBody(message=exc.message, type=exc.type, code=str(exc.status))
    return JSONResponse(body.model_dump(), status_code=exc.status, headers=headers)


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, _: RequestValidationError) -> JSONResponse:
    if _is_dashscope(request):
        return _dashscope_error(400, "InvalidParameter", "request validation failed")
    return JSONResponse(
        ErrorBody(message="request validation failed", code="422").model_dump(),
        status_code=422,
    )


@app.exception_handler(HTTPException)
async def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
    if _is_dashscope(request):
        return _dashscope_error(
            exc.status_code,
            _dashscope_error_code(exc.status_code),
            str(exc.detail),
        )
    body = ErrorBody(message=str(exc.detail), code=str(exc.status_code))
    return JSONResponse(body.model_dump(), status_code=exc.status_code)


@app.exception_handler(Exception)
async def _unhandled(request: Request, _: Exception) -> JSONResponse:
    if _is_dashscope(request):
        return _dashscope_error(500, "InternalError", "internal error")
    body = ErrorBody(message="internal error", type="internal_error", code="500")
    return JSONResponse(body.model_dump(), status_code=500)


# ------------------------------------------------------------------ 响应组装

def _images_from_job(job) -> list[GeneratedImage]:
    return [
        GeneratedImage(url=f"/files/{blob}", seed=seed)
        for blob, seed in zip(job.blobs, job.seeds)
    ]


def _job_payload(job) -> JobStatus:
    return JobStatus(
        job_id=job.job_id,
        status=job.status,
        created=job.created,
        images=_images_from_job(job) if job.status == "succeeded" else [],
        elapsed_ms=job.elapsed_ms,
        error=job.error,
    )


# ------------------------------------------------------------------ DashScope 兼容路由

@app.post(
    DASHSCOPE_PATH,
    response_model=DashScopeResponse,
    dependencies=[Depends(require_token)],
)
def dashscope_generate(
    req: DashScopeRequest,
    request: Request,
) -> DashScopeResponse:
    """使用本地 Qwen worker 提供 DashScope 同步响应格式。"""
    job = app.state.jobs.submit(dashscope_task(req))
    job = app.state.jobs.wait(job)
    if job.status != "succeeded":
        raise _failed(job)
    base_url = PUBLIC_BASE_URL or str(request.base_url).rstrip("/")
    return dashscope_response(job, public_base_url=base_url)


# ------------------------------------------------------------------ Qwen 原生路由

@app.post(
    "/generate",
    response_model=GenerateResponse,
    dependencies=[Depends(require_token)],
)
def generate(req: ImageRequest) -> GenerateResponse:
    """同步执行文生图或图像编辑，由请求中的 model 选择 pipeline。"""
    job = app.state.jobs.submit(req.to_task())
    job = app.state.jobs.wait(job)
    if job.status != "succeeded":
        raise _failed(job)
    return GenerateResponse(
        images=_images_from_job(job),
        elapsed_ms=job.elapsed_ms or 0,
    )


@app.post(
    "/jobs",
    status_code=202,
    response_model=JobAccepted,
    dependencies=[Depends(require_token)],
)
def create_job(req: ImageRequest, request: Request) -> JobAccepted:
    """异步提交文生图或编辑任务。请求体与 /generate 完全一致。"""
    job = app.state.jobs.submit(req.to_task())
    return JobAccepted(
        job_id=job.job_id,
        status_url=str(request.url_for("get_job", job_id=job.job_id)),
    )


@app.get(
    "/jobs/{job_id}",
    response_model=JobStatus,
    name="get_job",
    dependencies=[Depends(require_token)],
)
def get_job(job_id: str) -> JobStatus:
    return _job_payload(app.state.jobs.get(job_id))


@app.delete(
    "/jobs/{job_id}",
    response_model=JobStatus,
    dependencies=[Depends(require_token)],
)
def cancel_job(job_id: str) -> JobStatus:
    return _job_payload(app.state.jobs.cancel(job_id))


@app.get("/files/{name}")
def get_file(name: str) -> Response:
    """下载内容寻址的 PNG 结果。"""
    from errors import NotFound

    if not _BLOB_NAME.match(name):
        raise NotFound(f"invalid blob name: {name}")
    try:
        return Response(content=app.state.blobs.get(name), media_type="image/png")
    except FileNotFoundError:
        raise NotFound(f"blob {name} not found")
    except OSError as exc:
        raise ServiceError(f"blob read failed: {exc}")


# ------------------------------------------------------------------ 健康与指标

@app.get("/healthz", response_class=PlainTextResponse)
def healthz() -> str:
    return "ok"


@app.get("/readyz", response_class=PlainTextResponse)
def readyz() -> Response:
    snap = app.state.pool.ready_snapshot()
    ok = all(value == "ready" for value in snap.values())
    lines = [f"{op}={state}" for op, state in snap.items()]
    return PlainTextResponse(
        ("ready" if ok else "not_ready") + "\n" + "\n".join(lines),
        status_code=200 if ok else 503,
    )


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    snap = app.state.pool.ready_snapshot()
    lines = []
    for op, state in snap.items():
        lines.append(f'svc_op_ready{{op="{op}"}} {1 if state == "ready" else 0}')
        lines.append(f'svc_queue_depth{{op="{op}"}} {app.state.pool.queue_depth(op)}')
    return "\n".join(lines) + "\n"


def _failed(job) -> ServiceError:
    from errors import InferenceFailed, InferenceTimeout, Overloaded

    message = job.error or "unknown"
    if job.status == "expired":
        return Overloaded(message, retry_after=30)
    if "Timeout" in message or "timeout" in message:
        return InferenceTimeout(message)
    return InferenceFailed(message)