"""HTTP 薄壳：鉴权、路由、协议映射。不碰 torch。

唯一的职责边界在于：这里出现的所有错误都必须来自 errors.py，
handler 里不做「这是 400 还是 500」的判断。
"""
from __future__ import annotations

import os
import re
import secrets
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from blobs import BlobStore
from errors import ServiceError, Unauthorized
from jobs import JobStore
from pool import WorkerPool
from protocol import (
    ErrorBody,
    ImageDatum,
    ImageRequest,
    ImageResponse,
    JobAccepted,
    JobStatus,
    now,
)

AUTH_TOKEN = os.environ.get("SVC_TOKEN", "")
DATA_DIR = os.environ.get("SVC_DATA_DIR", "./data")
QUEUE_DEADLINE = float(os.environ.get("SVC_QUEUE_DEADLINE", "120"))
INFER_TIMEOUT = float(os.environ.get("SVC_INFER_TIMEOUT", "300"))
JOB_TTL = int(os.environ.get("SVC_JOB_TTL", str(24 * 3600)))

_BLOB_NAME = re.compile(r"^[0-9a-f]{64}\.png$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not AUTH_TOKEN:
        # 没有 token 就拒绝启动：对外开放的推理接口不该裸奔
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
    # 不阻塞启动等待权重加载：/readyz 负责表达「能不能接流量」。
    # 否则一个加载失败会变成进程起不来，运维看不到日志。
    yield
    app.state.jobs.close()
    pool.close()
    blobs.close()


app = FastAPI(title="qwen-image-svc", lifespan=lifespan)


def require_token(authorization: Optional[str] = Header(default=None)) -> None:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token, AUTH_TOKEN):
        raise Unauthorized("missing or invalid bearer token")


@app.exception_handler(ServiceError)
async def _service_error(_: Request, exc: ServiceError) -> JSONResponse:
    headers = {}
    if exc.retry_after:
        headers["Retry-After"] = str(exc.retry_after)
    body = ErrorBody(message=exc.message, type=exc.type, code=str(exc.status))
    return JSONResponse(body.model_dump(), status_code=exc.status, headers=headers)


@app.exception_handler(HTTPException)
async def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
    body = ErrorBody(message=str(exc.detail), code=str(exc.status_code))
    return JSONResponse(body.model_dump(), status_code=exc.status_code)


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
    body = ErrorBody(message="internal error", type="internal_error", code="500")
    return JSONResponse(body.model_dump(), status_code=500)


# ------------------------------------------------------------------ 响应组装

def _job_payload(job) -> JobStatus:
    data = []
    if job.status == "succeeded":
        for blob, seed in zip(job.blobs, job.seeds):
            data.append(_datum_from_format(job.response_format, blob, seed))
    return JobStatus(
        job_id=job.job_id,
        status=job.status,
        created=job.created,
        data=data,
        error=job.error,
    )


def _datum_from_format(fmt: str, blob: str, seed: Optional[int]) -> ImageDatum:
    if fmt == "b64_json":
        import base64

        return ImageDatum(
            b64_json=base64.b64encode(app.state.blobs.get(blob)).decode(), seed=seed
        )
    return ImageDatum(url=f"/files/{blob}", seed=seed)


# ------------------------------------------------------------------ 路由

@app.post(
    "/v1/images/generations",
    response_model=ImageResponse,
    dependencies=[Depends(require_token)],
)
def generate(req: ImageRequest) -> ImageResponse:
    """同步：阻塞到出图。客户端断连时 GPU 仍会跑完当前任务，
    这是不可避免的损失；需要可取消就改用异步入口。"""
    job = app.state.jobs.submit(req.to_task(), req.response_format)
    job = app.state.jobs.wait(job)

    if job.status != "succeeded":
        raise _failed(job)

    data = [_datum_from_format(req.response_format, b, s)
            for b, s in zip(job.blobs, job.seeds)]
    return ImageResponse(created=now(), data=data)


@app.post(
    "/v1/images/jobs", status_code=202,
    response_model=JobAccepted,
    dependencies=[Depends(require_token)],
)
def create_job(req: ImageRequest, request: Request) -> JobAccepted:
    job = app.state.jobs.submit(req.to_task(), req.response_format)
    return JobAccepted(
        job_id=job.job_id,
        poll_url=str(request.url_for("get_job", job_id=job.job_id)),
    )


@app.get(
    "/v1/images/jobs/{job_id}",
    response_model=JobStatus,
    name="get_job",
    dependencies=[Depends(require_token)],
)
def get_job(job_id: str) -> JobStatus:
    return _job_payload(app.state.jobs.get(job_id))


@app.delete(
    "/v1/images/jobs/{job_id}",
    response_model=JobStatus,
    dependencies=[Depends(require_token)],
)
def cancel_job(job_id: str) -> JobStatus:
    return _job_payload(app.state.jobs.cancel(job_id))


@app.get("/files/{name}")
def get_file(name: str) -> Response:
    """blob 下载。内容寻址的 hash 本身不可猜，因此不加鉴权；
    需要私密性时把这条路由移到 require_token 后面即可。

    名字非法与文件不存在都必须映射成 404：这是资源问题，
    不是服务端故障，500 会让客户端误以为该重试。
    """
    from errors import NotFound

    if not _BLOB_NAME.match(name):
        raise NotFound(f"invalid blob name: {name}")
    try:
        return Response(content=app.state.blobs.get(name), media_type="image/png")
    except FileNotFoundError:
        raise NotFound(f"blob {name} not found")
    except OSError as exc:
        from errors import ServiceError as _SE

        raise _SE(f"blob read failed: {exc}")


# ------------------------------------------------------------------ 健康

@app.get("/healthz", response_class=PlainTextResponse)
def healthz() -> str:
    return "ok"


@app.get("/readyz", response_class=PlainTextResponse)
def readyz() -> Response:
    snap = app.state.pool.ready_snapshot()
    ok = all(v == "ready" for v in snap.values())
    lines = [f"{op}={state}" for op, state in snap.items()]
    return PlainTextResponse(
        ("ready" if ok else "not_ready") + "\n" + "\n".join(lines),
        status_code=200 if ok else 503,
    )


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """Prometheus 文本格式。只暴露决定「要不要扩容/排障」的指标。"""
    snap = app.state.pool.ready_snapshot()
    lines = []
    for op, state in snap.items():
        lines.append(f'svc_op_ready{{op="{op}"}} {1 if state == "ready" else 0}')
        lines.append(f'svc_queue_depth{{op="{op}"}} {app.state.pool.queue_depth(op)}')
    return "\n".join(lines) + "\n"


def _failed(job) -> ServiceError:
    from errors import InferenceFailed, InferenceTimeout, Overloaded

    msg = job.error or "unknown"
    if job.status == "expired":
        return Overloaded(msg, retry_after=30)
    if "Timeout" in msg or "timeout" in msg:
        return InferenceTimeout(msg)
    return InferenceFailed(msg)