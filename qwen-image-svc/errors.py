"""错误语义：内部异常 -> HTTP 状态码的唯一定义处。

API 层只做映射，不做判断；jobs/pool 只 raise 这里的异常类型，
不 import fastapi。这样错误码只有一张表，不会在 handler 里散落。
"""
from __future__ import annotations


class ServiceError(Exception):
    status = 500
    type = "internal_error"

    def __init__(self, message: str, *, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after


class InvalidRequest(ServiceError):
    status = 400
    type = "invalid_request_error"


class Unauthorized(ServiceError):
    status = 401
    type = "authentication_error"


class NotFound(ServiceError):
    status = 404
    type = "not_found"


class QueueFull(ServiceError):
    """客户端该退避重试。"""

    status = 429
    type = "rate_limit_exceeded"


class Overloaded(ServiceError):
    """排队超时 —— 服务饱和，不是客户端的错。"""

    status = 503
    type = "server_overloaded"


class InferenceFailed(ServiceError):
    status = 500
    type = "inference_error"


class InferenceTimeout(ServiceError):
    status = 504
    type = "inference_timeout"