# qwen-image-svc

对外生图接口：同步 + 异步轮询双入口，静态 Bearer token。

## 部署机器要求

- 2 × 80 GiB NVIDIA 卡（A100/H100 级别）
- Python ≥ 3.10（`dataclass(slots=True)`）
- **uvicorn 必须单进程**：`--workers 1`。GPU 池是 API 进程的子进程，
  多开会让每份都加载一遍 ~60 GiB 权重。

```bash
pip install -r requirements.txt
export SVC_TOKEN=$(openssl rand -hex 32)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

启动时 API 立即可用，权重在子进程里异步加载；`/readyz` 返回 503
直到两个 op 都 ready。加载失败会显示 `init_failed: <原因>`，不会假装健康。

## 卡位分配

| GPU | op | 模型 |
|---|---|---|
| 0 | t2i | `Qwen/Qwen-Image` |
| 1 | edit | `Qwen/Qwen-Image-Edit` |

改容量只改 `pool.GPU_ASSIGNMENT`。换权重路径用 `T2I_REPO` / `EDIT_REPO` 环境变量。

## 接口

```bash
# 同步
curl -s localhost:8000/v1/images/generations \
  -H "Authorization: Bearer $SVC_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"一只穿西装的卡皮巴拉","size":"1664x928","n":2}'

# 异步
JOB=$(curl -s localhost:8000/v1/images/jobs \
  -H "Authorization: Bearer $SVC_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"...","response_format":"url"}' | jq -r .job_id)

curl -s localhost:8000/v1/images/jobs/$JOB -H "Authorization: Bearer $SVC_TOKEN"
curl -X DELETE localhost:8000/v1/images/jobs/$JOB -H "Authorization: Bearer $SVC_TOKEN"

# 编辑：model 换成 qwen-image-edit 并带 image_b64（≤3 张）
```

`response_format=url` 返回 `/files/<sha256>.png`，内容寻址、默认 24h TTL。

## 错误码

| 场景 | 状态码 | type |
|---|---|---|
| 参数非法 / 尺寸不在 preset | 400 | invalid_request_error |
| token 错误 | 401 | authentication_error |
| job 不存在或过期 | 404 | not_found |
| 队列满（两卡均占用且有 2 个排队） | 429 | rate_limit_exceeded |
| 排队超过 120s | 503 | server_overloaded |
| 推理失败（含 OOM） | 500 | inference_error |
| 单图超过 300s | 504 | inference_timeout |

## 超时预算

两个独立预算，不要合并：

- `SVC_QUEUE_DEADLINE=120` — 排队上限，超了 503 + Retry-After
- `SVC_INFER_TIMEOUT=300` — 已进 GPU 的上限，超了视为卡死

## 测试（无需 GPU）

```bash
pytest tests/ -v
```

覆盖协议校验、blob 内容寻址与去重、任务状态机全部终态。

## 已知取舍

- `n>1` 串行生成，不 batch：80 GiB 卡剩 ~20 GiB，并发激活值会 OOM。
- 进入 GPU 后的任务不可取消（pipeline 无 mid-step 取消点），`DELETE` 只对排队中的任务生效。
- 同步请求断连后 GPU 仍会跑完当前任务；需要可取消就用异步入口。
- 任务表在内存里，重启会丢排队任务。多副本部署前必须先把它换成外部存储。