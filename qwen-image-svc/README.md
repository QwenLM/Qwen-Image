# qwen-image-svc

基于 Qwen Image pipeline 的原生 HTTP 服务：同步和异步任务共用同一套 GPU worker、队列和任务状态机。
接口字段直接对应 Qwen diffusers 参数，不提供 OpenAI 兼容路径或响应格式。

## 部署机器要求

- 2 × 80 GiB NVIDIA 卡（A100/H100 级别）
- Python ≥ 3.10
- **uvicorn 必须单进程**：`--workers 1`。GPU 池是 API 进程的子进程，多开会重复加载权重。

```bash
cd qwen-image-svc
pip install -r requirements.txt
export SVC_TOKEN=$(openssl rand -hex 32)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

API 启动后立即监听，权重在 worker 子进程中异步加载；两个 worker 都 ready 前，`/readyz` 返回 503。

## GPU 分配

| GPU | op | 模型 |
|---|---|---|
| 0 | t2i | `Qwen/Qwen-Image` |
| 1 | edit | `Qwen/Qwen-Image-Edit` |

修改 `pool.GPU_ASSIGNMENT` 可以调整卡位。部署机也可以通过 `T2I_REPO` 和 `EDIT_REPO` 覆盖模型路径。

## 原生接口

同步和异步请求使用相同的请求体。`model` 是唯一的模型分派字段：

- `qwen-image`：文生图，必须使用 `width` 和 `height`
- `qwen-image-edit`：图像编辑，必须使用 `images`

### 同步生成

```bash
curl -s http://127.0.0.1:8000/generate \
  -H "Authorization: Bearer $SVC_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen-image",
    "prompt": "一只穿西装的卡皮巴拉",
    "negative_prompt": "低质量",
    "seed": 42,
    "width": 1664,
    "height": 928,
    "true_cfg_scale": 4.0,
    "guidance_scale": 1.0,
    "num_inference_steps": 50,
    "num_images_per_prompt": 1
  }'
```

响应：

```json
{
  "images": [
    {"url": "/files/<sha256>.png", "seed": 42}
  ],
  "elapsed_ms": 12345
}
```

### 同步编辑

`images` 是一个或多个输入图片的 base64 字符串列表。

```bash
curl -s http://127.0.0.1:8000/generate \
  -H "Authorization: Bearer $SVC_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen-image-edit",
    "images": ["<base64-image>"],
    "prompt": "把衣服改成蓝色",
    "seed": 42,
    "true_cfg_scale": 4.0,
    "guidance_scale": 1.0,
    "num_inference_steps": 50,
    "num_images_per_prompt": 1
  }'
```

### 异步生成或编辑

异步入口是 `/jobs`，请求体与 `/generate` 完全相同。

```bash
JOB=$(curl -s http://127.0.0.1:8000/jobs \
  -H "Authorization: Bearer $SVC_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen-image",
    "prompt": "一只穿西装的卡皮巴拉",
    "width": 1664,
    "height": 928,
    "seed": 42
  }' | jq -r .job_id)

curl -s "http://127.0.0.1:8000/jobs/$JOB" \
  -H "Authorization: Bearer $SVC_TOKEN"

curl -X DELETE "http://127.0.0.1:8000/jobs/$JOB" \
  -H "Authorization: Bearer $SVC_TOKEN"
```

结果图片通过 `/files/<sha256>.png` 下载，Blob 默认保留 24 小时。

## 运维接口

- `GET /healthz`：进程存活检查
- `GET /readyz`：两个模型是否完成加载
- `GET /metrics`：worker ready 状态和队列深度

## 参数与错误

请求额外字段会被拒绝，避免悄悄接受 OpenAI 兼容参数。有效的原生字段包括：

- `prompt`
- `negative_prompt`
- `seed`
- `true_cfg_scale`
- `guidance_scale`
- `num_inference_steps`
- `num_images_per_prompt`
- 文生图专用：`width`、`height`
- 编辑专用：`images`

服务错误类型：

- `401`：鉴权失败
- `404`：任务或 Blob 不存在
- `429`：GPU 队列已满
- `503`：排队超时或服务过载
- `500`：推理失败
- `504`：推理超时

## 测试

协议、Blob 和任务状态机测试不需要 GPU：

```bash
pytest qwen-image-svc/tests/ -v
```

启动 fake worker 后，可以验证完整 HTTP 链路：

```bash
cd qwen-image-svc
export SVC_TOKEN=dev-token
export SVC_FAKE=1
uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1

SVC_TOKEN=dev-token \
  BASE=http://127.0.0.1:8000 \
  ./scripts/smoke.sh
```

## 已知取舍

- `num_images_per_prompt > 1` 会串行执行，避免多个激活值同时占用显存。
- 任务进入 GPU 后不可中断；删除请求只保证排队中的任务取消。
- 任务表目前在内存中，服务重启会丢失未完成任务。