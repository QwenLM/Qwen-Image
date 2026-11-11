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

| GPU | op | 默认本地权重 |
|---|---|---|
| 0, 1 | t2i | `/data/models/Qwen-Image-2512` |
| 2, 3 | edit | `/data/models/Qwen-Image-Edit-2511` |
| 4, 5 | z_image | `/data/models/Z-Image` |
| 6, 7 | z_image_turbo | `/data/models/Z-Image-Turbo` |

默认四模型各 2 张卡，权重只读本地目录（`local_files_only=True`，不会上网下载）。可用环境变量覆盖卡位与路径：

```bash
export T2I_GPUS=0,1
export EDIT_GPUS=2,3
export Z_IMAGE_GPUS=4,5
export Z_IMAGE_TURBO_GPUS=6,7

# 仅当目录不在默认位置时才需要改
# export T2I_REPO=/data/models/Qwen-Image-2512
```

## DashScope 兼容接口

服务可以使用本地部署的模型提供 DashScope 风格的同步接口，不会请求 DashScope 云端：

```text
POST /api/v1/services/aigc/multimodal-generation/generation
```

请求使用 DashScope 的 `model`、`input.messages` 和 `parameters` 字段。模型名映射到本地 worker：

- `Qwen-Image-2512`：本地文生图模型
- `Qwen-Image-Edit-2511`：本地图像编辑模型
- `Z-Image` / `Z-Image-Turbo`：本地 Z-Image 文生图（官方参数：`size` / `prompt_extend` / `seed`，`n` 固定为 1）

```bash
curl -s http://127.0.0.1:8000/api/v1/services/aigc/multimodal-generation/generation \\
  -H "Authorization: Bearer $SVC_TOKEN" \\
  -H 'Content-Type: application/json' \\
  -d '{
    "model": "Qwen-Image-2512",
    "input": {
      "messages": [{
        "role": "user",
        "content": [{"text": "一只穿西装的卡皮巴拉"}]
      }]
    },
    "parameters": {
      "size": "1664*928",
      "n": 1,
      "seed": 42
    }
  }'
```

Z-Image-Turbo 示例：

```bash
curl -s http://127.0.0.1:8000/api/v1/services/aigc/multimodal-generation/generation \
  -H "Authorization: Bearer $SVC_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Z-Image-Turbo",
    "input": {
      "messages": [{
        "role": "user",
        "content": [{"text": "一只坐着的橘黄色的猫"}]
      }]
    },
    "parameters": {
      "size": "1024*1536",
      "prompt_extend": false,
      "seed": 42
    }
  }'
```

响应格式为 DashScope 风格，图片实际由本服务的 `/files/` 提供。若服务通过反向代理暴露，设置 `SVC_PUBLIC_BASE_URL` 为外部可访问的完整地址，避免返回相对 URL。

图像编辑时，在同一个 `content` 数组中传入 `image` 和 `text`；图片支持 `data:image/...;base64,...`，或可访问的公网 HTTP(S) URL。远程图片仅允许解析到公网地址，单张最大 10 MiB。


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
    "num_inference_steps": 25,
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
    "num_inference_steps": 25,
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
- `429`：该 op 的排队名额已满（在跑 + 等待超过 `卡数 × QUEUE_DEPTH_PER_GPU`）；未满时请求会排队等 GPU，不会因「卡正忙」立刻 429
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