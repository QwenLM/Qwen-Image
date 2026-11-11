#!/usr/bin/env bash
# 原生 Qwen HTTP 冒烟测试：只依赖 curl + jq + file。
# 用法: SVC_TOKEN=xxx BASE=http://127.0.0.1:8000 ./scripts/smoke.sh
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8000}"
TOKEN="${SVC_TOKEN:?SVC_TOKEN required}"
AUTH=(-H "Authorization: Bearer $TOKEN")
JSON=(-H "Content-Type: application/json")

ok()   { printf '  \033[32mPASS\033[0m %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; exit 1; }
code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }

printf '%s\n' '== 1. 健康与兼容路由'
[ "$(code "$BASE/healthz")" = "200" ] && ok "healthz" || fail "healthz"
[ "$(code "$BASE/v1/images/generations")" = "404" ] \
  && ok "OpenAI 兼容路由不存在" || fail "兼容路由不应存在"

printf '%s\n' '== 2. 鉴权'
[ "$(code -X POST "$BASE/generate" "${JSON[@]}" -d '{"model":"qwen-image","prompt":"x"}')" = "401" ] \
  && ok "无 token -> 401" || fail "无 token 应当 401"
[ "$(code -X POST "$BASE/generate" -H 'Authorization: Bearer wrong' "${JSON[@]}" -d '{"model":"qwen-image","prompt":"x"}')" = "401" ] \
  && ok "错 token -> 401" || fail "错 token 应当 401"

printf '%s\n' '== 3. 原生参数校验'
[ "$(code -X POST "$BASE/generate" "${AUTH[@]}" "${JSON[@]}" \
  -d '{"model":"qwen-image","prompt":"x","size":"1664x928"}')" = "422" ] \
  && ok "OpenAI size 字段被拒绝" || fail "size 不应被接受"
[ "$(code -X POST "$BASE/generate" "${AUTH[@]}" "${JSON[@]}" \
  -d '{"model":"qwen-image-edit","prompt":"x"}')" = "422" ] \
  && ok "edit 缺少 images -> 422" || fail "edit 缺图应当 422"

printf '%s\n' '== 4. 同步文生图'
resp=$(curl -s -w '\n%{http_code}' -X POST "$BASE/generate" "${AUTH[@]}" "${JSON[@]}" \
  -d '{
    "model":"qwen-image",
    "prompt":"smoke test",
    "seed":7,
    "width":1664,
    "height":928,
    "true_cfg_scale":4.0,
    "guidance_scale":1.0,
    "num_inference_steps":10,
    "num_images_per_prompt":2
  }')
body=$(printf '%s\n' "$resp" | sed '$d'); status=$(printf '%s\n' "$resp" | tail -n1)
[ "$status" = "200" ] && ok "POST /generate -> 200" || fail "/generate -> $status: $body"
count=$(printf '%s' "$body" | jq '.images | length')
[ "$count" = "2" ] && ok "返回 2 张图" || fail "期望 2 张，实际 $count"
seeds=$(printf '%s' "$body" | jq -c '[.images[].seed]')
[ "$seeds" = "[7,8]" ] && ok "seed 递增 $seeds" || fail "seed 应为 [7,8]，实际 $seeds"
url=$(printf '%s' "$body" | jq -r '.images[0].url')
[ "$(code "$BASE$url")" = "200" ] && ok "结果 URL 可下载" || fail "结果 URL 下载失败"

printf '%s\n' '== 5. 同步图像编辑'
edit=$(curl -s -X POST "$BASE/generate" "${AUTH[@]}" "${JSON[@]}" \
  -d '{
    "model":"qwen-image-edit",
    "images":["AAAA"],
    "prompt":"make it blue",
    "seed":9,
    "true_cfg_scale":4.0,
    "guidance_scale":1.0,
    "num_inference_steps":10
  }')
[ "$(printf '%s' "$edit" | jq -r '.images[0].seed')" = "9" ] \
  && ok "edit 使用原生 images 参数" || fail "edit 请求失败: $edit"

printf '%s\n' '== 6. DashScope 兼容同步接口'
dashscope=$(curl -s -w '\\n%{http_code}' -X POST \\
  "$BASE/api/v1/services/aigc/multimodal-generation/generation" \\
  "${AUTH[@]}" "${JSON[@]}" \\
  -d '{
    "model":"Qwen-Image-2512",
    "input":{"messages":[{"role":"user","content":[{"text":"dashscope smoke"}]}]},
    "parameters":{"size":"1664*928","n":2,"seed":11,"num_inference_steps":10}
  }')
body=$(printf '%s\n' "$dashscope" | sed '$d'); status=$(printf '%s\n' "$dashscope" | tail -n1)
[ "$status" = "200" ] && ok "DashScope 入口 -> 200" || fail "DashScope 入口 -> $status: $body"
count=$(printf '%s' "$body" | jq '.output.choices[0].message.content | length')
[ "$count" = "2" ] && ok "DashScope 返回 2 张图" || fail "DashScope 期望 2 张，实际 $count"
url=$(printf '%s' "$body" | jq -r '.output.choices[0].message.content[0].image')
[ "$(code "$url")" = "200" ] && ok "DashScope 图片 URL 可下载" || fail "DashScope 图片 URL 下载失败"

printf '%s\n' '== 7. 异步任务'
job=$(curl -s -X POST "$BASE/jobs" "${AUTH[@]}" "${JSON[@]}" \
  -d '{
    "model":"qwen-image",
    "prompt":"async smoke",
    "width":1664,
    "height":928,
    "num_inference_steps":10
  }' | jq -r .job_id)
[ -n "$job" ] && [ "$job" != "null" ] && ok "job 创建 $job" || fail "job 创建失败"
final=""
for _ in $(seq 1 120); do
  st=$(curl -s "$BASE/jobs/$job" "${AUTH[@]}")
  status=$(printf '%s' "$st" | jq -r .status)
  [ "$status" = "succeeded" ] && { final="$st"; break; }
  case "$status" in
    failed|expired|cancelled) fail "job 终态 $status: $st" ;;
  esac
  sleep 1
done
[ -n "$final" ] && ok "job 轮询到 succeeded" || fail "轮询超时"
url=$(printf '%s' "$final" | jq -r '.images[0].url')
[ "$(code "$BASE$url")" = "200" ] && ok "异步结果 URL 可下载" || fail "异步结果下载失败"

echo
echo "全部通过"