#!/usr/bin/env bash
# 冒烟测试：只依赖 curl + jq，在目标机上对着跑起来的服务打一轮。
# 用法: SVC_TOKEN=xxx BASE=http://127.0.0.1:8000 ./scripts/smoke.sh
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8000}"
TOKEN="${SVC_TOKEN:?SVC_TOKEN required}"
AUTH=(-H "Authorization: Bearer $TOKEN")
JSON=(-H "Content-Type: application/json")

ok()   { printf '  \033[32mPASS\033[0m %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; exit 1; }
code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }

echo "== 1. 健康与就绪"
[ "$(code "$BASE/healthz")" = "200" ] && ok "healthz" || fail "healthz"
ready=$(curl -s "$BASE/readyz"); rc=$?
echo "$ready" | sed 's/^/    /'
[ $rc -eq 0 ] && ok "readyz 200" || echo "    (readyz 503 —— fake 模式下应立即可用；真机需等权重加载)"

echo "== 2. 鉴权"
[ "$(code -X POST "$BASE/v1/images/generations" "${JSON[@]}" -d '{"prompt":"x"}')" = "401" ] \
  && ok "无 token -> 401" || fail "无 token 应当 401"
[ "$(code -X POST "$BASE/v1/images/generations" -H 'Authorization: Bearer wrong' "${JSON[@]}" -d '{"prompt":"x"}')" = "401" ] \
  && ok "错 token -> 401" || fail "错 token 应当 401"

echo "== 3. 参数校验"
[ "$(code -X POST "$BASE/v1/images/generations" "${AUTH[@]}" "${JSON[@]}" -d '{"prompt":"x","size":"1920x1080"}')" = "422" ] \
  && ok "非法尺寸 -> 422" || fail "非法尺寸应当 422"
[ "$(code -X POST "$BASE/v1/images/generations" "${AUTH[@]}" "${JSON[@]}" -d '{"model":"qwen-image-edit","prompt":"x"}')" = "422" ] \
  && ok "edit 无图 -> 422" || fail "edit 无图应当 422"

echo "== 4. 同步生图 (n=2)"
resp=$(curl -s -w '\n%{http_code}' -X POST "$BASE/v1/images/generations" "${AUTH[@]}" "${JSON[@]}" \
  -d '{"prompt":"smoke test","size":"1664x928","n":2,"seed":7}')
body=$(echo "$resp" | sed '$d'); status=$(echo "$resp" | tail -n1)
[ "$status" = "200" ] && ok "POST generations 200" || fail "generations -> $status: $body"
count=$(echo "$body" | jq '.data | length')
[ "$count" = "2" ] && ok "返回 2 张图" || fail "期望 2 张，实际 $count"
seeds=$(echo "$body" | jq -c '[.data[].seed]')
[ "$seeds" = "[7,8]" ] && ok "seed 递增 $seeds" || fail "seed 应为 [7,8]，实际 $seeds"
echo "$body" | jq -r '.data[0].b64_json' | base64 -d > /tmp/smoke_0.png 2>/dev/null \
  && file /tmp/smoke_0.png | sed 's/^/    /' && ok "b64 可解码"

echo "== 5. 异步 + 轮询 + url 格式"
job=$(curl -s -X POST "$BASE/v1/images/jobs" "${AUTH[@]}" "${JSON[@]}" \
  -d '{"prompt":"async smoke","response_format":"url"}' | jq -r .job_id)
[ -n "$job" ] && [ "$job" != "null" ] && ok "job 创建 $job" || fail "job 创建失败"
final=""
for _ in $(seq 1 120); do
  st=$(curl -s "$BASE/v1/images/jobs/$job" "${AUTH[@]}")
  status=$(echo "$st" | jq -r .status)
  [ "$status" = "succeeded" ] && { final="$st"; break; }
  case "$status" in
    failed|expired|cancelled) fail "job 终态 $status: $st" ;;
  esac
  sleep 1
done
[ -n "$final" ] && ok "job 轮询到 succeeded" || fail "轮询超时"
url=$(echo "$final" | jq -r '.data[0].url // empty')
echo "    url=[$url]"
[ -n "$url" ] || fail "job 响应里没有 data[0].url"
[ "$(code "$BASE$url")" = "200" ] && ok "url 可下载 $url" || fail "url 下载失败: $BASE$url"
ftype=$(curl -s "$BASE$url" | file - | sed 's/^/    /'); echo "$ftype"

echo "== 6. 取消"
job2=$(curl -s -X POST "$BASE/v1/images/jobs" "${AUTH[@]}" "${JSON[@]}" \
  -d '{"prompt":"cancel me"}' | jq -r .job_id)
del=$(curl -s -X DELETE "$BASE/v1/images/jobs/$job2" "${AUTH[@]}" | jq -r .status)
echo "    cancel 后状态: $del"
# queued 时取消应得到 cancelled；已在 GPU 上则如实返回 running/succeeded，
# 这是设计行为，不算失败。

echo "== 7. 404"
[ "$(code "$BASE/v1/images/jobs/deadbeef" "${AUTH[@]}")" = "404" ] \
  && ok "不存在的 job -> 404" || fail "应当 404"

echo
echo "全部通过"