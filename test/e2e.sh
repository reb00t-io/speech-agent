#!/usr/bin/env bash
set -euo pipefail

: "${PORT:?PORT must be set}"

mkdir -p "${HOME}/.bootstrap-template/data"

if [ "${SKIP_DOCKER_BUILD:-0}" != "1" ]; then
  ./scripts/build.sh
fi
docker compose up -d
trap 'docker compose down' EXIT

echo "waiting for server..."
wait_timeout_seconds=120
wait_interval_seconds=2
deadline=$((SECONDS + wait_timeout_seconds))
attempt=0
last_status=""

while (( SECONDS < deadline )); do
  attempt=$((attempt + 1))
  status=$(curl -sS -o /dev/null -w "%{http_code}" "http://localhost:${PORT}" || true)
  last_status="$status"

  if [ "$status" = "200" ]; then
    echo "server is up (attempt ${attempt})"
    break
  fi

  if [[ "$status" == 5* ]]; then
    echo "FAIL: server returned HTTP ${status} while starting (attempt ${attempt})"
    docker compose logs --tail 50 || true
    exit 1
  fi

  if [ -z "$status" ] || [ "$status" = "000" ]; then
    echo "waiting... attempt ${attempt}/${wait_timeout_seconds}s (server not reachable yet)"
  else
    echo "waiting... attempt ${attempt}/${wait_timeout_seconds}s (HTTP ${status})"
  fi

  sleep "$wait_interval_seconds"
done

if [ "$last_status" != "200" ]; then
  echo "FAIL: server did not become ready within ${wait_timeout_seconds}s (last status: ${last_status:-none})"
  docker compose logs --tail 50 || true
  exit 1
fi

echo "checking response..."
body=$(curl -sf http://localhost:"$PORT")

if ! echo "$body" | grep -q "hello"; then
  echo "FAIL: response does not contain 'hello'"
  echo "$body"
  exit 1
fi

echo "checking deploy date..."
deploy_date=$(echo "$body" | sed -n 's/.*deployed \([^)]*\).*/\1/p')
if deploy_ts=$(date -u -d "$deploy_date" +%s 2>/dev/null); then
  : # GNU date (Linux)
elif deploy_ts=$(date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "$deploy_date" +%s 2>/dev/null); then
  : # BSD date (macOS)
else
  echo "FAIL: could not parse deploy date: ${deploy_date}"
  exit 1
fi
now_ts=$(date -u +%s)
age=$(( now_ts - deploy_ts ))

if [ "$age" -gt 300 ]; then
  echo "FAIL: deploy date is ${age}s old (max 300s)"
  exit 1
fi

echo "deploy date is ${age}s old, ok"
echo "e2e test passed"
