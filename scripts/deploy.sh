#!/usr/bin/env bash
set -euo pipefail

# Deploy script for bootstrap-template
# This script handles building, saving, uploading, and starting the Docker container.
# It also checks the public endpoint and prints diagnostics on failure.

REMOTE_HOST="test.k3rnel-pan1c.com"
REMOTE_PORT=2223
REMOTE_USER="marko"
IMAGE_NAME="speech-agent"
REMOTE="$REMOTE_USER@$REMOTE_HOST"
SSH_OPTS=(-p "$REMOTE_PORT" -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3)
: "${PUBLIC_URL:?PUBLIC_URL must be set}"
: "${LLM_BASE_URL:?LLM_BASE_URL must be set}"
: "${LLM_API_KEY:?LLM_API_KEY must be set}"
: "${API_KEY:?API_KEY must be set}"
: "${PORT:?PORT must be set}"
: "${AUTH_PASSWORD:?AUTH_PASSWORD must be set}"

print_remote_diagnostics() {
  ssh "${SSH_OPTS[@]}" "$REMOTE" '
    docker ps -a --filter "name='"$IMAGE_NAME"'" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo
    docker logs --tail 80 '"$IMAGE_NAME"' 2>&1 | tail -80 || true
  ' || true
}

printf "==> building image ($IMAGE_NAME, linux/amd64)..."
if [ "${SKIP_DOCKER_BUILD:-0}" != "1" ]; then
  ./scripts/build.sh linux/amd64 > /dev/null 2>&1
fi
echo "ok"

printf "==> saving image..."
docker save "$IMAGE_NAME" | gzip > /tmp/"${IMAGE_NAME}".tar.gz
echo "ok"

printf "==> uploading to $REMOTE_HOST..."
scp -P "$REMOTE_PORT" -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 /tmp/"${IMAGE_NAME}".tar.gz "$REMOTE":/tmp/"${IMAGE_NAME}".tar.gz
rm /tmp/"${IMAGE_NAME}".tar.gz
echo "ok"

printf "==> loading image on remote..."
ssh "${SSH_OPTS[@]}" "$REMOTE" '
  docker load < /tmp/'"${IMAGE_NAME}"'.tar.gz
  rm /tmp/'"${IMAGE_NAME}"'.tar.gz
' > /dev/null 2>&1
echo "ok"

printf "==> ensuring remote data dir..."
ssh "${SSH_OPTS[@]}" "$REMOTE" 'mkdir -p "$HOME/.bootstrap-template/data"' > /dev/null 2>&1
echo "ok"
printf "==> starting container..."
printf -v image_name_q '%q' "$IMAGE_NAME"
printf -v port_q '%q' "$PORT"
printf -v llm_base_url_q '%q' "$LLM_BASE_URL"
printf -v llm_api_key_q '%q' "$LLM_API_KEY"
printf -v api_key_q '%q' "$API_KEY"
printf -v auth_password_q '%q' "$AUTH_PASSWORD"
printf -v llm_model_q '%q' "${LLM_MODEL:-}"
printf -v asr_model_q '%q' "${ASR_MODEL:-}"
printf -v mistral_api_key_q '%q' "${MISTRAL_API_KEY:-}"
printf -v tts_model_q '%q' "${TTS_MODEL:-}"
printf -v tts_voice_q '%q' "${TTS_VOICE:-}"
printf -v asr_language_q '%q' "${ASR_LANGUAGE:-}"
printf -v barge_in_threshold_q '%q' "${BARGE_IN_THRESHOLD_RMS:-}"
printf -v barge_in_min_ms_q '%q' "${BARGE_IN_MIN_MS:-}"
if ! container_id=$(ssh "${SSH_OPTS[@]}" "$REMOTE" 'bash -se' <<EOF
set -euo pipefail
image_name=$image_name_q
port=$port_q
llm_base_url=$llm_base_url_q
llm_api_key=$llm_api_key_q
api_key=$api_key_q
auth_password=$auth_password_q
llm_model=$llm_model_q
asr_model=$asr_model_q
mistral_api_key=$mistral_api_key_q
tts_model=$tts_model_q
tts_voice=$tts_voice_q
asr_language=$asr_language_q
barge_in_threshold=$barge_in_threshold_q
barge_in_min_ms=$barge_in_min_ms_q
data_dir="\$HOME/.bootstrap-template/data"

extra_env=()
if [ -n "\$llm_model" ]; then extra_env+=(-e LLM_MODEL="\$llm_model"); fi
if [ -n "\$asr_model" ]; then extra_env+=(-e ASR_MODEL="\$asr_model"); fi
if [ -n "\$asr_language" ]; then extra_env+=(-e ASR_LANGUAGE="\$asr_language"); fi
if [ -n "\$mistral_api_key" ]; then extra_env+=(-e MISTRAL_API_KEY="\$mistral_api_key"); fi
if [ -n "\$tts_model" ]; then extra_env+=(-e TTS_MODEL="\$tts_model"); fi
if [ -n "\$tts_voice" ]; then extra_env+=(-e TTS_VOICE="\$tts_voice"); fi
if [ -n "\$barge_in_threshold" ]; then extra_env+=(-e BARGE_IN_THRESHOLD_RMS="\$barge_in_threshold"); fi
if [ -n "\$barge_in_min_ms" ]; then extra_env+=(-e BARGE_IN_MIN_MS="\$barge_in_min_ms"); fi

docker stop -t 2 "\$image_name" > /dev/null 2>&1 || true
docker rm "\$image_name" > /dev/null 2>&1 || true
docker run -d \
  -p "\$port:\$port" \
  -e PORT="\$port" \
  -e LLM_BASE_URL="\$llm_base_url" \
  -e LLM_API_KEY="\$llm_api_key" \
  -e API_KEY="\$api_key" \
  -e AUTH_MODE=password \
  -e AUTH_PASSWORD="\$auth_password" \
  -e SESSIONS_PATH=/data/sessions.json \
  -e REQUEST_LOG_PATH=/data/requests.log \
  -e DOWNLOADS_DIR=/data/downloads \
  \${extra_env[@]+"\${extra_env[@]}"} \
  -v "\$data_dir:/data" \
  --name "\$image_name" \
  --restart unless-stopped \
  "\$image_name"
EOF
)
then
  echo "FAIL"
  echo "    remote container start failed"
  echo "    remote diagnostics:"
  print_remote_diagnostics
  exit 1
fi
echo "started (${container_id:0:12})"

printf "==> waiting for server..."
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-120}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-2}"
WAIT_DEADLINE=$(( $(date +%s) + WAIT_TIMEOUT_SECONDS ))
server_ready=false

while (( $(date +%s) < WAIT_DEADLINE )); do
  if ssh "${SSH_OPTS[@]}" "$REMOTE" 'curl -sf --max-time 3 http://localhost:'"$PORT"' > /dev/null' 2>/dev/null; then
    server_ready=true
    break
  fi
  sleep "$WAIT_INTERVAL_SECONDS"
done

if [[ "$server_ready" != true ]]; then
  echo "FAIL"
  echo "    server did not start within ${WAIT_TIMEOUT_SECONDS}s"
  echo "    remote diagnostics:"
  print_remote_diagnostics
  exit 1
fi
echo "server reachable"

printf "==> checking public endpoint ($PUBLIC_URL)..."
if ! body=$(curl -sfL --max-time 10 "$PUBLIC_URL"); then
  echo "FAIL"
  echo "    could not reach $PUBLIC_URL"
  exit 1
fi

if ! echo "$body" | grep -qE "hello|Sign in"; then
  echo "FAIL"
  echo "    $PUBLIC_URL response did not look right"
  echo "    $body"
  exit 1
fi
echo "ok"

./scripts/get_logs.sh

echo "==> deployed $IMAGE_NAME to $PUBLIC_URL"
