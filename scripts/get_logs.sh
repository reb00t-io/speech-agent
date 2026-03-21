#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="test.k3rnel-pan1c.com"
REMOTE_PORT=2223
REMOTE_USER="marko"
IMAGE_NAME="speech-agent"
REMOTE="$REMOTE_USER@$REMOTE_HOST"
SSH_OPTS=(-p "$REMOTE_PORT" -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3)

LOG_FETCH_RETRIES="${LOG_FETCH_RETRIES:-3}"
LOG_FETCH_RETRY_DELAY_SECONDS="${LOG_FETCH_RETRY_DELAY_SECONDS:-2}"

logs=""

print_attempt_message() {
    local attempt="$1"
    if [ "$attempt" -eq 1 ]; then
        printf "==> fetching container logs..."
        return
    fi
    printf "==> fetching container logs (retry %s/%s)..." "$attempt" "$LOG_FETCH_RETRIES"
}

fetch_logs_once() {
    ssh "${SSH_OPTS[@]}" "$REMOTE" "docker logs -t \"$IMAGE_NAME\"" 2>&1
}

fetch_logs_with_retries() {
    local attempt
    for ((attempt = 1; attempt <= LOG_FETCH_RETRIES; attempt++)); do
        print_attempt_message "$attempt"

        if logs=$(fetch_logs_once); then
            echo "ok"
            return 0
        fi

        echo "skipped"
        if [ "$attempt" -lt "$LOG_FETCH_RETRIES" ]; then
            sleep "$LOG_FETCH_RETRY_DELAY_SECONDS"
        fi
    done

    return 1
}

if fetch_logs_with_retries; then
    if [ -n "$logs" ]; then
        echo "$logs" | sed 's/^/    /'
    fi
else
    if [ -n "$logs" ]; then
        echo "    failed to fetch logs after $LOG_FETCH_RETRIES attempts"
        echo "$logs" | sed 's/^/    /'
    fi
fi
