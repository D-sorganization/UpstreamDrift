#!/usr/bin/env bash
# Smoke test for the runtime API entrypoint (#2786).
#
# Builds the runtime image, starts a container using the default CMD, and
# verifies that /health returns 200 within a short timeout. This is the
# regression gate that salvages the intent of stale PR #2723: a plain
# `docker run <image>` must produce a live API (not an interactive shell).
#
# Usage:
#   scripts/smoke_test_docker_entrypoint.sh [image-tag]
#
# Exits 0 on success, non-zero otherwise. Requires Docker.

set -euo pipefail

IMAGE_TAG="${1:-upstream-drift-runtime:smoke}"
CONTAINER_NAME="ud-smoke-$$"
HOST_PORT="${SMOKE_HOST_PORT:-18001}"
HEALTH_TIMEOUT_SECONDS="${SMOKE_HEALTH_TIMEOUT:-60}"

cleanup() {
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[smoke] Building runtime image ${IMAGE_TAG}..."
docker build --target runtime -t "${IMAGE_TAG}" .

echo "[smoke] Starting container ${CONTAINER_NAME} on host port ${HOST_PORT}..."
docker run -d --rm \
    --name "${CONTAINER_NAME}" \
    -p "${HOST_PORT}:8001" \
    "${IMAGE_TAG}" >/dev/null

echo "[smoke] Waiting up to ${HEALTH_TIMEOUT_SECONDS}s for /health to respond..."
deadline=$(( $(date +%s) + HEALTH_TIMEOUT_SECONDS ))
while :; do
    if curl -fsS "http://127.0.0.1:${HOST_PORT}/health" >/dev/null 2>&1; then
        echo "[smoke] OK — /health responded successfully."
        exit 0
    fi
    if [ "$(date +%s)" -ge "${deadline}" ]; then
        echo "[smoke] FAIL — /health did not respond within ${HEALTH_TIMEOUT_SECONDS}s."
        echo "[smoke] Container logs:"
        docker logs "${CONTAINER_NAME}" || true
        exit 1
    fi
    sleep 2
done
