#!/bin/sh
# Bound Docker disk use on runner hosts (fleet policy RM #2071, 2026-10-11).
# Keeps recent build cache (CI layer reuse) and any image used in the last 7 days.
# Images labelled keep=true (UpstreamDrift profile / explorer / ud-sim) are spared
# (UD #12878, #12941; see DOCKER_SETUP.md and deploy/runner-cleanup.sh in
# Runner_Dashboard).
KEEP_BUILD=${KEEP_BUILD:-40GB}
docker container prune -f --filter "until=24h" >/dev/null
docker builder prune -f --filter "until=72h" >/dev/null
docker builder prune -f --keep-storage "$KEEP_BUILD" >/dev/null
docker image prune -af --filter "until=168h" --filter "label!=keep" >/dev/null
logger -t docker-cache-prune "$(docker system df --format '{{.Type}}={{.Size}}' | tr '\n' ' ')"
