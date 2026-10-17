#!/bin/sh
# One-command launcher for the Model Explorer containers (UD #12878 D).
#
#   scripts/docker/explorer.sh headless [export ARGS...]   MP4 export, no display
#   scripts/docker/explorer.sh gui                         noVNC at localhost:6080
#   scripts/docker/explorer.sh gui-wslg                    native window via WSLg
#   scripts/docker/explorer.sh engines | probe-gl | build
#   scripts/docker/explorer.sh ensure   rebuild any missing image (health check)
#
# The image build verifies the pinned Tools tree against two build args
# (TOOLS_GITLINK_SHA, TOOLS_SOURCE_SHA256); this wrapper computes them so the
# plain `docker compose` commands in docs/development/DOCKER_SETUP.md work.
# Run it from the repository root, inside the WSL distro that owns the daemon.
set -eu

die() {
    echo "ERROR: $*" >&2
    exit 64
}

[ -f docker-compose.yml ] || die "run from the repository root (docker-compose.yml not found)"
[ -d vendor/ud-tools/src/shared ] ||
    die "vendor/ud-tools is not materialised; run: git submodule update --init vendor/ud-tools"

if [ -z "${TOOLS_GITLINK_SHA:-}" ]; then
    TOOLS_GITLINK_SHA="$(git rev-parse HEAD:vendor/ud-tools 2>/dev/null)" ||
        die "cannot read the vendor/ud-tools gitlink; export TOOLS_GITLINK_SHA=<40-hex sha> explicitly"
fi
if [ -z "${TOOLS_SOURCE_SHA256:-}" ]; then
    TOOLS_SOURCE_SHA256="$(python3 scripts/packaging/pinned_tools_provenance.py --root vendor/ud-tools)" ||
        die "cannot compute the pinned Tools source digest"
fi
export TOOLS_GITLINK_SHA TOOLS_SOURCE_SHA256

mode="${1:-}"
[ -n "$mode" ] || die "usage: $0 headless|gui|gui-wslg|engines|probe-gl|build|ensure [ARGS...]"
shift

mkdir -p output/explorer

# Health check: a pruned image must be rebuilt, never pulled (a missing local
# tag makes compose try `docker pull` and fail with "pull access denied").
# ensure_image IMAGE SERVICE PROFILE
ensure_image() {
    if docker image inspect "$1" >/dev/null 2>&1; then
        kept="$(docker image inspect --format '{{ index .Config.Labels "keep" }}' "$1" 2>/dev/null || true)"
        [ "$kept" = "true" ] || echo "[explorer] WARNING: $1 lacks label keep=true; host prune jobs may delete it (rebuild to label it)" >&2
        return 0
    fi
    echo "[explorer] image $1 is missing (pruned?); rebuilding service $2" >&2
    docker compose --profile "$3" build "$2" || die "could not rebuild $1"
    docker image inspect "$1" >/dev/null 2>&1 || die "$1 is still missing after the rebuild"
}

ensure_all() {
    ensure_image upstream-drift:explorer-headless explorer-headless explorer-headless
    ensure_image upstream-drift:explorer-gui explorer-gui explorer-gui
    ensure_image ud-sim:latest ud-sim explorer-images
    ensure_image upstream-drift:full upstream-drift-full explorer-images
}

case "$mode" in
    ensure)
        ensure_all
        echo "[explorer] all required images present" >&2
        ;;
    headless)
        ensure_image upstream-drift:explorer-headless explorer-headless explorer-headless
        exec docker compose --profile explorer-headless run --rm explorer-headless "$@"
        ;;
    engines | probe-gl)
        ensure_image upstream-drift:explorer-headless explorer-headless explorer-headless
        exec docker compose --profile explorer-headless run --rm explorer-headless "$mode" "$@"
        ;;
    gui)
        ensure_image upstream-drift:explorer-gui explorer-gui explorer-gui
        exec docker compose --profile explorer-gui up explorer-gui "$@"
        ;;
    gui-wslg)
        ensure_image upstream-drift:explorer-gui explorer-gui explorer-gui
        exec docker compose --profile explorer-gui-wslg up explorer-gui-wslg "$@"
        ;;
    build)
        exec docker compose --profile explorer-headless --profile explorer-gui build "$@"
        ;;
    *)
        die "unknown mode '$mode' (headless|gui|gui-wslg|engines|probe-gl|build|ensure)"
        ;;
esac
