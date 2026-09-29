#!/usr/bin/env bash
# Migration script for legacy UpstreamDrift Docker image names to the canonical 'upstream-drift' tag schema.
set -euo pipefail

DRY_RUN=false
ROLLBACK=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true ;;
        --rollback|--reverse) ROLLBACK=true ;;
        -h|--help) 
            echo "Usage: ./docker_migrate_names.sh [--dry-run] [--rollback]"
            echo "  --dry-run   Show what would be tagged without doing it"
            echo "  --rollback  Reverse the transition, recreating legacy tags from canonical tags"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

tag_image() {
    local source=$1
    local target=$2

    if docker image inspect "$source" >/dev/null 2>&1; then
        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] Would tag '$source' -> '$target'"
        else
            echo "Tagging '$source' -> '$target'..."
            docker tag "$source" "$target"
        fi
    else
        echo "Source image '$source' not found locally. Skipping."
    fi
}

echo "=== UpstreamDrift Docker Name Migration ==="
if [ "$DRY_RUN" = true ]; then echo "MODE: Dry Run"; fi
if [ "$ROLLBACK" = true ]; then echo "MODE: Rollback (Reverse)"; fi
echo "-------------------------------------------"

if [ "$ROLLBACK" = true ]; then
    tag_image "upstream-drift:engine" "robotics_env:latest"
    tag_image "upstream-drift:runtime" "golf-suite:latest"
    tag_image "upstream-drift:dev" "golf-suite-dev:latest"
else
    tag_image "robotics_env:latest" "upstream-drift:engine"
    tag_image "golf-suite:latest" "upstream-drift:runtime"
    tag_image "golf-suite-dev:latest" "upstream-drift:dev"
fi

echo "Done."
