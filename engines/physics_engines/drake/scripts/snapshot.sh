#!/usr/bin/env bash
set -euo pipefail
DATE=$(date +%F)
git add -A
git commit -m "chore(wip): snapshot ${DATE}" || true
git tag -f "snapshot-${DATE}"
git branch -f "safety-net/${DATE}"
git push origin "safety-net/${DATE}" --tags || true
echo "Snapshot complete for ${DATE}."
