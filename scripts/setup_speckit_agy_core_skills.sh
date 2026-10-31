#!/usr/bin/env bash
# Materialize Spec Kit core Antigravity skills from the pinned Specify CLI.
# Git extension skills are repo-owned under .agents/skills/speckit-git-*.
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
cd "$root"
if ! command -v specify >/dev/null 2>&1; then
  echo "specify CLI not found; install with: pip install specify-cli" >&2
  exit 1
fi
specify integration install agy --force --trust-integration
