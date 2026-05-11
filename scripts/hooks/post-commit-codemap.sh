#!/usr/bin/env bash
# Post-commit hook: incrementally refresh the code-map index.
#
# This is the fallback path described in `chat_codemap_design.md` Part 2 §6
# for developers who do not run `codemap-watch` as a background daemon.
#
# It runs in the background so commit latency is not affected.
#
# ── Installation ────────────────────────────────────────────────────────────
# Git hooks cannot be checked into `.git/hooks/` directly; enable manually:
#
#   ln -s ../../scripts/hooks/post-commit-codemap.sh .git/hooks/post-commit
#
# or add it via your preferred hook manager (pre-commit's `local` stanza,
# husky, lefthook, etc.). See `docs/codemap-integration.md` for details.
# ───────────────────────────────────────────────────────────────────────────

set -euo pipefail

# Bail silently if codemap isn't available on PATH (devs without it installed
# should not have commits blocked or noisy).
if ! command -v codemap >/dev/null 2>&1; then
    exit 0
fi

# Bail silently if the index has never been built — first-time devs run
# `codemap rebuild` (or `make codemap`) explicitly.
if [ ! -f .codemap/index.db ]; then
    exit 0
fi

# Incremental rebuild covering only the just-committed delta. Run in the
# background so we don't slow down the next `git` invocation.
( codemap rebuild --since HEAD~1 >/dev/null 2>&1 & )

exit 0
