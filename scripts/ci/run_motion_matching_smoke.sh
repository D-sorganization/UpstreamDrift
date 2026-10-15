#!/usr/bin/env bash
# run_motion_matching_smoke.sh -- UpstreamDrift (#12784)
# Single entry point for the slow motion-matching integration smokes. Used by
# .github/workflows/motion-matching-smoke.yml and runnable locally as-is, so the
# scheduled lane and a developer's machine run the same command.
#
# `-m "slow ..."` overrides the pyproject addopts `-m "not slow ..."`, and
# `--timeout=0` is required: full-body IK alone outlasts the default per-test
# timeout and would hide the real failure (#12702). The owner-capture case
# skips when the private dataset (CAPTURE_DATA_DIR) is unavailable.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec python -m pytest tests/integration/motion_matching \
  -m "slow and not live_simulation" \
  --timeout=0 \
  -p no:randomly \
  -rA \
  --tb=short \
  "$@"
