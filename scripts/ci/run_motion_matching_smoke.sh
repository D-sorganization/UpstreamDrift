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
#
# Headless GL (#12855): the playback GIF render needs an OpenGL context. On a
# host with no DISPLAY and no MUJOCO_GL, probe egl then osmesa via a tiny
# mujoco.GLContext test; if neither works, fall back to xvfb-run when present.
# The pipeline also treats playback as best-effort, so this only restores GIFs.
set -euo pipefail
cd "$(dirname "$0")/../.."

probe_gl() {
  MUJOCO_GL="$1" python -c     "import mujoco; c = mujoco.GLContext(8, 8); c.make_current(); c.free()"     >/dev/null 2>&1
}

runner=()
if [[ -z "${MUJOCO_GL:-}" && -z "${DISPLAY:-}" ]]; then
  for backend in egl osmesa; do
    if probe_gl "$backend"; then
      export MUJOCO_GL="$backend"
      echo "run_motion_matching_smoke: using MUJOCO_GL=$backend" >&2
      break
    fi
  done
  if [[ -z "${MUJOCO_GL:-}" ]]; then
    if command -v xvfb-run >/dev/null 2>&1; then
      runner=(xvfb-run -a)
      echo "run_motion_matching_smoke: using xvfb-run" >&2
    else
      echo "run_motion_matching_smoke: no headless GL; playback GIFs will be skipped" >&2
    fi
  fi
fi

exec ${runner[@]+"${runner[@]}"} python -m pytest tests/integration/motion_matching \
  -m "slow and not live_simulation" \
  --timeout=0 \
  -p no:randomly \
  -rA \
  --tb=short \
  "$@"
