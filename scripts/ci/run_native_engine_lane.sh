#!/usr/bin/env bash
# Run OpenSim or MyoSuite native pytest lanes and write nightly receipts (MS-43 #10342).
#
# Intended for ControlTower / labeled fleet runners.  Standard CI skips these
# markers; refresh receipts at least weekly via nightly-cross-engine.yml or:
#
#   bash scripts/ci/run_native_engine_lane.sh --engine opensim \
#     --out docs/development/matched_swing_program/evidence/nightly
#
# ControlTower example (opensim-10003 venv):
#
#   bash scripts/ci/run_native_engine_lane.sh --engine opensim \
#     --venv /home/dieterolson/opensim-10003 \
#     --out docs/development/matched_swing_program/evidence/nightly
#
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

ENGINE=""
OUT="docs/development/matched_swing_program/evidence/nightly"
VENV=""
PYTHON=""
TIMEOUT="600"

usage() {
  sed -n '2,16p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine)
      ENGINE="$2"
      shift 2
      ;;
    --out)
      OUT="$2"
      shift 2
      ;;
    --venv)
      VENV="$2"
      shift 2
      ;;
    --python)
      PYTHON="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$ENGINE" ]]; then
  echo "--engine is required (opensim|myosuite)" >&2
  exit 2
fi

ARGS=(--engine "$ENGINE" --out "$OUT" --timeout "$TIMEOUT")
if [[ -n "$VENV" ]]; then
  ARGS+=(--venv "$VENV")
fi
if [[ -n "$PYTHON" ]]; then
  ARGS+=(--python "$PYTHON")
fi

exec python "$REPO/scripts/ci/run_native_engine_lane.py" "${ARGS[@]}"
