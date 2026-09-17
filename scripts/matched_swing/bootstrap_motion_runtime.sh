#!/bin/bash
# Bootstrap the qualified Pinocchio + Crocoddyl + Pink runtime (Linux) for the
# Matched Swing Program fits (MS-31 #10338). Idempotent.
#
#   bash scripts/matched_swing/bootstrap_motion_runtime.sh [ROOT]
#
# ROOT defaults to $HOME. Creates $ROOT/mm-root (micromamba root) and the env
# `upstream-motion-runtime` from scripts/config/motion_runtime/linux-64.explicit.txt,
# then adds the packages the fit driver needs beyond the lock.
set -euo pipefail
ROOT=${1:-$HOME}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export MAMBA_ROOT_PREFIX="$ROOT/mm-root"
mkdir -p "$ROOT/bin"
if [ ! -x "$ROOT/bin/micromamba" ]; then
  curl -sSL -o /tmp/mm.tar.bz2 https://micro.mamba.pm/api/micromamba/linux-64/latest
  tar -xjf /tmp/mm.tar.bz2 -C "$ROOT" bin/micromamba
fi
MM="$ROOT/bin/micromamba"
if ! "$MM" env list | grep -q upstream-motion-runtime; then
  "$MM" create -y -n upstream-motion-runtime --file "$REPO/scripts/config/motion_runtime/linux-64.explicit.txt"
fi
"$MM" install -y -n upstream-motion-runtime -c conda-forge ezc3d pydantic imageio matplotlib pytest scipy >/dev/null
"$MM" run -n upstream-motion-runtime python -c "import pinocchio, crocoddyl, pink, ezc3d, scipy; print('runtime OK', pinocchio.__version__, crocoddyl.__version__, pink.__version__)"
