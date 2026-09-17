#!/bin/bash
# Launch a detached native Crocoddyl full-body fit (MS-31 #10338) and log it.
#
#   bash scripts/matched_swing/run_crocoddyl_fit.sh <name> <t_end_s> <max_iter> [extra driver args]
#
# Example G1 run (0.85 s) with horizon continuation:
#   bash scripts/matched_swing/run_crocoddyl_fit.sh driver_g1 0.85 150 --quiet \
#       --continuation 0.05,0.10,0.20,0.30,0.45,0.60 --stage-iterations 40
#
# Environment: micromamba env `upstream-motion-runtime` under $MAMBA_ROOT_PREFIX
# (default $HOME/mm-root, see bootstrap_motion_runtime.sh). Output goes to
# $FIT_OUT_ROOT/<name> (default $HOME/fits) and the log to $FIT_OUT_ROOT/<name>.log.
# Poll with:  grep -E '^EXIT|Traceback' $HOME/fits/<name>.log
set -euo pipefail
NAME=$1; T_END=$2; ITER=$3; shift 3
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/mm-root}
MM=${MICROMAMBA:-$HOME/bin/micromamba}
OUT_ROOT=${FIT_OUT_ROOT:-$HOME/fits}
mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/$NAME.log"
DOC=${FIT_DOCUMENT:-docs/development/full_body_models/evidence/ground_support/anthro_driver/full_body_spec_hipcal_scaled.json}
ATT=${FIT_ATTACHMENTS_RECEIPT:-docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json}
CAP=${FIT_CAPTURE:-data/C3D_TA_Driver.c3d}
cd "$REPO"
nohup bash -c "echo START \$(date -Is) t_end=$T_END iter=$ITER $* > '$LOG'; '$MM' run -n upstream-motion-runtime python -m src.engines.physics_engines.pinocchio.python.full_body_fit --document '$DOC' --attachments-receipt '$ATT' --capture '$CAP' --t-end $T_END --max-iterations $ITER --out '$OUT_ROOT/$NAME' $* >> '$LOG' 2>&1; echo EXIT \$? \$(date -Is) >> '$LOG'" > /dev/null 2>&1 &
disown
echo "launched $NAME -> $LOG"
