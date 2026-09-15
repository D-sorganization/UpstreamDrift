#!/bin/bash
set -euo pipefail

rm -rf /mnt/c/Users/diete/native-two-window-fit-9967-101

export PYTHONPATH=/home/dieterolson/native-two-window-fit-9967-78
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

/home/dieterolson/simscape-pinocchio-9967/.venv/bin/python /mnt/c/Users/diete/two_window_fit_101.py \
  --model /mnt/c/Users/diete/native_geometry_spec_9967.json \
  --candidate /mnt/c/Users/diete/native-regularized-fit-9967-73/returned-candidate.json \
  --parent /mnt/c/Users/diete/native-ms-fit-9967-19/returned-candidate.json \
  --returned73 /mnt/c/Users/diete/native-regularized-fit-9967-73/returned.json \
  --target /mnt/c/Users/diete/driver_marker_payload_9967.json \
  --runtime /home/dieterolson/native-two-window-fit-9967-78 \
  --restart /mnt/c/Users/diete/native-two-window-fit-9967-100/returned-candidate.json \
  --output /mnt/c/Users/diete/native-two-window-fit-9967-101 \
  --max-iterations 40 \
  --max-nfev 120 \
  --node-bound 0.075 \
  --box-factor 6.5 \
  --terminal-weight 25.0 \
  --pelvis-yaw-weight 40.0
