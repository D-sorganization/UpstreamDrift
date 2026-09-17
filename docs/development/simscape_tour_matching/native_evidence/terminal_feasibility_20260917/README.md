# Terminal Feasibility and Native Chart Audit

These are completed diagnostic runs on ControlTower / WSL ControlTower-Runner,
Python 3.12.3 and Pinocchio 4.1.0. They do not replace run102 or qualify a new swing.
See manifest.json for archive SHA256 values and the pinned corrected provider.

## Evidence Inventory

- exact_yaw.tar.gz / exact_yaw.json: six local solves, two rotation halfwidths
  (0.5 and 1 rad), translation halfwidth 0.25 m, original plus two perturbed starts.
- yaw_range.tar.gz / yaw_range.json: eighteen solves, rotation halfwidth 1 rad,
  translation halfwidth 0.25 m, three starts at each of five yaw offsets
  (-5%, -2.5%, 0, +2.5%, +5%) plus unrestricted yaw. Percentage denominator is
  max(abs(target yaw in degrees), 1 degree); 5% equals 0.04765597218473765 rad.
  Boundary comparison allows 1e-8 percentage points of roundoff.
- chart.tar.gz / chart.json: closure derivative checks at 0, 0.6 and 0.85 s,
  original and wrist-offset (+0.1 rad) states, steps 1e-4, 1e-5 and 1e-6; separate
  nonzero chart retraction derivative checks. Nine records pass.
- terminal_residual_comparison.png: all 25 terminal marker errors from saved MATLAB
  prediction versus the best tested yaw-valid static pose. Visually inspected.

Each archive contains the exact executed audit.py and receipt.json. Static archives
also contain corrected_native.py and constrained_marker_pose.py. No target markers,
model dimensions or attachment coordinates were changed. The rotation/translation
boxes are local numerical search bounds, not anatomical joint-limit certification.
SLSQP uses analytic marker and weld pose Jacobians, maximum 150 iterations and
ftol 1e-12. The combined closure/yaw derivative was independently checked before
solving. All 24 solves converged with closure residuals around 1e-13.

## Reproduction and Preservation

Completed remote directories (do not overwrite):

- /home/dieterolson/native-terminal-feasibility-20260917-01
- /home/dieterolson/native-terminal-feasibility-20260917-02
- /home/dieterolson/native-chart-audit-20260917-01

Input model.bin/candidate.json/reference.npz live in
/home/dieterolson/native-clean-replay-20260917-01. Parent evidence contains the
original raw model (runtime78_original_model.bin), six hash-matched fitting inputs
(run102_original_inputs.tar.gz), replay restoration receipt and driver. Tools source
is pinned at 1ac89c18e6280752d949e520c2143d2fb584d31e. Audit source/runtime path is
/home/dieterolson/native-fit-audit-20260917-01. Exact script paths are preserved in
the archives; inspect their imports/inputs before relocation to another host.

Extract a verified archive into a NEW directory and run its audit.py there using:

```text
ssh controltower "wsl -d ControlTower-Runner --cd <NEW_DIRECTORY> -- env PYTHONPATH=/home/dieterolson/native-fit-audit-20260917-01 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/dieterolson/simscape-pinocchio-9967/.venv/bin/python audit.py"
```

This is a host-specific evidence reproduction command, not a portable product CLI.
A portable configuration-driven diagnostic driver is next-agent work. Capture exit
status and new receipts; preserve original files and hashes. Avoid login shells
with the broken .cargo-rust-lane startup reference on this host.

## Results and Limits

Exact yaw: 40.37778 mm. Best tested allowed yaw: 39.76310 mm at +5%.
Unrestricted yaw: 39.20167 mm, yaw error 15.70349%. All fail 35 mm.
No numerical bound is active at the best allowed solution. This repeated local
result does not prove a global minimum or impossibility. No anatomical-limit,
full-trajectory sensitivity, acceleration/reaction or new dynamic qualification
is implied. See the parent progress review and resume prompt for next actions.
