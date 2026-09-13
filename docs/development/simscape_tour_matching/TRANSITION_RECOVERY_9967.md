# Transition Recovery Evidence and Execution Plan

## Evidence That Changes the Next Action

Run 33 is a full-capture static continuation with fixed geometry and attachments.
Its independent FK audit reproduces the marker RMS and records local active bounds.
LFInput is active throughout 0.6–0.85 s. Eight coordinates are active at 1.2 s;
many more saturate during the later large error spike. Optimizer convergence
therefore does not establish a geometric error floor.

The separate rigid-marker audit does not depend on any fitted pose. For every
pair attached to the same body it evaluates the absolute difference between
observed pair separation and the fixed offset separation. Both valid flags are
required. It checks all 654 frames and separately 0.6–0.85 s.

BackLeft and HeadFront are both attached to Hub. Their separation mismatch peaks
at 75.2327 mm during transition and 89.0449 mm over the capture. By the triangle
inequality, their two positional errors must sum to at least that mismatch;
at least one must be 37.6163 mm wrong at the transition peak. This is a
pair-specific bound, not a 37.6 mm lower bound on overall marker RMS.
Changing torques or rigid-body pose cannot remove it. Recalibrating fixed offsets
can redistribute errors but cannot reproduce a time-varying target separation.

Other maximum pair mismatches: Hip 9.2214 mm, LS 31.3598 mm, LScap 20.6044 mm,
RS 32.9539 mm, Clubhead 12.7101 mm. These may reflect marker mapping, capture
averaging, soft tissue, or model rigidity; the audit does not identify which.
Do not silently remove markers or change the Simscape-equivalent model topology.

## Immediate Experiment

Run 34 repeats the existing static fitter with frames 0,9,...,648,653 and the
same 0.15 local-coordinate bound, fixed model, candidate and payload. This halves
the interval to 25 ms and increases allowed cumulative coordinate movement;
it tests a continuation restriction, not a physical rate limit or mesh convergence
at equal admissible velocity. Compare common frames and independently audit bounds.
The runtime is native-ms-pilot-9967-25 in ControlTower-Runner. Remote executable
is C:/Users/diete/fit_native_marker_pose_sequence_9967_33.py; output is
C:/Users/diete/native-full-pose-sequence-9967-34.json. Handle24560 exited0; receipt is archived. All74 poses satisfy closure, maximum
5.448e-11, and two solves hit their iteration limit. At0.85 s RMS improves only
36.14 to35.77 mm. At1.35 s it improves427.20 to165.36 mm, and at the final frame
144.01 to61.41 mm. Thus continuation restrictions account for a substantial
part of the late spike, but this change barely improves transition. No dynamic
acceptance or global geometric floor is established.

## Follow-On Decision

1. If finer continuation removes the late spike, retain its feasible poses as
   an initializer, not an accepted trajectory. Audit marker groups and source hashes.
2. Review anatomical body assignments, especially head markers on Hub, against
   the R2025b source. Preserve the equivalent baseline and record any proposed
   model revision separately across all engines.
3. Calibrate fixed attachments and supported lengths across representative
   frames with held-out validation. Report irreducible residuals explicitly.
4. Return to existing constrained forward multiple shooting. Compare flexible
   continuous torques and global sextic torques under identical geometry and
   acceptance gates. Never infer full-swing feasibility from reset windows.
5. Refine global sextic coefficients against uninterrupted forward marker error,
   then replay the identical candidate in Simscape R2025b and other engines.

## Artifacts

- native_evidence/native-full-pose-sequence-9967-34.json
- native_evidence/native-full-pose-sequence-9967-33.json
- native_evidence/native-full-pose-audit-9967-33.json
- native_evidence/native-rigid-marker-audit-9967-34.json

Reproduce the pair audit with candidate marker_bodies/marker_offsets_m and payload
labels/points_world_m/valid/time_s. For each within-body pair i,j, compute
abs(norm(target_i-target_j)-norm(offset_i-offset_j)) on jointly valid frames.
The forward audit calls NativePinocchioModel.marker_derivatives at each recorded
pose, matches candidate labels to payload labels, and compares each coordinate
with the preceding pose (candidate q0 for the first), marking abs(delta)>=0.15-1e-6.
No new production implementation or acceptance test is claimed by these analyses.
