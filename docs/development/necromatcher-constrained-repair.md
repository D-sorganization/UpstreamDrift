# Necromatcher Constrained Motion Repair

## Repair the Native Boundaries First

The native plant frame query now resolves the exact requested specification
aliases through the existing native body-pose implementation. Previously marker
construction lost those aliases and the public query returned an empty mapping.
Regression tests check alias identity, world positions, orthonormal rotations,
invalid coordinate vectors and unknown frames.

Native marker IK now uses the principal SO(3) rotation vector for grip closure.
The former sine-angle residual vanished at a half turn. The derivative now uses
the inverse left logarithm Jacobian and the relative-frame angular Jacobian.
Red-first native tests verify finite-angle errors and central-difference agreement
at two large rotations. The principal logarithm has an inherent branch at pi;
local derivative checks away from that branch do not prove global convergence.
Shared and Drake legacy implementations retain their previous behavior.

## Produce a Separate Discrete Research Candidate

In a clean native-SDK process, import mujoco before the workspace package and
load the saved resource with workspace.load_native_fit_binding. Call
workspace.repair_native_motion(binding, iterations=150). The operation reuses the
existing bounded native pose IK; targets are world marker positions generated
from each original fitted pose. Each original pose is also its own prior.
These targets are inferred model geometry, not measured three-dimensional motion
or original image observations.

The procedure converts explicit authored degree ranges to radians after checking
named compiled angular units. Every returned sample is independently checked
against those ranges. Unbounded coordinates remain explicitly listed. Closure
position, finite orientation and ground use weights of 1e6; the pose prior uses
1e-4. These are soft penalties, not hard grip or ground equalities. The report
records actual residuals, iterations, budget exhaustion and projected pixel
changes against the original hypothesis. Budget exhaustion is not convergence.

The report preserves exact source/model hashes, native XML identity, ordered
units and source frames. Source/runtime fingerprints are checked again after
execution. It returns discrete samples only. It publishes no library fit,
physical clock, velocities, interpolation or effort profile. Save the complete
JSON outside Git and keep a compact receipt with artifact hashes for turnover.

## Actual Source Diagnostics

After repairing the frame query, the unchanged v6 Tiger track has a maximum
0.243580 m grip separation and 156.876 degrees orientation error. Eight authored
coordinate ranges are violated, including spine Y and both hip rotations.
Hogan has a maximum 0.111762 m separation and 45.162 degrees orientation error;
its right-knee coordinate exceeds its range in all 750 stored samples. The generic
model declares ranges for 32 of 44 coordinates. These ranges do not establish
measured Tiger or Hogan anatomy.

Initial 150-iteration experiments close both grips below one micrometre, with
maximum projected marker changes of 13.292 px for Tiger and 3.974 px for Hogan.
Both experiments hit the iteration budget. Whole-track experiments and source
image residual review remain required; initial-frame results qualify neither.

## Local Validation

The combined library, fitting, native, repair, effort, replay, placement, handoff
and marker suite passes 163 tests. Scoped standard mypy, Ruff, title capitalization,
file/function budgets and manual governance pass. CI remediation remains exhausted
under the existing report; these local results are not remote green CI evidence.

## Continue Toward Simulation

Compare a whole-track candidate against the actual image observations and review
Hogan swing windows and cropped feet. Preserve the original hypotheses and record
all tradeoffs. A later spline must enforce and audit limits between samples;
bounded samples alone do not prevent Hermite overshoot or nonlinear grip drift.
Only after a usable closed continuous path exists should authored controls and
independent forward replay be attached. Historical effort identification, source
clock calibration, downstream analysis and AffineDrift remain open under #11235
and #11232. Scientific qualification remains false, and PR #11240 remains draft.
