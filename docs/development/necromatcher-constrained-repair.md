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

The repair opts into the native bounded SciPy TRF solver, using analytic
Jacobians, mixed finite/infinite bounds and cached paired FK evaluations.
Locked coordinates remain exact; lock/bound conflicts and unsupported providers
fail. The existing LM default remains available for other callers. Native ground
residuals now retain zero rows for inactive contacts, preserving the same objective
while keeping TRF residual dimensions fixed. The TRF budget counts residual
evaluations (nfev), not optimizer iterations; reaching it is not convergence.

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

## Whole-Track Probe Results

The [Committed-Source Probe Receipt](historical_capture/constrained-repair-probe-receipt.json)
records the earlier clipped-LM experiment from 7d8c04946d9ebb69a698d1e83d9d67322e76c87c. Complete candidate
JSON files remain outside Git in Downloads/historical-capture/native-fit-research;
receipt hashes bind their exact bytes. All original library fits are unchanged.

| Diagnostic                          | Tiger, 210 Frames | Hogan, 750 Frames |
| ----------------------------------- | ----------------- | ----------------- |
| Maximum Grip Gap                    | 0.023295 m        | 0.013759 m        |
| Maximum Grip Rotation               | 0.622767 deg      | 0.220956 deg      |
| Maximum Sample Ground Penetration   | 2.03e-7 m         | 0.000267 m        |
| Maximum Pixel Change from Original  | 188.932 px        | 28.425 px         |
| Original Dense Image RMS            | 21.352 px         | 9.849 px          |
| Candidate Dense Image RMS           | 34.094 px         | 12.508 px         |
| Maximum Midpoint Ground Penetration | 0.000060 m        | 0.053227 m        |
| Iteration Budget Reached            | 210/210           | 750/750           |

Dense image RMS uses canonical camera residuals against original capture
landmarks, weighted by recorded visibility with an explicit 0.5 prior for unknown
visibility. It is distinct from held-out optimization RMS. All 32 authored ranges
hold at samples. Midpoint checks evaluate coordinate averages, not a qualified
continuous trajectory. These candidates fail usable closure and image-fidelity
review and must not become accepted simulation resources. The next solver step is
an explicit bounded optimizer, followed by the same full-track and source checks.

## Local Validation

The combined library, fitting, native, repair, effort, replay, placement, handoff
and marker suite passes 169 tests. Scoped standard mypy, Ruff, title capitalization,
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
