# Historical Matching Development Notes

These prior chronological entries are retained as history, not current status.
The canonical DEVELOPMENT_LOG and HANDOFF contain current continuation state.

## 2026-09-12 - Pointwise Dynamics Audit Narrows Remaining Diagnosis

All307 saved scalar states were checked through direct and actual-RHS manifold
routes. Maximum native acceleration difference2.18e-6 at a539073rad/s² reference;
effort roundtrip1.25e-12; exact zero same-state history difference in both models.
No substantial sampled conversion/history defect was found. Updated next action
to quantify perturbation growth and avoid further blind tolerance sweeps.

## 2026-09-12 - Run59 Does Not Confirm Robust Parity

Smaller-step59 completes but native velocity error0.000587270 fails. Isolated58
is not promoted. Root archives terminal evidence and switches from tolerance
repetition to a full saved-state acceleration/actuator-route/history audit.

## 2026-09-12 - DOP853 Run58 Passes Prefix Parity

Run58 passes unchanged marker/native-state/closure gates over0.85 s in37.69 s.
It is a representation fixture, not an accepted C3D fit. Smaller-step59 is active.
Replaced1262-line accumulating HANDOFF with a concise current-status document;
preserved all prior entries and incoming contexts in HANDOFF_HISTORY_20260912.md.
Historical ACTIVE statements are explicitly nonauthoritative.

## 2026-09-12 - DOP853 Pilot57 Terminal

Both pilot levels complete, improving cost relative to tight RK4 but still
failing native velocity parity. Smaller step is not monotonic convergence.
Archived57 and launched one tighter-tolerance58 with the same explicit budget.
Implementation push through ae06e933b passed all pre-push checks.

## 2026-09-12 - Tested Local-Chart DOP853 and Native Pilot57

Added shared high-order chart integration using the same checked RHS as RK4.
TDD tests cover noncommuting rotations, absolute clock across rebases, global
evaluation budget and malformed callbacks. Root independently passes27 local
tests, Ruff and mypy; remote qualification96passed1optional skip. Driver now
records scalar tolerances independently. Run57 launched with tighter scalar
reference and unchanged parity gates; result pending at this checkpoint.

## 2026-09-12 - Tighter Manifold Replay56

Run56 completed181.85 s with52116 evaluations. Native rate discrepancy improves
to0.000604599 but still fails the unchanged gate; marker/position/closure pass.
Saved physical-velocity audit explains chart amplification without waiving error.
Started bounded TDD high-order local-chart integrator to reduce evaluation cost.

## 2026-09-12 - Scalar Reference and Physical Velocity Diagnostics

Archived54/55 first-level scalar refinements and explicit second-level budget
failures. The finest completed scalar changes native rate by5.53e-5 relative to
baseline, insufficient to explain manifold discrepancy0.00471. Parallel saved
state diagnostic finds chart amplification but nonzero body angular velocity
error. Launched bounded tighter manifold56 without changing acceptance gates.

## 2026-09-12 - Run52 Completed Without Boundary Failure

Both adaptive levels completed after the TDD boundary fix. Finest native velocity
discrepancy0.00465279 still fails parity despite marker discrepancy1.17068e-6 m
and good closure. Same-tolerance max-step refinement is insufficient evidence of
convergence. Archived terminal outputs and ordered physical-velocity/conditioning
and independent reference-tolerance diagnostics. Implementation push through
c1cd4de8c passed Ruff, mypy, Bandit and unit pre-push checks.

## 2026-09-12 - Terminal Replay Evidence and Convergence Review

Verified run51 terminal exit1, archived its completed first level and failed
second-level status. First level has good closure but fails state/marker parity;
second level underflows at an output boundary. Assigned a focused TDD regression
fix. Added CONVERGENCE_REVIEW_20260912.md separating initializer quality,
open-loop polynomial feasibility, representation bugs and numerical qualification.
The review orders bounded implementation gates and preserves R2025b acceptance.
Boundary regression fixed without relabeling accepted states;20 generic tests
pass independently and runtime16 passes88 with one optional skip. Run52 launched
with identical comparison settings on immutable runtime16; result pending.

## 2026-09-12 - Branch-Preserving Adaptive Manifold Replay

Added tested native branch preservation after run49 exposed wrong actuator
mapping. Run50 improves marker discrepancy to16micrometres but still fails
closure/state gates. Added shared fixed/adaptive local-chart RK4 with independent
order tests, budgets and contracts. Run51 adaptive comparison launched on
ControlTower with fixed limits; no result claimed while active. Root reran
18generic tests, Ruff and mypy; real runtime qualification records86passes.

## 2026-09-12 - Manifold Replay Exposes Actuator Branch Bug

Local-chart RK4 passes independent fourth-order and real engine checks. Same-input
run48 .2 s passes, but run49 .85 s does not converge to scalar trajectory. Root
identified opposite shoulder branch selection at.786111 s and requested explicit
branch-preserving inverse regression. Run46 strict comparison intentionally
interrupted at33CPUmin; terminal evidence archived, no convergence conclusion.

## 2026-09-12 - Native Pinocchio Spherical Prototype

Integrated shared-construction alternate native builder with current-state effort
conversion, contracts and TDD. Root independently reran five real Pinocchio4.1.0
tests on ControlTower; saved swing-state qualification also passes. Exact source
and receipts archived; original/formatted geometry parsed equality verified.
Validated production lazy initializer, Ruff and mypy. Dedicated manifold handoff
records missing time integration, trajectory, derivative and R2025b gates.

## 2026-09-12 - Standalone Numerical Import Boundary

Converted the pose_interchange facade to lazy public-provider loading, preserving
all exported symbols and type-checking imports. Fresh-process application-blocked
import test went red then green; full representation unit selection, Ruff and
mypy pass. Removes remote numerical runtime need to replace the package initializer.

## 2026-09-12 - Effort-Regularized Native Forward Initializer

Added optional rank-preserving effort regularization with TDD (five new red tests,
all10 allocator tests now pass), Ruff and mypy. Run47 full feedback RMS45.514 mm
reduces peak sampled force to3292 N and torque to1288 Nm; no open-loop claim.
Archived exact inputs and outputs. Measured poor global sextic approximation of
run45 efforts and recorded requirement to optimize trajectory and controls jointly.
Run46 strict comparison remains live. Parallel native manifold builder underway.

## 2026-09-12 -- Time-Only Replay Sensitivity Audit

Measured run45 first divergence and launched fixed-input run46 tolerance pair
on ControlTower. Baseline reproduces saved states exactly at requested sample
times through1.1 s; tight comparison remains pending. Diagnostic uses original
initial state and actual constrained forward dynamics without tracking feedback.

## 2026-09-12 — Gimbal-Safe Initializer and Representation Interchange

Added tested gimbal-branch restrictions and archived runs42–45. Full feedback
tracking reaches35.2153 mm RMS, but time-only replay remains346.534 mm in run45
and global sextic655.546 mm: no accepted full swing. Increased sampling and
reference-state effort evaluation do not resolve divergence. Updated authoritative
handoff with bounded diagnosis and dynamically feasible shooting next steps.
Created epic #10043 and integrated shared angle/quaternion, differential effort,
frame and native-state interchange with dedicated representation turnover.
Alternate manifold engine builders and R2025b dynamics parity remain open.
