# Pinocchio Manifold Prototype Handoff

## Authority and Status

Epic #10043, parent matching #9921/#9967. MATLAB R2025b remains the reference.
This is a bounded, live-tested alternate representation. Full trajectory and
Simscape parity are **not accepted**. The scalar model is retained.

`NativeManifoldPinocchioModel` replaces only the three eligible contiguous
rotational triples inside native hip/shoulder joints with Pinocchio spherical
joints. All 18 remaining scalar coordinates, 31 solids, body placements, marker
frames and the original six-dimensional right-hand weld use the same builder.
The original builder has only a protected construction hook extraction; no
inertia, actuation, damping or constraint law is changed.

## API and Conventions

Implementation: `src/engines/physics_engines/pinocchio/python/native_manifold_model.py`.
The internal construction-only subclass is hidden behind a facade so scalar-only
replay and derivative methods are not accidentally inherited. The public model
has **nq=30, nv=27**. Its representation identity is
`native-pinocchio-spherical-xyzw-body-v1`, separate from specification identity.

- `native_state(q, v, primitive_efforts)` maps named native SI values into engine
  q, body-tangent velocity and conjugate effort. Spherical q is **xyzw**;
  spherical velocity and moment are expressed in the rotating follower frame.
- `native_coordinates(q, v, reference_coordinates)` restores native angles/rates
  preserving the explicit reference middle-angle interval between gimbal poles,
  with nearest outer-axis winding.
- `acceleration_from_native_efforts(q, v, primitive_efforts, reference_coordinates)`
  maps native efforts using the **current alternate configuration every call**.
  The reference only chooses a branch; it is never a tracking target or reset.
- `acceleration` is the lower-level API for already mapped tangent efforts.
- `native_accelerations` supplies pointwise constrained native qdd, including
  the convective angular acceleration conversion.
- `frame_poses` and `frame_velocities` provide every native reference frame;
  velocity is frame-origin, world-aligned, **linear then angular**, as Pinocchio
  specifies. This is not the angular-first canonical spatial interchange layout.
- `integrate` and `difference` expose manifold operations only. They are **not a
  qualified time integrator**. Quaternion norm and finite shape checks reject
  malformed states. Inverse native charts reject singularity without pseudoinverse.

The common `NativeJointStateAdapter` and `SerialRotationChart` own all chart
math, rotation branches and power-dual effort conversions. No alternate model
may replace a two-axis wrist/spine/scapula mechanism by an unconstrained ball.

## Evidence and Reproduction

`native_evidence/manifold_10043_09/qualification.json` records Pinocchio **4.1.0**,
exact numerical source SHA-256 values and 5 passing real-runtime tests. Tests
cover eight random nonsingular states for frame FK, frame velocity, kinetic
energy, instantaneous power and branch restoration; manifold integrate/difference;
invalid shapes/quaternion norms; six random nonzero-rate constrained acceleration
and weld comparisons; and displaced-reference current-state effort mapping.
TDD missing-module red and missing-current-state-effort-method red were observed.
Local Windows tests skip the five engine tests because the suite supplies a
Pinocchio MagicMock; the two scalar ordering tests pass. Ruff and focused mypy
pass. A mock is never accepted as engine evidence.

`native_evidence/manifold_10043_09/swing-state-qualification.json` also compares
actual run45 feedback q/v/efforts at 0.6, 0.9 and 1.3 seconds. All gates pass.
Maximum native qdd disagreement is approximately 4.9e-9; maximum frame transform
error 8.9e-16; frame velocity error 1.5e-14. Exact values, explicit tolerances,
per-probe timings, input NPZ hash and script hash are in the receipt. The probes
take approximately 8 milliseconds each; this is not a simulation speed benchmark.
The run45 feedback trajectory itself is not an accepted open-loop swing.

Raw geometry SHA differs between formatted repository JSON (0202c8b2...) and
original runtime JSON (b817fea7...). Parsed objects were compared for equality,
and both produce canonical specification SHA
`db7cd60496edd1adef789b965bce897f12e20d9460b957a31d7fd99c8fd31518`.
Both raw hashes and the equality assertion are recorded in qualification.json.

ControlTower runtime: `/home/dieterolson/native-manifold-10043-09`, venv
`/home/dieterolson/simscape-pinocchio-9967/.venv/bin/python`.
The final qualification uses the exact production lazy pose_interchange
initializer, included in the source hashes. Earlier trials used a facade shim;
those are superseded. Outer src/package namespaces still inherit the small
gimbal42 qualification runtime. Numerical modules and production pose_interchange
initializer are exact hashed source and real Pinocchio C++ is used. Full
GUI/application integration is not established by these headless tests.

```powershell
ssh -o BatchMode=yes controltower 'wsl -d ControlTower-Runner -- env PYTHONPATH=/home/dieterolson/native-manifold-10043-09 /home/dieterolson/simscape-pinocchio-9967/.venv/bin/python -m pytest /home/dieterolson/native-manifold-10043-09/tests/unit/motion_matching/test_native_pinocchio_manifold.py -q -o addopts='
ssh -o BatchMode=yes controltower 'wsl -d ControlTower-Runner -- env PYTHONPATH=/home/dieterolson/native-manifold-10043-09 /home/dieterolson/simscape-pinocchio-9967/.venv/bin/python /home/dieterolson/native-manifold-10043-09/docs/development/simscape_tour_matching/qualify_native_manifold.py --model /mnt/c/Users/diete/native_geometry_spec_9967.json --states /mnt/c/Users/diete/native-feedback-initializer-9967-45/feedback.npz --output /mnt/c/Users/diete/native-manifold-repeat.json'
```

For a fresh checkout with real Pinocchio installed, run the checked-in test and
`qualify_native_manifold.py` directly. The latter accepts model, states and output
paths. Run45 raw feedback data are archived by the root matching work; its NPZ
SHA is recorded in the saved-state receipt. `raw-source.zip` beside the receipts
preserves exact tested source bytes despite later Git newline normalization.

## Local-Chart Time Integrator Checkpoint

`src/shared/python/motion_matching/manifold_forward.py` supplies
`integrate_manifold_forward(initial_configuration, initial_velocity, time,
acceleration, *, integrate, difference_rate, max_step=0.001)`.
The result contains read-only `time`, `configuration[T,nq]`, `velocity[T,nv]`,
`evaluations`, `steps` and `elapsed_s`. Every requested time is an integration
boundary. The initial clock is zero and all sample times increase strictly.

Each numerical step holds anchor q fixed, starts displacement u=0 and applies
classical RK4 to `(u,v)`, with q=integrate(anchor,u),
u_dot=difference_rate(anchor,q,v), and v_dot=acceleration(t,q,v).
The next anchor is the integrated endpoint. This changes coordinates only;
there is no weld projection, state reset, target-state replacement, quaternion
normalization rule, or tracking correction in the solver. Engine retraction
owns quaternion geometry. The method has no adaptive error estimate.

`NativeManifoldPinocchioModel.difference_rate` uses the actual Pinocchio
`dDifference(..., ARG1) @ v`; replacing this by v loses the noncommuting chart
correction. `closure_errors()` exposes checked detached weld pose/rate residuals
from the latest acceleration call through the existing scalar implementation.

Evidence: `native_evidence/manifold_integrator_10043_12/qualification.json` and
its exact `raw-source.zip`. Ten independent solver tests pass locally; all 17
solver/native tests pass on real Pinocchio 4.1.0 in
`/home/dieterolson/native-manifold-10043-12`. Missing-module and missing-adapter-
method red states preceded implementation. Ruff and focused mypy pass.
Independent exact noncommuting motion R(t)=Rx(t)Ry(t�) gives orientation errors
4.04923e-5, 2.60661e-6 and 1.66814e-7 radians at h=0.2, 0.1 and 0.05 seconds.
Refinement ratios 15.53 and 15.63 establish fourth-order behavior for this case.
Actual Pinocchio spherical operations reproduce those errors. Additional tests
cover the Euclidean harmonic oscillator order, exact constant acceleration,
clock/output ownership, malformed callback outputs, and an actual Pinocchio
centered tangent probe of dDifference.

These are integrator order and adapter tests, not candidate replay acceptance.
Root owns same-input run19 replay and will perform step-size convergence on the
constrained golf problem. Run19 is itself a rejected C3D fit, suitable here only
for a controlled representation comparison. The native inverse now explicitly preserves the reference middle-angle branch
(see the correction below). Outer-axis winding remains nearest the reference;
full-swing winding continuity and avoidance of actual singular crossings still
require audit.

```powershell
ssh -o BatchMode=yes controltower 'wsl -d ControlTower-Runner -- env PYTHONPATH=/home/dieterolson/native-manifold-10043-12 /home/dieterolson/simscape-pinocchio-9967/.venv/bin/python -m pytest /home/dieterolson/native-manifold-10043-12/tests/unit/motion_matching/test_native_pinocchio_manifold.py /home/dieterolson/native-manifold-10043-12/tests/unit/motion_matching/test_manifold_forward.py -q -o addopts='
```

## Middle-Angle Branch Correction

The first 0.85-second same-input attempt failed to converge with step refinement
because nearest-angle inverse selection changed the left-shoulder middle branch
near 0.786111 seconds. Equal orientation did not imply equal native actuator
semantics. The scalar state `[1.1848704063696793, -1.650976651746636,
-1.1484230300553964]` was restored on the opposite branch with middle angle
`-1.490616001843157`; the corresponding native effort mapping was wrong.
Root owns the rejected run49 receipt and the corrected run50 replay.

`SerialRotationChart.coordinates(..., preserve_middle_branch=True)` now filters
orientation-equivalent candidates to the reference interval between adjacent
`pi/2 + k*pi` poles before choosing nearest outer-axis winding.
`NativeJointStateAdapter.restore` exposes the same explicit option; its default
nearest behavior is unchanged for generic pose users. Both inverse paths in
`NativeManifoldPinocchioModel` explicitly opt in. Singular references/targets
remain rejected; no pseudoinverse or state reset was introduced.

TDD regressions cover target `[2.8,-1.6,2.8]`, reference `[0,-2,0]`, all six
axis sequences, both middle-angle signs and positive/negative 2\*pi winding.
The nearest default demonstrably selects a different branch while the explicit
option preserves q/v/qdd/effort semantics. A real native Pinocchio test compares
current-state applied effort acceleration to directly mapped original effort.

Final fix runtime: `/home/dieterolson/native-manifold-10043-13`, a new clone of
runtime12; earlier runtime11/12 evidence remains immutable. Live tests:
**77 passed, one optional MuJoCo test skipped**. Local shared tests: 59 passed.
Ruff/format and focused mypy pass. Exact changed source bytes and stdout are in
`native_evidence/manifold_branch_10043_13/{qualification.json,raw-source.zip}`.
These tests prove the branch correction at the tested states; corrected
same-input trajectory convergence remains a separate root-owned gate.

Preserving the interval does not authorize crossing an actual native gimbal
singularity. A quaternion may remain regular while original actuator/rate
coordinates become undefined. Detect and qualify that event explicitly rather
than interpreting alternate-coordinate integration as proof of original-model
fidelity through the singularity.

## Adaptive Step-Doubling Checkpoint

`integrate_manifold_adaptive` is additive; the fixed-step API remains available.
Both use one extracted `_ChartStepper` RK4 kernel. The adaptive signature is:

```python
integrate_manifold_adaptive(
    initial_configuration, initial_velocity, time, acceleration,
    integrate=model.integrate,
    difference_rate=model.difference_rate,
    difference=model.difference,
    rtol=1e-8, atol=1e-10, max_step=0.001, max_evaluations=1_000_000,
)
```

Each attempted macrostep computes one full step and two half steps. It accepts
only the fine two-half-step state; there is no extrapolation, tangent transport,
constraint projection or target-state correction. Error estimates divide
`difference(q_full,q_fine)` and `v_fine-v_full` by 15. Maximum componentwise
scaled error controls acceptance and step adjustment, using exponent 1/5,
safety factor 0.9 and a bounded change factor.

Configuration error scales use `atol + rtol*max(abs(h*v_start),abs(h*v_fine))`;
velocity scales use `atol + rtol*max(abs(v_start),abs(v_fine))`. Consequently atol
applies to the declared tangent displacement/velocity component units (meters
and m/s or radians and rad/s); relative scaling uses local displacement rather
than global configuration. These are **not numerically equivalent to the same
rtol/atol passed to Euclidean solve_ivp**. Compare resulting physical trajectories
and closure residuals under refinement, not tolerance numbers alone.

`evaluations` counts all RHS calls, including rejected trials; each trial costs
12 calls. `steps` counts accepted macrosteps, each comprising two half steps.
Explicit positive integer evaluation budgets prevent runaway attempts. Budget
exhaustion, time underflow, malformed outputs and callback failures raise;
partial results are not returned as successes. The integrator does not catch
and reinterpret a singular native chart as ordinary truncation error.

Initial adaptive runtime `/home/dieterolson/native-manifold-10043-15` passes **86 tests,
one optional MuJoCo test skipped**. The 18 generic tests include tighter-tolerance
error reduction, independent noncommuting rotation, the prior fourth-order
fixed tests, explicit budget enforcement, underflow and malformed output checks.
The real Pinocchio adaptive case matches the independent noncommuting solution.
Ruff/format and focused mypy pass. Exact source and receipt:
`native_evidence/manifold_adaptive_10043_15/{qualification.json,raw-source.zip}`.
Runtime14 was a superseded pre-mypy-cast checkpoint; runtime15 contains current
source. Fixed12 and branch13 evidence remains preserved.

Root owns candidate comparison with adaptive stepping and a bounded RHS budget.
Successful adaptive test problems are not closure or full-swing acceptance.

## Output-Boundary Roundoff Correction

Run51's first adaptive level completed; its second level hit an artificial
underflow at `t=0.033333333333333326`. The next output time was `1/30` seconds,
only one representable floating-point increment later. A constant-velocity
regression with `max_step=nextafter(1/30,0)` reproduced the exact failure, proving
this case was an output-boundary residual rather than a physical singularity.

The smallest correction extends an upcoming trial endpoint to the sample
boundary when the remaining gap is at most four ULPs and the trial itself spans
at least eight ULPs. This happens **before** evaluating any RK stages, so the
physical state is integrated to the actual output time; no state is relabeled,
reset or projected afterward. Truly unrepresentable requested steps and
subnormal sample intervals still raise the original explicit underflow error.

Latest runtime `/home/dieterolson/native-manifold-10043-16`: **88 passed, one
optional MuJoCo test skipped**. All 20 generic integrator tests pass locally;
Ruff/format and mypy pass. Exact source SHA for `manifold_forward.py` is
`a92c3242b09af1e58d55f1fd3a8d7dd25dc5f125665df4977fb26d845edf4d72`.
Receipt and source archive are in
`native_evidence/manifold_boundary_10043_16/{qualification.json,raw-source.zip}`.
Root owns run51's partial-result archive and the run52 repeated candidate
comparison. This correction alone does not establish trajectory acceptance.

## Next Controlled Steps

1. Retain the passed production lazy-initializer qualification and verify full
   application import integration when enabling this variant in the launcher.
2. Use the tested local-chart integrators above; establish step-size
   convergence on the constrained same-input golf replay before acceptance.
   Never feed nq30 into the existing Euclidean qdot=v path with nv27.
3. First replay the same run19 native actuator polynomial on a short qualified
   prefix, with identical original q0/qd0, model, gravity, closure and tolerances.
   Every RHS evaluation must transform the native polynomial effort at its
   actual current state. Use no state resets, arbitrary spherical moments or
   tracking feedback. Then extend to the full trajectory and compare native
   scalar and manifold rollouts under step refinement.
4. Qualify weld Jacobian, state/effort derivatives and perturbation propagation
   in tangent coordinates before plugging the alternate model into shooting.
   Check both random states and archived swing states near the transition.
5. Validate the resulting same-input profile in MATLAB R2025b. A native sixth-
   order torque profile becomes state-dependent nonpolynomial spherical moment;
   this is physical coordinate conversion, not permission to change the target
   actuation family. Singular original charts cannot uniquely recover native
   rates or moments even though quaternion orientation remains regular.

Root owns HANDOFF.md and DEVELOPMENT_LOG.md and the final commit. This subagent
made no commit and did not alter the root goal or declare full equivalence.
