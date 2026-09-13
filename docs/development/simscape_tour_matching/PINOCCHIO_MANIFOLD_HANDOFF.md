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
  using the explicit branch/winding reference.
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

## Next Controlled Steps

1. Retain the passed production lazy-initializer qualification and verify full
   application import integration when enabling this variant in the launcher.
2. Implement and test a Lie-group time integrator, or an explicit quaternion
   derivative integrator with a documented stage-normalization policy. Never
   feed nq30 into the existing Euclidean qdot=v path with nv27. Establish order
   and step-size convergence independently before applying swing acceptance.
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
