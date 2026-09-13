# MuJoCo Manifold Handoff

## Status and Scope

The opt-in `src/engines/physics_engines/mujoco/python/native_spherical_mjcf.py`
exports a spherical representation by transforming the canonical MJCF. The default
`export_native_mjcf` and `NativeMujocoModel` are unchanged. No dynamics adapter,
optimizer wiring, remote computation, or stock `mj_step` qualification is included.

Only the three validated same-joint XYZ groups are replaced: hip, left shoulder,
and right shoulder. The hip translations and eighteen scalar coordinates remain.
The canonical exporter places each primitive sequence on one body with colocated
rotation axes. Replacing its three hinge elements with one ball therefore requires
no new bodies, transforms, inertia calculations, sites, or weld construction.
`NativeJointStateAdapter` supplies group validation and quaternion conversion.

## Evidence and Reproduction

Local Windows Python 3.13 and MuJoCo 3.3.4 executed the real compile/FK test.
TDD began with a missing-module collection failure; five new test cases now pass.
Ruff passes and focused source mypy passes. Run from the repository root:

```powershell
python3 -m pytest tests/unit/pose_interchange/test_native_spherical_mjcf.py -q --no-cov -m 'unit or live_simulation'
python3 -m ruff check src/engines/physics_engines/mujoco/python/native_spherical_mjcf.py tests/unit/pose_interchange/test_native_spherical_mjcf.py
python3 -m mypy --follow-imports=silent src/engines/physics_engines/mujoco/python/native_spherical_mjcf.py
```

The live marker is excluded by default repository selection; explicitly include it
as above. The spherical test actually executes locally. The earlier native batch
consumer tests separately skip Drake and Pinocchio when their runtimes are absent.

The real-specification fixture is
`docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json`.
These are manufactured states at three distinct poses, including >2pi winding and
nonprincipal middle-angle branches. They are not closure-feasible C3D trajectories.
All body and site positions/rotation matrices agree with the scalar export at
absolute tolerance 2e-14. Compile yields 3 balls + 18 scalar joints, nq=30, nv=27.
Inertial COM/orientation, body masses/inertias, site placements, weld data and solver
parameters compare exactly; gravity is identical and damping, armature, friction,
stiffness and effective limits remain zero. Removing joint XML elements leaves
identical documents, and retained scalar joint attributes compare exactly.
This preserves all canonical solid aggregation; it does not independently requalify
that aggregation against MATLAB.

## Identity and Layout

For the checked-in fixture used here:

| Identity                      | SHA256                                                             |
| ----------------------------- | ------------------------------------------------------------------ |
| Source model bytes            | `0202c8b2e08e8f2664a2dae453a58caa0eae2cabcea4c067fafdc4bc49933c1d` |
| Canonical specification JSON  | `db7cd60496edd1adef789b965bce897f12e20d9460b957a31d7fd99c8fd31518` |
| Canonical scalar MJCF         | `ca9a923b27e303b1e3bec721ba1b561c8fc4f12526cc821a3f8cafa95319671a` |
| Spherical MJCF representation | `ba0dc584a5971d683b9e463d6e20a584a3f3d9e9a93be9644d50c32487673975` |

The source-byte hash is deliberately distinct from the fitted remote runtime's
historical b817fea... model hash. This test qualifies its checked-in fixture only;
repeat on the exact current runtime snapshot before claiming runtime equivalence.
Source physics identity is retained, while `representation_sha256` changes with
MJCF representation. Hash equality alone is not physical or dynamics qualification.

Metadata `ball_joints` maps each full native joint name to generated joint name,
original ordered coordinate triple, child body, XYZ sequence, four wxyz qpos
components and three tangent dimensions. `native_ball_0`, `_1`, `_2` correspond to
adapter group order. Scalar names remain unchanged. Resolve compiled addresses
with `jnt_qposadr` and `jnt_dofadr`; do not infer layout from native coordinate order.

MuJoCo's [joint reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint)
specifies quaternion ball rotations with three rotational freedoms and permits
slide joints on the same body. Its
[orientation convention](https://mujoco.readthedocs.io/en/stable/modeling.html#frame-orientations)
uses scalar-first quaternions. Those conventions were exercised by compilation and
FK, not merely assumed from documentation.

## Next Sequential Gates

1. Implement and test explicit manifold qpos and tangent velocity/acceleration
   packing through the shared adapter. Verify MuJoCo's ball angular-velocity frame
   using primary source and finite-difference kinematics; the adapter's physical
   angular vectors are joint-base expressed. Do not copy them into qvel blindly.
2. Map native primitive-conjugate efforts to physical joint moments and then the
   compiled tangent frame. Prove virtual-power equality; world-frame polynomial
   inputs must first pass the existing native actuation map. Preserve the original
   polynomial channels and branch references for force interpretation.
3. Compare mass/bias and constraint Jacobians under the complete tangent map,
   including the acceleration map's time derivative. Reuse the explicit six-row
   rigid grip solve; exported compliant equality is not a substitute.
4. Qualify same-state physical acceleration and constraint residuals against the
   exact source-bound native runtime, then bounded continuous replay from original
   q0/qd0. Preserve independent state/marker/closure gates and failed receipts.
5. Only after those gates, connect fitting/sensitivities. A spherical coordinate
   chart can remove Euler kinematic poles but does not establish that a global
   sextic tour swing fit exists or that the native Euler actuator inverse remains
   well conditioned near its poles.

Do not present this exporter as a completed alternate dynamics model or full-swing
match. No native optimizer or acceptance threshold has changed.
