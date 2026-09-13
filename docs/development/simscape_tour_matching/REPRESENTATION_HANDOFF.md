# Native Joint Representation Handoff

## Authority and Current Scope

Representation epic: [#10043](https://github.com/D-sorganization/UpstreamDrift/issues/10043), under tour matching #9921. Read [Representation Epic](REPRESENTATION_EPIC_9921.md) for full acceptance. MATLAB R2025b remains the reference release. This checkpoint supplies tested representation conversions; it does not establish alternate quaternion engine dynamics or full-swing parity.

## Existing Infrastructure and Gaps

`pose_interchange` is the canonical convention authority. Its existing canonical-v2 state supports a quaternion floating base but scalar internal joints. Existing Pinocchio, Drake and MuJoCo pose adapters do not by themselves replace native internal gimbal coordinates with quaternion joints. The native golf models are separate engine adapters, with scalar coordinates and an explicit right-hand weld. Native specification `native_evidence/native_geometry_spec_9967.json` contains HipInputX/Y/Z after three translations and two shoulder XYZ triplets. Spine, scapula and wrist joints have only two rotations and must not become unconstrained spherical joints.

Open issue [#8867](https://github.com/D-sorganization/UpstreamDrift/issues/8867) documents independent motion_pipeline and pose_interchange conventions. This work extends pose_interchange and reuses existing `se3` quaternion routines and `spatial_algebra.transforms.xtrans`; do not create another engine-specific convention stack. Migrating the existing motion_pipeline boundary remains separate work, requiring ownership coordination and parity tests.

## Implemented Contracts

`SerialRotationChart` in `pose_interchange/joint_chart.py` supports every intrinsic distinct-axis sequence using SI radians. Its rotation maps child vectors into the joint parent frame. The quaternion is Hamilton wxyz, and `rate_map` is E in omega_parent = E(q) qdot; its columns are the instantaneous parent-expressed screw axes. `angular_acceleration` includes Edot qdot, and its inverse subtracts that term. Efforts transform by tau = E.T moment, preserving instantaneous power. Inverse rate, moment and quaternion-to-coordinate maps reject ill-conditioned configurations with `SingularChartError`, never a silent pseudoinverse. Forward orientation remains defined at lock.

Quaternion-to-angle conversion requires explicit reference coordinates and selects the nearest of both Tait-Bryan branches plus integer windings. A quaternion alone cannot retain revolution count. Discrete nearest-branch selection is not a global continuity guarantee: sample sufficiently and retain native branch metadata. `max_condition` is a numerical rejection threshold, not a physical joint limit.

`FixedFrameTransport` in `pose_interchange/frame_transport.py` holds explicit source and target names and T_target_source. It transports rigid poses, angular-first spatial twists and moment-first wrenches, including origin shifts and virtual-work preservation. Twists refer to frame origins, not arbitrary marker point velocities. This is a fixed reference-frame transform; moving-frame accelerations require additional transport terms and are not advertised as implemented. Stored transforms are copied and marked read-only.

`NativeJointStateAdapter` in `pose_interchange/native_joint_state.py` derives exactly three rotational groups (hip and both shoulders) from the archived real native specification. It exports named native q/qd/qdd/tau to immutable rotational tuples (wxyz, base angular velocity/acceleration/moment), preserving all 18 other scalar coordinates and their conjugate efforts. Restore uses explicit reference angles and verifies native-joint-manifold-v1 convention tag, canonical specification hash and inventory. Fixed parent-to-base and child-to-follower metadata are retained. This common named-SI boundary is the one consumed by all three native providers; no raw engine array layout assumptions are made. It is a state interchange adapter, not a new simulator. The rotational quaternion omits fixed attachment rotations and the hip translation prefix; full body pose requires composition with those retained transforms. The canonical-JSON specification fingerprint differs from the raw model file hash used by existing replay receipts.

The package public API advances from 2.0.0 to 2.1.0 for these additive exports. Canonical pose/state schema versions are unchanged. Engine-native quaternion ordering and local/world angular conventions belong at adapters; canonical ordering must never be inferred from four numerical entries.

## Verification

TDD red states were observed before implementation: absent joint_chart and frame_transport modules, and absent inverse acceleration. Fifteen joint-chart tests cover all six sequences against independent SciPy rotations, rotation-matrix finite differences for angular velocity, finite differences for angular acceleration, branch/winding and quaternion sign equivalence, power invariance, inverse acceleration, singularity errors and malformed inputs. Five frame tests verify translated origin signs, dual wrench power, inverse maps, transform ownership and invalid transforms. Seven native state tests verify real specification inventory, scalar/body/fixed-frame preservation, identity and convention rejection, singularity failure and an actual local MuJoCo native frame roundtrip (1e-12 absolute comparison tolerance). The final focused command passes 30 tests with `-m unit`, including the installed MuJoCo smoke. This checks quaternion state roundtrip through existing native MuJoCo FK, not alternate-model dynamics. Ruff and focused mypy pass. The full pose_interchange regression passed (247 tests at that checkpoint, before the final native adapter refinements). These mathematical tests are not live-engine equivalence evidence.

## Controlled Follow-On Implementation

1. Derive native group inventory from the hashed specification, preserving coordinate names, primitive sequence, fixed parent/child transforms, all remaining scalar joints and actuator map. Only collapse contiguous three-axis primitives inside a single native joint where no intervening body inertia, frame attachment or constraint exists. The present specification permits investigation of hip rotation and shoulder groups; verify every body and constraint attachment before construction.
2. Keep both builders: immutable native scalar reference and explicit alternate-manifold variant. Carry representation identity separately from physical-model hash. Do not overwrite native URDF: URDF lacks these quaternion/constraint semantics. Preserve the model specification and sidecar as authority.
3. In Pinocchio, replace eligible groups with spherical manifold joints and use native `integrate`/`difference`. Map qdot to angular velocity and scalar effort to parent moment via the shared chart, then rotate to the joint tangent convention actually used by the installed engine. Transform constraints, acceleration and all state Jacobians consistently. Universal joints remain constrained two-DOF mechanisms.
4. In MuJoCo, build eligible ball joints in MJCF and retain exact closed-loop enforcement; do not accept a soft equality as equivalent by default. Handle qpos quaternion size separately from qvel. Apply transformed efforts through the documented tangent convention; record the inspected engine version and convention tests.
5. In Drake, build an explicit supported quaternion joint/free-body representation with unchanged attachments and constraints. Verify nq/nv layouts and spatial velocity reference points using real engine queries. Never infer layout by checking which four numbers happen to have unit norm.
6. For each engine, compare all body/marker transforms, closure pose/Jacobian, rates, kinetic energy, instantaneous power and acceleration away from singularities at random states and saved swing states. Include qdot-to-omega and convective acceleration. Then compare trajectories under identical native actuator profiles using step-size convergence. Same FK is insufficient.
7. At the native gimbal lock, report that recovering unique original rates/efforts is impossible. A spherical representation regularizes orientation coordinates but cannot provide a unique native torque mapping at lock. Separate genuine physical model change from coordinate reparameterization. Do not claim full equivalence by passing through singular states with an arbitrary pseudoinverse.
8. Replay qualified profiles on MATLAB R2025b. A sixth-order polynomial in native scalar actuator torques generally becomes state-dependent, nonpolynomial moments in the alternate representation. Preserve the polynomial in its declared actuator coordinates and transform its physical effort during simulation; this coordinate conversion is not tracking feedback.
9. Add motion_pipeline delegating shims and cross-representation parity tests under coordinated #8867 work. UI/export must identify frame, quaternion order, native branch, units, representation version, model hash, and acceptance status.

## Resume Commands

Run from the UpstreamDrift worktree:

```powershell
python -m pytest tests/unit/pose_interchange/test_joint_chart.py tests/unit/pose_interchange/test_frame_transport.py tests/unit/pose_interchange/test_public_surface.py tests/unit/pose_interchange/test_native_joint_state.py -q --no-cov -m unit
python -m mypy src/shared/python/pose_interchange/joint_chart.py src/shared/python/pose_interchange/frame_transport.py --follow-imports=silent
python -m ruff check src/shared/python/pose_interchange/joint_chart.py src/shared/python/pose_interchange/frame_transport.py
```

Root agent owns the main handoff and development log. This subtask must not mark the parent goal or representation epic complete until alternate builders and full runtime/reference acceptance are verified.
