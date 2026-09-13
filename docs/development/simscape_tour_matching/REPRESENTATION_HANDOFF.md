# Native Joint Representation Handoff

## Authority and Current Scope

Representation epic: [#10043](https://github.com/D-sorganization/UpstreamDrift/issues/10043), under tour matching #9921. Read [Representation Epic](REPRESENTATION_EPIC_9921.md) for full acceptance. MATLAB R2025b remains the reference release. Shared representation conversions and an experimental native spherical Pinocchio builder are implemented. Robust alternate-representation trajectory equivalence, full-swing matching and MATLAB parity remain unqualified. The [Canonical Matching Handoff](../HANDOFF.md) owns current run status and acceptance evidence; this document owns representation interfaces and the next implementation task.

## Existing Infrastructure and Gaps

`pose_interchange` is the canonical convention authority. Its existing canonical-v2 state supports a quaternion floating base but scalar internal joints. Existing Pinocchio, Drake and MuJoCo pose adapters do not by themselves replace native internal gimbal coordinates with quaternion joints. The native scalar golf models remain separate engine adapters with an explicit right-hand weld. An additional Pinocchio manifold builder now exists; MuJoCo and Drake alternate builders remain outstanding. Native specification `native_evidence/native_geometry_spec_9967.json` contains HipInputX/Y/Z after three translations and two shoulder XYZ triplets. Spine, scapula and wrist joints have only two rotations and must not become unconstrained spherical joints.

Open issue [#8867](https://github.com/D-sorganization/UpstreamDrift/issues/8867) documents independent motion_pipeline and pose_interchange conventions. This work extends pose_interchange and reuses existing `se3` quaternion routines and `spatial_algebra.transforms.xtrans`; do not create another engine-specific convention stack. Migrating the existing motion_pipeline boundary remains separate work, requiring ownership coordination and parity tests.

## Implemented Contracts

`SerialRotationChart` in `pose_interchange/joint_chart.py` supports every intrinsic distinct-axis sequence using SI radians. Its rotation maps child vectors into the joint parent frame. The quaternion is Hamilton wxyz, and `rate_map` is E in omega_parent = E(q) qdot; its columns are the instantaneous parent-expressed screw axes. `angular_acceleration` includes Edot qdot, and its inverse subtracts that term. Efforts transform by tau = E.T moment, preserving instantaneous power. Inverse rate, moment and quaternion-to-coordinate maps reject ill-conditioned configurations with `SingularChartError`, never a silent pseudoinverse. Forward orientation remains defined at lock.

Quaternion-to-angle conversion requires explicit reference coordinates. Its generic default selects the nearest of both Tait-Bryan branches plus integer windings; `preserve_middle_branch=True` retains the reference middle-angle branch and is required when translating the native actuator trajectory. The native manifold engine uses this mode because switching branches can change the native actuator map. A quaternion alone cannot retain revolution count. Discrete nearest-branch selection is not a global continuity guarantee: sample sufficiently and retain native branch metadata. `max_condition` is a numerical rejection threshold, not a physical joint limit.

`FixedFrameTransport` in `pose_interchange/frame_transport.py` holds explicit source and target names and T_target_source. It transports rigid poses, angular-first spatial twists and moment-first wrenches, including origin shifts and virtual-work preservation. Twists refer to frame origins, not arbitrary marker point velocities. This is a fixed reference-frame transform; moving-frame accelerations require additional transport terms and are not advertised as implemented. Stored transforms are copied and marked read-only.

`NativeJointStateAdapter` in `pose_interchange/native_joint_state.py` derives exactly three rotational groups (hip and both shoulders) from the archived real native specification. It exports named native q/qd/qdd/tau to immutable rotational tuples (wxyz, base angular velocity/acceleration/moment), preserving all 18 other scalar coordinates and their conjugate efforts. Restore uses explicit reference angles and verifies native-joint-manifold-v1 convention tag, canonical specification hash and inventory. Fixed parent-to-base and child-to-follower metadata are retained. This common named-SI boundary is the one consumed by all three native providers; no raw engine array layout assumptions are made. It is a state interchange adapter, not a new simulator. The rotational quaternion omits fixed attachment rotations and the hip translation prefix; full body pose requires composition with those retained transforms. The canonical-JSON specification fingerprint differs from the raw model file hash used by existing replay receipts.

The package public API advances from 2.0.0 to 2.1.0 for these additive exports. Canonical pose/state schema versions are unchanged. Engine-native quaternion ordering and local/world angular conventions belong at adapters; canonical ordering must never be inferred from four numerical entries.

## Native Motion Sequence Milestone

`NativeMotionSequence`, `export_native_motion` and `restore_native_motion` are
now lazy public exports from `pose_interchange`. The batch provider delegates
all numerical conversions to `NativeJointStateAdapter`. It owns immutable time,
model fingerprint, coordinate/primitive inventory, group fixed frames, manifold
samples and the original native q reference at every sample. Restore always
preserves the middle-angle branch; singular failures identify sample and time.
Efforts are SI native primitive-conjugate efforts, not raw world-force polynomial
inputs. Group quaternions remain joint-local orientations, not world body poses.

```python
from src.shared.python.pose_interchange import (
    NativeJointStateAdapter, export_native_motion, restore_native_motion,
)

adapter = NativeJointStateAdapter(native_specification)
# Each matrix is [sample, coordinate] in adapter.coordinate_order.
motion = export_native_motion(adapter, time_s, q, qd, qdd, primitive_efforts)
q_copy, qd_copy, qdd_copy, effort_copy = restore_native_motion(adapter, motion)
```

This milestone provides in-memory conversion. File serialization, user interface,
complete frame transport and all-engine trajectory qualification remain open.
Eighteen new tests cover branch/winding, quaternion sign, power, deep ownership,
metadata and malformed/singular samples; RED was observed for missing module and
public exports. Root independently passes89 related representation tests, Ruff
and pinned mypy. This includes the existing MuJoCo FK smoke, not full dynamics.

## Verification

TDD red states were observed before implementation: absent joint_chart and frame_transport modules, and absent inverse acceleration. Fifteen joint-chart tests cover all six sequences against independent SciPy rotations, rotation-matrix finite differences for angular velocity, finite differences for angular acceleration, branch/winding and quaternion sign equivalence, power invariance, inverse acceleration, singularity errors and malformed inputs. Five frame tests verify translated origin signs, dual wrench power, inverse maps, transform ownership and invalid transforms. Seven native state tests verify real specification inventory, scalar/body/fixed-frame preservation, identity and convention rejection, singularity failure and an actual local MuJoCo native frame roundtrip (1e-12 absolute comparison tolerance). The final focused command passes 30 tests with `-m unit`, including the installed MuJoCo smoke. This checks quaternion state roundtrip through existing native MuJoCo FK, not alternate-model dynamics. Ruff and focused mypy pass. The full pose_interchange regression passed (247 tests at that checkpoint, before the final native adapter refinements). These mathematical tests are not live-engine equivalence evidence.

## Experimental Pinocchio Builder and Acceptance Limits

[NativeManifoldPinocchioModel](../../../src/engines/physics_engines/pinocchio/python/native_manifold_model.py) constructs eligible hip and shoulder spherical joints while reusing the scalar builder's solids, placements, frames and weld. Its representation is `native-pinocchio-spherical-xyzw-body-v1`. `native_state` maps native coordinates/rates/primitive efforts to engine q/v/tau; `native_coordinates` restores the original middle-angle branch; `integrate`, `difference` and `difference_rate` use Pinocchio manifold operations. `acceleration_from_native_efforts` transforms native efforts at the current configuration, and `native_accelerations` includes the convective conversion back to native qdd. This is an explicit alternate builder, not quaternion storage over scalar integration.

The [Canonical Matching Handoff](../HANDOFF.md#latest-verified-result) records one 0.85-second same-input parity pass (run58), followed by a tighter-step run59 that fails native velocity parity. The isolated pass therefore does not qualify robust trajectory convergence. Pointwise acceleration and effort audits do not establish trajectory or MATLAB equivalence. The manifold implementation is currently slower than the scalar reference in those experiments; no general speed improvement is established. Keep the scalar native baseline available for matching while qualifying alternate integration. Historical test counts below describe their original checkpoints rather than current total coverage.

## Next Bounded Task: Native Motion Files and Consumers

The validated in-memory envelope and batch converters above are implemented.
Do not recreate them. Coordinate #10043 ownership before edits and #8867 before
changing motion_pipeline consumers. Continue in these bounded TDD stages:

1. Add `load_native_motion` and `save_native_motion` around the existing envelope,
   with an explicit file schema and separately labeled raw model artifact hash.
   Validate version, units, quaternion/frame conventions, coordinate inventory
   and sample counts through the existing constructor. Do not infer conventions.
   Keep CanonicalPose and initial-state formats unchanged. Test malformed files
   and atomic writes that preserve the previous result on interruption.
2. Roundtrip a real-specification multi-sample motion through disk and the public
   export/restore API, including nonzero velocity/acceleration, both angle
   branches/windings, quaternion sign equivalence and virtual-work preservation.
   Preserve failure index/time; never silently skip a singular sample.
3. Add a distinct motion import/export command or service alongside pose-only
   routing. Record executable example and output location. The current in-memory
   example is not a file-format or UI completion claim.
4. Expose only supported fixed-frame pose/twist/wrench conversions by delegating
   FixedFrameTransport. Do not apply a rotation alone to a twist at a shifted
   origin or advertise moving-frame acceleration transport without its terms.
5. Feed the restored named native sequence into real Pinocchio, MuJoCo and Drake
   frame and acceleration providers. Archive runtime versions, input hashes and
   transform/physical-velocity/acceleration/power discrepancies. A serializer or
   FK roundtrip does not qualify alternate dynamics. Independent R2025b and
   uninterrupted representative trajectory parity remain later acceptance gates.

## Controlled Follow-On Implementation

1. Derive native group inventory from the hashed specification, preserving coordinate names, primitive sequence, fixed parent/child transforms, all remaining scalar joints and actuator map. Only collapse contiguous three-axis primitives inside a single native joint where no intervening body inertia, frame attachment or constraint exists. The present specification permits investigation of hip rotation and shoulder groups; verify every body and constraint attachment before construction.
2. Keep both builders: immutable native scalar reference and explicit alternate-manifold variant. Carry representation identity separately from physical-model hash. Do not overwrite native URDF: URDF lacks these quaternion/constraint semantics. Preserve the model specification and sidecar as authority.
3. Continue qualification of the implemented Pinocchio spherical builder and its manifold operations described above. Keep its native effort mapping and branch preservation. Qualify tangent derivatives before optimization, repeat representative same-input trajectory comparisons with convergence evidence, and retain universal joints as two-DOF mechanisms. Do not reimplement the builder from this historical plan.
4. In MuJoCo, build eligible ball joints in MJCF and retain exact closed-loop enforcement; do not accept a soft equality as equivalent by default. Handle qpos quaternion size separately from qvel. Apply transformed efforts through the documented tangent convention; record the inspected engine version and convention tests.
5. In Drake, build an explicit supported quaternion joint/free-body representation with unchanged attachments and constraints. Verify nq/nv layouts and spatial velocity reference points using real engine queries. Never infer layout by checking which four numbers happen to have unit norm.
6. For each engine, compare all body/marker transforms, closure pose/Jacobian, rates, kinetic energy, instantaneous power and acceleration away from singularities at random states and saved swing states. Include qdot-to-omega and convective acceleration. Then compare trajectories under identical native actuator profiles using step-size convergence. Same FK is insufficient.
7. At the native gimbal lock, report that recovering unique original rates/efforts is impossible. A spherical representation regularizes orientation coordinates but cannot provide a unique native torque mapping at lock. Separate genuine physical model change from coordinate reparameterization. Do not claim full equivalence by passing through singular states with an arbitrary pseudoinverse.
8. Replay qualified profiles on MATLAB R2025b. A sixth-order polynomial in native scalar actuator torques generally becomes state-dependent, nonpolynomial moments in the alternate representation. Preserve the polynomial in its declared actuator coordinates and transform its physical effort during simulation; this coordinate conversion is not tracking feedback.
9. Add motion_pipeline delegating shims and cross-representation parity tests under coordinated #8867 work. UI/export must identify frame, quaternion order, native branch, units, representation version, model hash, and acceptance status.

## Resume Commands

Run from the UpstreamDrift worktree:

```powershell
python -m pytest tests/unit/pose_interchange/test_native_motion_sequence.py tests/unit/pose_interchange/test_joint_chart.py tests/unit/pose_interchange/test_frame_transport.py tests/unit/pose_interchange/test_public_surface.py tests/unit/pose_interchange/test_native_joint_state.py -q --no-cov -m unit
python -m mypy src/shared/python/pose_interchange/joint_chart.py src/shared/python/pose_interchange/frame_transport.py --follow-imports=silent
python -m ruff check src/shared/python/pose_interchange/joint_chart.py src/shared/python/pose_interchange/frame_transport.py
```

Root agent owns the main handoff and development log. This subtask must not mark the parent goal or representation epic complete until alternate builders and full runtime/reference acceptance are verified.
