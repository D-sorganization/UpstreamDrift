# Native Port Implementation Checkpoint

## Current Resume Point: 22:02 PDT Review

Read `REVIEW_AND_EXECUTION_HANDOFF_20260911_2200.md` first. It supersedes the
historical next-step lists below and reviews Gemini commit 58df3af67 plus its
actual DeskComputer runner. The purported multiple-shooting run was single
shooting and changed the warm-start polynomial time basis; it passed 0/5 gates.
Repair identity and acceptance before further long optimization runs.

The full native geometry now assembles in Pinocchio: all 27 scalar coordinates,
31 uncommented solids, native fixed transforms and a 6D right-hand closure.
Actual ControlTower execution matched the initial 25-marker native fixture
with RMS 2.895e-13 m; zero-effort/zero-velocity free fall passed with maximum
acceleration discrepancy 8.786e-13. Source, portable geometry and the historical
receipt are preserved under `native_evidence`. These results qualify initial
FK and an invariant only. Multi-pose FK, native effort mapping, damping/limits,
acceleration parity and constraint-consistent continuous integration remain.

Current targeted native conversion suite: 16 tests passed. Ruff check and
format check passed for the seven new/modified Python implementation and test
files. The full matching goal remains incomplete.

## Complete Native Solid Port Coverage and Fixed Transforms

All 31 uncommented solids now have connector mappings measured in R2025b on
ControlTower: 51 physical ports, with exact coverage checked against the source
inventory. The final batch used one frozen probe source, hashed inventory and
hashed block list; its terminal exit code is zero. Results and per-file hashes
are recorded in `native_solid_port_bindings_20260911.json`. Raw probes, frozen
source, native log and start/finish receipts are archived at
`C:/Users/diete/Repositories/simscape-tour-checkpoints/native-solid-ports-20260911-02`
and `ControlTower:C:/Users/diete/native-solid-ports-20260911-02`.

`bind_solid_ports` compares separately measured named-frame and physical-port
poses, retaining aliases if frames coincide. It rejects unmatched ports and
requires original-layout R2025b evidence. The receipt additionally checks that
every expected solid and every physical connector was covered. It does not
certify inertia or dynamics. A native upper-arm connector fixture is committed.

The native probe now supports cylinders/spheres with or without custom frames.
The first exploratory batch stopped at cylinder 14 (no custom frame); that
terminal failure was repaired, then all 31 solids were rerun in the final frozen
batch. Do not resume either batch: both are terminal and complete for their
documented scope. The final batch is the authoritative connector evidence.

`rigid_transform` converts all 14 exported rigid-transform records (13 without
commented ancestry). It preserves intrinsic follower-axis versus extrinsic
base-axis sequence order, as specified by the MathWorks Rigid Transform block.
The targeted graph/solid/transform suite has 12 passing tests; Ruff passes.
Fixed-transform conversion has unit/source-documentation evidence; it still
needs full-model native FK comparison.

Next assembly decision: test cutting the native
`GolfSwing3D_Kinetic/Grip/RightHandOnClubForce` weld to form the Pinocchio tree,
then reimpose that exact six-dimensional closure. Its hand-side standoff has
0.01 kg native mass, avoiding an invented massless leaf for the actuated wrist.
Prove that the cut removes the loop without losing any of the 27 joint
primitives. The native hip is Bushing (Px, Py, Pz, Rx, Ry, Rz), not quaternion
Six-DOF; preserve its translation/rotation and virtual-work conventions.
Assemble fixed frames/inertias first, check loop consistency and native FK,
then implement constraint-consistent forward integration and pulse parity.

## Solid Properties and First Native Frame Parity

`native_solids.py` now converts native CalculateFromGeometry cylinders and
spheres, using the existing shared primitive-inertia API. It explicitly handles
mass versus density, SI conversion, nonnegative mass (including zero-mass visual
solids), and reference-axis custom frames. Unsupported modes, unresolved active
values and degenerate axes fail instead of receiving surrogate defaults.

All 42 solid records, including commented descendants, parse. The 31 solids
without commented ancestors sum to 77.60581783574676 kg and define 41 custom
frames. This sum has not yet been verified against a native assembled-system
inertia sensor, and is not a claim about independently moving body count.

An independent ControlTower R2025b fixture reconstructed the source LUpperArm
cylinder and queried its first custom frame through KinematicsSolver at a zero
revolute angle. Native translation matches Python exactly; rotation-matrix max
difference is 2.220446049250313e-16. The fixture and its input-inventory/output
hashes are committed at
`tests/fixtures/motion_matching/native_left_upper_arm_reference.json`.
The reproducible native probe is
`motion_matching/tests/export_native_solid_probe.m` under the MATLAB tree.
Native log: `ControlTower:C:/Users/diete/native-solid-probe-green3-9967.log`.
Raw output: `ControlTower:C:/Users/diete/native-solid-probe-9967.json`, also copied
to the local `simscape-tour-checkpoints` directory. Native exit code was zero.
The combined graph/solid suite now has eight passing tests; Ruff passes.

Native API details established by execution: custom-frame lookup uses its display
name (e.g. `Top of Left Arm`), not serialized ID `Frame1`; KinematicsSolver refuses
queries between rigidly connected frames, so the isolated fixture uses a zero-angle
revolute joint. The fixture validates the first custom frame only, not all 41
frames or native inertia values. Do not broaden this claim in the handoff.

Next: convert the remaining rigid-transform rotation sequences and native joint
primitives; qualify the mapping between solid custom-frame names and physical
wire endpoints; assemble the complete tree plus grip constraint. Preserve native
Bushing Joint force/rotation conventions and both forearm rotation joints. Then
compare full native poses before acceleration/rollout parity.

## Completed Inventory and Connection Graph

Update after the initial checkpoint: both native export jobs completed with
exit code 0. Version 2 adds library references and stable physical endpoints,
tested with an actual R2025b rigid-transform/revolute-joint connection.
Both native tests pass. The Python reader has three passing contract tests
covering subsystem traversal, joint-side separation, commented ancestors,
dangling endpoints and duplicate block paths. Ruff checks pass.

The real v2 export contains 3,679 blocks. The reader reconstructed 436 physical
wire nets containing 1,473 endpoints, including correct traversal from the
hip joint base through the nested subsystem to the upstream rigid transform.
These counts include physical signal nets, not just mechanical frames.
See `native_inventory_v2_receipt_20260911.json` for hashes and artifact locations.
The local inventory and wire-net graph are in `simscape-tour-checkpoints` outside
Git; the code and receipt are committed. There is no remaining inventory job
to wait for from this checkpoint.

The native library reference identifies the hip as a Bushing Joint. Preserve
its native primitive/effort conventions when mapping to a Pinocchio tree;
do not assume that a quaternion free joint has the same effort coordinates.
All 138 multibody blocks, including commented descendants, have native library
references. The earlier exporter field `source_block` was empty at runtime;
use `library_reference` and the preserved `BlockFunction` parameter instead.

Numeric expressions for the sampled physical properties resolved, including
dimensions, mass, density, COM and inertia parameters. This is not proof those
fields are the active inertia configuration: CalculateFromGeometry requires
deriving inertia using the original BasedOnType enum, geometry and units.
For example, the string `Mass` itself resolves to a workspace numeric value.
Always interpret enums from `expression`, never from `numeric_value`.

Next: reconstruct solid custom-frame transforms, explicit SI inertia and joint
primitive transforms from this native inventory. Compare the resulting FK with
native body-frame exports before implementing full constrained rollouts.

## Scope and Verified Results

The preceding review was progress: it corrected the completed 0.80 s result and
identified concrete engine discrepancies. The full matching goal remains active.
This checkpoint adds native infrastructure, not a completed golfer port.

- ControlTower: isolated Python 3.12.3 environment at
  `/home/dieterolson/simscape-pinocchio-9967/.venv` in WSL `ControlTower-Runner`.
  Pinocchio 4.1.0 imports successfully. Dependency versions are recorded in
  `pinocchio_controltower_20260911_requirements.txt`.
- Native Pinocchio test: `test_native_constraint_dynamics.py` passed (1 test,
  0.79 s). It applies six independent forces/moments to a free body and verifies
  zero acceleration under a rigid world closure versus nonzero unconstrained
  acceleration. This is a capability test, not golfer topology or rollout parity.
- Pinocchio 4.1 requires `initConstraintDynamics(model, data, models, datas)`.
  The three-argument invocation failed in the installed native library; retain
  this evidence when adapting earlier examples or supporting other releases.
- R2025b native exporter test passed on DeskComputer. TDD began with an actual
  missing-function failure, then found serialization errors during development.
  The exporter records original expressions, resolved numeric values, units as
  separate parameter entries, block paths, source libraries, comment state and
  port connectivity. No unresolved parameter is substituted with a physical
  default. It does not compile variants or certify active topology.

## New Files and Commands

MATLAB source is under
`src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/`:
`shared/export_native_inventory.m` and `tests/test_export_native_inventory.m`.
These deliberate source files were explicitly staged because an existing broad
`motion_matching/` gitignore pattern otherwise hides them.

Run native tests in R2025b after adding those directories:

```matlab
results = runtests('test_export_native_inventory');
assertSuccess(results);
```

Run the Pinocchio capability test in the pinned environment, using the repo's
native-test lane (`-m live_simulation`). On ControlTower the standalone transferred
test was executed with:

```text
/home/dieterolson/simscape-pinocchio-9967/.venv/bin/python -m pytest /mnt/c/Users/diete/test_native_constraint_dynamics_9967.py --confcutdir=/mnt/c/Users/diete -q
```

## Full Golf Inventory Job

Launched on DeskComputer at 21:00:29 PDT with explicit
`C:/Program Files/MATLAB/R2025b/bin/matlab.exe`. Process IDs observed:
94704 launcher and 82556 worker. Revalidate command line and creation time;
do not restart based on this document alone.

Log: `C:/Users/diete/SimscapeTour9921/native-inventory-green4.log`.
Expected result: `C:/Users/diete/SimscapeTour9921/native-inventory-20260911.json`.
It loads the existing runtime checkout model, assigns UpperArmLength and
LowerArmLength from `initial_velocity_seed_qualified_r2025b.json`, exports,
and closes without saving. The job is an uncompiled parameter inventory;
it does not replay the selected candidate's torques or dynamic state.
At this checkpoint the native test passed and the golf export remained running.

## Findings That Change the Port

1. The static SLX contains a Flexible Cylindrical Beam, but its parent Flexible
   Beam Model subsystem has `Commented=on`. Verify ancestry in the native
   inventory and compiled variant selection before treating the shaft as rigid.
2. Native `golf_kinematic_schema.json` has separate left and right forearm Rz
   coordinates, LFInput and RFInput, in addition to LEInput and REInput elbows.
   The proposed MuJoCo 19-joint internal chain omits those forearm rotations.
   Do not derive the Pinocchio port from that reduced topology without a proven
   reduction. There are 27 native effort channels, not necessarily 27 independent
   unconstrained degrees of freedom.
3. Existing shared Simscape-to-URDF conversion admits unevaluated expressions,
   skipped constraints and approximations. Its helpers also have Tools ownership
   notices. Reuse compatible infrastructure but do not edit mirrored files or
   silently accept approximations as a native physical export.

## Next Execution Steps

1. Poll the exact inventory job, collect its terminal exit and hash the export.
   Inspect numeric resolution failures for physical parameters and resolve them
   from the actual candidate workspace; distinguish enum strings from failures.
2. Reconstruct physical frame connectivity through subsystem ports and fixed
   transforms. Exclude commented ancestors and qualify variant activity natively.
   Export masses/COM/inertias and joint transforms in explicit SI units.
3. Preserve the two forearm joints, both grip frames and all base efforts in the
   canonical representation. Use a tree plus engine-level closure constraints,
   preserving virtual work under coordinate changes.
4. Build the Pinocchio model and perform same-pose FK against the existing native
   body-frame exports before force-pulse or full-rollout parity tests.
5. Keep all-source and candidate identities separate from this capability probe.
   Only a full continuous native rollout can qualify the final swing.
