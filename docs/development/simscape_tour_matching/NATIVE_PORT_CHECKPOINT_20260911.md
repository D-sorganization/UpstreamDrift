# Native Port Implementation Checkpoint

## Program Expansion and First Continuous Diagnostic

OpenSim is now part of the working program under
[Epic #10003](https://github.com/D-sorganization/UpstreamDrift/issues/10003).
A parallel planning agent committed/pushed its detailed seven-stage epic and
handoff as 0517dea90 in `Worktrees/UpstreamDrift-opensim-10003`, branch
`docs/10003-opensim-matching-epic`. OpenSim implementation has not started;
the requested user check-in is the next program milestone. Pinocchio's
analogous lower-agent plan is `PINOCCHIO_EXECUTION_PLAN.md` in this directory.

The shared `continuous_forward.py` helper was developed red-to-green with
seven unit tests, including analytic nonconstant sextic forcing, absolute
time, invalid clocks and invalid derivatives. It integrates one explicit
Euclidean state equation from t=0 without target injection or feedback.
Native adapters remain responsible for physical constraints and validation.

Actual ControlTower Pinocchio first-prefix run completed in 0.739 s for
9002 derivative evaluations, ending at the last native sample before 0.60 s:
0.5992645005689832 s. Its maximum grip pose residual is 3.68e-11 and velocity
residual 1.15e-10. Maximum coordinate difference from the saved native replay
is 0.002785 and rate difference 0.06994 (mixed coordinate units; inspect
per-coordinate values). This is preliminary evidence, not an accepted
continuous-parity result or an end-to-end optimizer speedup benchmark.

Raw result: `simscape-tour-checkpoints/native-continuous-600ms-9967-01.json`
locally and `ControlTower:C:/Users/diete/native-continuous-600ms-9967-01.json`.
Compact summary: `native_evidence/native_continuous_prefix_diagnostic_9967.json`.
Reference: `native-rollout-reference-9967-01.json` in the local archive and on
both remote hosts (under SimscapeTour9921 on DeskComputer, user root on
ControlTower). It preserves exact native time/q/qd and saved polynomial
coefficients, exported by a fresh R2025b process from the existing MAT replay.
No new optimized or tighter-tolerance native replay has been run yet.

Reproducer: `check_native_continuous_rollout.py` with module/integrator/spec/
fixture/output paths and `--duration 0.6`. Source identities are in the raw
receipt. ControlTower's isolated Pin environment now additionally has SciPy
1.18.1. Next: derive marker errors, perform step/tolerance convergence, and
compare a fresh tighter-tolerance R2025b baseline before extending the horizon.
All native export and first-prefix diagnostic jobs from this checkpoint are
terminal; do not restart them as if still active.

## Current Result: Acceleration Parity Repaired and Verified

The failure below is now resolved for the tested states and inputs. The port
added joints in breadth-first order, interleaving left/right subtrees. This
violated Pinocchio's compact subtree indexing assumption: even with positive
individual body inertias, CRBA produced a negative mass-matrix eigenvalue
(-0.41694). The constrained stationary mobility was also indefinite.
The depth-first requirement is explicit in the
[Pinocchio Model API](https://docs.ros.org/en/ros2_packages/rolling/api/pinocchio/generated/structpinocchio_1_1ModelTpl.html).

`NativePinocchioModel` now orders the input tree depth-first before adding
joints. It preserves coordinate identities, geometry, masses, inertias, effort
mapping and closure. Invalid/disconnected trees are rejected. New unit tests
failed before implementation and pass afterward (2 tests). No native physical
parameters were changed to achieve agreement.

Actual execution evidence after correction:

- DeskComputer R2025b ran baseline plus all 27 unit input cases, each at exactly
  the same assembled initial q and zero qd. All cases completed and include
  complete actuator-log audits. Raw case files are in
  `C:/Users/diete/SimscapeTour9921/native-input-pulses-9967-01` on DeskComputer,
  copied to local `simscape-tour-checkpoints/native-input-pulses-9967-01` and
  ControlTower `C:/Users/diete/native-input-pulses-9967-01`.
- Actual ControlTower Pinocchio stationary response comparison passes with
  `--require-parity`: baseline max discrepancy 5.10e-12 and maximum scaled
  response discrepancy 9.68e-12. Receipt: `native_input_pulse_parity_9967.json`.
- The same six moving-state acceleration tests through 0.80 s now pass.
  Maximum absolute discrepancy 3.63e-9, maximum scaled discrepancy 2.54e-10.
  Receipt: `native_dynamics_parity_9967.json`. The original geometry comparison
  still passes. These are mixed generalized-coordinate diagnostics; do not
  label a mixed maximum exclusively as rad/s^2 or m/s^2.
- Corrected mass matrix minimum eigenvalue is positive (2.259e-6). Constrained
  mobility minimum eigenvalue is -1.79e-13, consistent with roundoff at the
  constrained null directions. `native_mass_matrix_parity_9967.json` records
  the dense KKT diagnostic and component inertias; it is not itself a full
  native mass-matrix measurement.

Reproduction: `export_native_input_pulses.m`, `check_native_input_pulses.py`,
and `diagnose_pinocchio_pulse_solver.py` under `native_evidence/reproduction`.
The pulse checker requires module/spec/cases/output paths and accepts
`--require-parity` to fail if agreement or common-state conditions are violated.
All native checks used production source copied to
`ControlTower:C:/Users/diete/native_model_9967_dfs.py`. The pulse receipt records
its hash and all 28 native case hashes. Raw historical failed receipts remain
in `simscape-tour-checkpoints`; do not mistake them for the latest results.

Next: implement and verify continuous forward integration with a single initial
state and native polynomial effort mapping. Compare native time histories,
closure residuals and solver-step convergence through transition, then extend
to the full observed swing. Preserve no-feedback/no-target-reset acceptance.
Pinocchio is now qualified for these sampled acceleration tests, not yet for
continuous-rollout equivalence or full-swing matching. The portable native JSON
retains the closure; plain URDF alone cannot carry the entire closed-loop
execution contract. Export a tree plus explicit closure/actuation sidecar when
adding interoperability, and test each engine's reconstruction independently.

## Historical Failure: Acceleration Parity

The next actual native check found a material dynamics mismatch. Do not run
optimization on this Pinocchio model as a native-equivalent oracle yet.
`native_evidence/native_dynamics_check_9967.json` preserves the failure:
maximum absolute generalized acceleration discrepancies at the six samples
are approximately 54.49, 21.24, 277.24, 396.25, 13267.89 and 22555.82
(translation channels in m/s^2, rotation channels in rad/s^2; do not present
the combined maximum as a single physical unit). FK still passes.

Evidence that narrows the issue:

- All 27 native actuator logs are present. Existing
  `audit_golf_actuator_torques` agrees with the polynomial efforts after rotating
  the three world forces into hip-base axes: max force error 3.41e-13 N,
  max torque error 2.84e-14 Nm. This verifies the existing log audit, not an
  independent virtual-work/pulse qualification of every actuator route.
- Full primitive audit (including unprefixed Revolute parameter names) records
  zero stiffness/damping, disabled limits, InputTorque and ComputedMotion in
  `native_passive_joint_audit_9967.json`. This is uncompiled inventory evidence.
- Native total mass 77.60581783574678 kg versus Pinocchio 77.60581783574676 kg.
  Native initial world COM [0.9366603077856251, 0.055828222965153544,
  1.3489299522131155] matches Pinocchio within about 6e-15 m.
- Native q/qd satisfy the Pinocchio grip pose/velocity constraint to numerical
  precision. J times the acceleration difference is around 1e-9 or smaller:
  the discrepancy is predominantly within the allowed motion, not a gross
  closure violation. Do not infer that all constraint-force conventions pass.
- Native finite-difference checks broadly support the logged derivatives
  before transition (sampled qdd discrepancies around 0.02-0.03 rad/s^2 versus
  the much larger Pinocchio mismatch). Rapid-transition finite differences
  are less accurate and are diagnostic only.
- The inertia-signature diagnostic matches 10 of 12 attached sensor groups
  to native COM/inertia signatures. The two hand sensors do not match the
  fully weld-collapsed groups used in this diagnostic. Sensor extent can stop
  at joints while the Pinocchio tree aggregates welded bodies; this is not
  yet proof of an inertia error. Resolve sensor routing and extent before
  drawing a conclusion. Symmetric groups have duplicate numeric signatures.

Reproduction files are `export_native_pose_samples.m` (optional fourth
argument: geometry specification), `check_native_pose_samples.py` with
`--check-accelerations`, and `check_native_inertia_samples.py`. All under
`native_evidence/reproduction`. Native R2025b exports 01, 02 and 03 completed;
03 fixes initial-sample extraction for timeseries whose time axis is last.
The tracked dynamics fixture comes from native export 03. Raw executed files
remain in local `simscape-tour-checkpoints`; the acceleration receipt hashes
refer to raw export 01 and the then-executed module, not reformatted JSON.
Both exports represent the same saved candidate states; 03 adds diagnostics.
Native log: `DeskComputer:C:/Users/diete/SimscapeTour9921/native-dynamics-samples-9967-03.log`.
Pinocchio diagnostic intentionally exits nonzero because parity fails.

Next decisive experiment: native zero-effort and 27 single-channel force/torque
pulses at the same initial pose with zero rates. Export the actual assembled
q/qd and qdd for every pulse. Compare baseline-subtracted acceleration response
columns with Pinocchio to isolate inertia/actuation from gravity and velocity
bias. Reuse R2025b `simulate_with_coefficients`, native initial-state overrides,
the actuator audit and explicit SI extraction. Keep polynomial and primitive
effort frames distinct. If the response matrix differs, inspect actuator
virtual work and body-group inertias before time integration. If it matches,
isolate gravity and velocity-dependent bias next. No inverse dynamics is
required for this experiment. Do not change physics to fit the observed
accelerations without locating and testing the discrepancy.

## Six Native Poses Verified Through Transition

The new `export_native_pose_samples.m` diagnostic ran successfully in a fresh
DeskComputer MATLAB R2025b process. It loaded the saved 0.80 s native replay,
selected actual raw states near 0, 0.4, 0.6, 0.7, 0.75 and 0.8 seconds, and
queried all schema frame poses using the native KinematicsSolver. It did not
interpolate coordinates or save the source model.

`check_native_pose_samples.py` then executed the actual production Pinocchio
module on ControlTower. All six comparisons passed: maximum position component
error 1.333e-15 m and maximum rotation-matrix component error 3.442e-15.
The fixture and receipt are `native_evidence/native_pose_samples_9967.json`
and `native_evidence/native_multipose_check_9967.json`. The receipt hashes refer
to the raw executed files, preserved under local `simscape-tour-checkpoints`
as `native-pose-samples-9967-01.json`, `native-multipose-check-9967-01.json`,
and `native_geometry_spec_9967.json`; tracked JSON may differ in formatting.

Native log: `DeskComputer:C:/Users/diete/SimscapeTour9921/native-pose-samples-9967-01.log`.
Both MATLAB and Pinocchio diagnostic exit codes were zero. The native query
uses the saved replay's arm geometry and all exact measured joint coordinates.
This supports multi-pose geometry/coordinate parity, not effort, inertia or
continuous dynamics equivalence. Next: inspect native upstream effort routing
and active passive-joint parameters, then compare same-state qdd under force
pulses before integrating complete trajectories.

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
