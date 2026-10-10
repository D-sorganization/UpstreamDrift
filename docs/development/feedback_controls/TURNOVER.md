# Feedback Controls Planning Turnover

## Current State

Repository: UpstreamDrift. Topic branch: `docs/feedback-controls-11784`. Commit: `SELF` (resolve with `git rev-parse HEAD`). Documentation PR: not created at packet preparation; the governing epic records its final URL/state. Epic: https://github.com/D-sorganization/UpstreamDrift/issues/11784.

This packet completes research/planning only. No controller, native simulation, muscle fit, qualification receipt or video was produced. All implementation issues stay open. Expanded target: all registered models across six engines with defensible parity; ultimate muscle-driven OpenSim matching and independent excitation replay.

## Read First and Resume

1. Read repo AGENTS/CLAUDE/SPEC, the linked epic, this packet's implementation plan and UpstreamDrift design.
2. Review current MOSAIC #11532 and Same-Input Parity #11605 providers and live ownership; do not create competing estimators/integrators.
3. Claim/lease the bounded first task (F01) through Repository_Management. Freeze capabilities/metrics before human data fitting.
4. Write failing independent behavioral tests; implement behind small facades with DbC, LoD and DRY; update governed manual QMD/registries and change fragments with each implementation PR.
5. Record native evidence and remaining gates. Torque tracking/replay is an intermediate milestone; muscle-only acceptance requires bounded reserves and complete-state excitation replay under declared contact.

## Validation and Boundaries

Planning validation covers issue/dependency resolution, document structure/local links, title case where required, and whitespace. Exact executed checks/results are in the documentation PR. No Python runtime or physics behavior changed; no physiological or cross-engine result is claimed. Governed manual sources/generated artifacts were not changed.

Private source data was not downloaded into these public worktrees. Use the authorized private inventory and public synthetic fixtures; never expose original identifiers or source provenance. Dataset availability, EMG/force coverage and independent trials must be verified by implementation owners.

## Resource and Preview Policy

Preview directory: `%USERPROFILE%\Desktop\Motion_Matching_Previews` (created; no video yet). Use privacy-aware per-run manifests. Check free space, bound workers/caches/checkpoints and avoid duplicate raw data. Optional Tailscale jobs require host capacity, exact provider/license/path checks and private access controls. Simscape native acceptance requires MATLAB R2025b.

Only this task's verified merged clean worktrees may be removed after needed receipts/ignored artifacts are preserved. These documentation worktrees are unmerged at packet preparation and remain available for review. Do not remove another agent's worktree.

## Coordination and Known Limits

Issue claims and presence registration succeeded for the public epics. The central mailbox reports incomplete evidence (malformed historical comments/page limit); absence of messages is not proof of exclusive ownership. Existing issue claims remain the coordination authority. No remote fleet jobs were dispatched and no performance numbers were invented.

Next bounded step: F01 capability/gate baseline and F07 muscle/contact capability probe after prerequisites.

## Executed Planning Checks

- Central `scripts/pre_pr.py` with the exact changed-file list from this branch: all five gates passed; no Python or mapped runtime tests changed.
- `git diff --cached --check`: passed.
- Packet validation: nine documents and thirteen implementation-issue payloads passed structure, local-link, engineering-contract and public path/privacy checks.
- `python3 scripts/check_document_title_case.py --staged`: five documents checked, zero violations. The pinned Tools submodule was initialized for the unchanged divergence-inventory check; no provider pin or source was modified.

## F09A Observation Sampling Readiness

The F09a slice adds `NativeMarkerPositionOutput`,
`ObservedMarkerPositions`, and `align_native_positions_to_observations` to the
public `motion_matching` facade. Separate typed records keep actual native
3-D marker positions and observations on their own clocks. The sampler maps
native positions onto
the exact observation clock with an explicit position interpolation version,
matching frame/timebase and marker order, path-free source/output/observation
hashes, and no extrapolation. It retains the native output and observation
clocks separately and delegates scoring to the existing
`compute_replay_five_metrics` implementation.

RED: with the pinned Tools submodule initially absent, collection first failed
at the repository's existing submodule guard; after initializing only the
already pinned local Tools commit, the new test failed to import the missing
F09a API. GREEN: all 12 focused synthetic tests pass. The submodule pointer and
Tools pin was unchanged at the F09a slice. Tests use only generated marker
trajectories; no private capture or acceptance threshold is involved.

The slice does not integrate T01/T02 receipts with an engine, establish replay
reproduction, score collocation feasibility, or qualify any model. Full F09
remains open. The next step from F09a is to consume native replay output from
F06/F07 through this sampling boundary, map observations into F01 comparison
levels, and preserve each unavailable/unqualified row; F09b implements that
scoring boundary.

The method is also documented in the canonical
`manuals/upstreamdrift/chapters/13-feedback-comparison.qmd`. The calculation
registry remains empty because its release is explicitly blocked pending the
owner program; this slice does not register or imply an approved manual
calculation. `python3 -m scripts.check_design_manual_governance` verifies the
unchanged blocked registry/governance envelope.

## F05B Derivative-Aware Native Candidate

The bounded one-hinge MuJoCo RK4 BoxFDDP candidate, exact derivative and
independent frozen-torque replay are described in
`F05B_NATIVE_BOX_FDDP_TURNOVER.md` and the provisional design-manual chapter 24. The source-controlled paired receipt records supported MuJoCo/Crocoddyl
runtime, all predeclared statuses, execution/setup/replay timing and source
hashes. F02 TVLQR remains the broader default; F05 full-body and capture
qualification stay open.

## F05C Native Multi-DOF Tangent Boundary

`F05C_NATIVE_TANGENT_TURNOVER.md` and provisional chapter 25 document an
unactuated floating root plus two directly torqued hinges. Supported native
MuJoCo Euler derivatives are checked in the $n_v$ configuration tangent and
the complete post-limit torque history independently replays through the
pinned Tools T01 contract. The source-hashed receipt retains derivative and
cold replay timing; it does not select a multi-DOF optimizer or promote F05.

## F05D Native Multi-DOF Manifold Candidate

`F05D_NATIVE_MANIFOLD_BOX_FDDP_TURNOVER.md` and provisional chapter 26
describe the contact-free floating-root/two-hinge Crocoddyl state and
native-step action. Supported MuJoCo 3.8/Crocoddyl 3.2.1 paired evidence
retains matched native nonlinear admission, both solver orders, every
fallback and full-state frozen-torque replay. BoxFDDP is a restricted
candidate; F02 TVLQR remains the broader default and F05 remains open.
PR #11944 merged only into the F05c feature branch at
`c1e76ce91db11b257604d9ddbd227268d2e27db1`; F05c PR #11922 remains
open against the F05b feature branch. Issue #11932 remains open and this
stack has not reached `main`. Preserve the commit ancestry and keep later
stacked PRs unarmed until their prerequisites land and their base is `main`.

## F02 Native Floating-Root Manifold Feedback

`F02_NATIVE_MANIFOLD_FEEDBACK_TURNOVER.md` and provisional chapter 27 record
actual MuJoCo 3.8 F02 exact-state TVLQR on the 9/8/2 floating-root fixture.
Native quaternion tangent error, physical inverse mass, direct bounded motor
inputs and fresh complete-state frozen-torque replay pass scoped tests. The
source-hashed receipt measures 0.1261 rad final hip error versus 0.4 rad
frozen nominal over 12 native steps; no capture/contact/full-body claim follows.
A native-replayed moving teacher also supplies per-step tangent Jacobians,
MOSAIC gains and frozen feedforward. From a held-out perturbation, the
frozen and feedback-applied histories independently replay in full and
finish 0.07508 versus 0.02460 rad from the teacher; the teacher/frozen
input identities match and feedback input differs. F03 does not yet export
an optimized native trajectory into this consumer.

## F04 Native Coupling Fixture

`F04_NATIVE_COUPLING_TURNOVER.md`, provisional chapter 21, and the source-bound
`F04_NATIVE_COUPLING_RECEIPT_MJ38.json` document actual MuJoCo 3.8 train-only
feedback-row tuning plus a disjoint held-out trial on the contact-free 9/8/2
floating-root fixture. The held-out joint-error RMSE improves 0.11917 to
0.09647 rad; fresh complete-state replay matches at tested precision. Separate
paired direct-motor interventions show synthetic in-model off-diagonal joint
response, while loss covariance and human causation remain distinct and
unclaimed. The joint stage exhausts its budget, uncertainty is unavailable,
and F04 remains open for capture, contact, physiology and production timing.
This slice is scoped to child #11952 under parent #11788. The child was
checked free and leased after GitHub quota recovery; publication remains
stacked on the F02 native manifold branch and does not close parent F04.

## F03 Native Marker Fitting Boundary

`F03_NATIVE_MARKER_FIT_TURNOVER.md` and provisional chapter 28 document
exact-clock, masked site-position fitting on the same native 9/8/2
fixture. F05c native tangent derivatives are chained through F05d's
next-state BoxFDDP action; its bounded nonlinear admission and Tools T01
fresh full-state torque replay are reused. The supported-provider receipt
retains observation/model/input/source hashes, four accepted commands
versus full-trajectory acceptance, independent marker RMSE and total
wall/CPU work. This is model-generated data without gravity/contact;
real capture geometry/clock, F02 optimized-reference handoff, native
muscles, production full swing and all-engine qualification remain open.
GitHub quota initially paused child claim/publication; after recovery,
child #11948 was created and leased to `codex` for a stacked, unarmed PR.

## F09b Native Observation Scoring

Child issue #11837 adds `src/engines/feedback_observation_qualification.py`
and uses the F01 `FeedbackComparisonRegistry`, F09a position sampler,
canonical replay metrics, pelvis-yaw calculation, and existing acceptance
gates. Each scored case carries the actual Tools T01
`ExperimentReplayBundle`. The adapter checks the complete ordered initial
state and binds its actual `initial_state_sha256` with the model and capability
payload hashes; it does not infer state identity from a schema digest. It also
checks the model/provider, state schema, channel ordering/schema, actual input
history and grid, policy, evidence mode, timebase, and horizon against F01.
An altered full state or input cannot be rebound to stale comparison evidence.
Replay admission requires F01c `feedback-comparison/1.1.0`; observation-
accuracy admission receives the retained F09a observation-grid digest.

The report preserves every required registry cell and all six engine IDs.
Missing evidence remains missing, unavailable stays in the denominator, and
score computation cannot promote an unqualified registry row. The workflow
uses frozen gate IDs/configuration without inventing or relaxing tolerances.
The canonical calculation description is in
`manuals/upstreamdrift/chapters/13-feedback-comparison.qmd`; implementation
limits and commands are in
`docs/development/feedback_controls/F09B-OBSERVATION-QUALIFICATION.md`.

The F09b focused suite has twelve passing synthetic tests, including full
initial-state identity binding and applied-input tamper rejection. These tests
do not provide a native run, private capture acceptance, model qualification,
or physiology evidence. F09 remains open for the native consumer and further
per-engine qualification workflow.

## F09c Strict Native Replay Execution

Child #11898 adds `src/engines/feedback_native_execution.py`. It requires an
explicit binding between an existing F01 inventory row and a Tools T01 native
bundle; it keeps the inventory provider separate from the T01 execution
provider, loaded model, state schema, and channel schema. It refuses unsupported
or unavailable required rows, torque/excitation mismatches, incomplete T01
capabilities, observation/state-feedback access, reset-enabled policy, stale
identities, and inputs outside the current reviewed direct-torque paths.

The executor calls the actual MuJoCo or Drake adapter and verifies returned
initial physical/numerical state, full time grid, applied input, policy,
channel mapping, finite outputs, and complete horizon. Its receipt remains
unqualified and excludes model paths. The report preserves every F01 row and
the six required engines. No marker FK mapping is inferred from generalized
coordinates or native integration states. The receipt keeps the reset policy
(`state_reset_allowed=False`) separate from `state_reset_count=None`; the
adapter does not instrument a reset counter, so the workflow does not invent a
measured zero.

The focused suite passes with independent contract fixtures and a generated
one-hinge MJCF through the real MuJoCo adapter. It proves API wiring and
integrity refusal only, not a production model's physics. No private capture
data, physiological tolerance, or qualification claim was used. The canonical
manual is `manuals/upstreamdrift/chapters/13-feedback-comparison.qmd`; the
current API and constraints are recorded in
`docs/development/feedback_controls/F09C-NATIVE-EXECUTION.md` and `SPEC.md`.
Further acceptance remains in F09/F10. F09d #11907 adds the separately scoped
engine-owned FK bridge to named 3-D marker positions before observation scoring
consumes a replay receipt.

## F09d Native Marker Forward Kinematics

Child #11907 adds `src/engines/feedback_native_markers.py` and uses the public
native MuJoCo/Drake replay adapters to execute a frozen bundle once, then map
each actual output configuration to explicitly attached body-local points.
`NativeMarkerMap` binds engine/model/variant, source and loaded-model digests,
adapter identity, world frame, simulation-relative timebase, ordered labels,
native frame names, and finite offsets in metres. Its content digest and the
native replay receipt/output digests travel with `NativeMarkerPositionOutput`
in an unqualified `NativeMarkerReplayEvidence`. `NativeMarkerReplayReport`
retains each inventory row and all six required engines; FK coverage is not
qualification. Model paths and captured data are omitted.

The independent synthetic MuJoCo fixture uses a nonzero floating base pose,
hinge angle and marker offset; expected positions come from direct quaternion
and hinge-transform equations. The Drake fixture uses a nonzero floating-base
translation, a nonzero local offset and distinct `nq=8`, `nv=7`; expected
positions use the body's world pose applied to the local point. Unknown frame
and stale loaded-model mappings fail closed. These tests prove the FK seam and
identity checks only. OpenSim reuses F07 #11903's separate geometry provider
only after its full-state replay seam is available; MyoSuite and Simscape
remain blocking denominator rows. No observation threshold,
physiology, contact qualification or cross-engine result is asserted.

Focused validation: `python -m pytest -q --confcutdir=tests/unit/engines
tests/unit/engines/test_feedback_native_markers.py` passes on Windows (with the
optional Drake dependency skipped there). The actual Drake floating-base
fixture passes in the owned Ubuntu 24.04 Drake environment with pytest's
repository config disabled because that environment lacks pytest-asyncio;
its only warning is the unregistered `unit` mark in that isolated invocation.
The canonical calculation note is
`manuals/upstreamdrift/chapters/28-native-marker-forward-kinematics.qmd`.

## F09e Pinocchio Native Marker Forward Kinematics

F09e #11914 extends the F09d marker contract with actual Pinocchio replay
output from F06c #11900. It reuses the exact Pinocchio native replay adapter
through merge ancestry and passes its q trajectory to the existing
`PinocchioPhysicsEngine` frame-transform API. The `NativeAdapterBinding`
continues to distinguish F01 package/variant/drive inventory identity from
native model/variant, provider, loaded-model, state-schema, channel and policy
identities. Local marker mappings bind frame names, ordered labels, offsets in
metres, output frame and timebase.

The executor admits Pinocchio's q/v state schema without inventing a native
cache payload. Its `nq=8`, `nv=7` free-flyer fixture preserves distinct
configuration and tangent dimensions. Policy forbids reset; the receipt leaves
the uninstrumented reset count unknown. All model/drive rows and all six engine
IDs remain visible, and every generated marker result stays unqualified.
MyoSuite, OpenSim and Simscape remain blockers without reviewed providers.

The native Pinocchio 4.1 integration fixture passed in the owned Ubuntu 24.04
environment. Its independent analytic expected position includes nonzero
floating-base translation/yaw, hinge angle, joint origin and marker-local
offset. See `docs/development/feedback_controls/F09E-NATIVE-MARKER-FK.md` for
the exact command, environment, observed API incompatibility, and validation
limits. The canonical calculation note is
`manuals/upstreamdrift/chapters/30-pinocchio-native-marker-forward-kinematics.qmd`.

## F09f MyoSuite Native Excitation Replay (#11918)

Branch `feat/11918-f09f-myosuite-native-replay` adds a native MyoSuite 3.0.0 /
MuJoCo 3.6.0 consumer for Tools T01 muscle-excitation bundles. The input is the
actual normalized post-mapping excitation written to native `data.ctrl`; Gym
`[-1, 1]` actions are not called excitation and are not inverse-mapped. The
adapter steps the plant directly using explicit frame skip and fixed-horizon
ZOH input and does not invoke Gym `step`, observation, reward, task termination,
or reset. It binds full physical/numerical initial state plus supported wrapper
flags and counters and verifies exact native control readback. The current
source also rejects global MuJoCo callbacks, observes and verifies each
post-step `data.time`, checks restored full integration state, rejects
nonfinite output, and ensures `mj_forward` does not change the clock.

Wrapper snapshot and restore now share one chain-walk helper. Focused tests
check the explicit supported order and restoration of the captured counters
and flags.

Focused local validation: Ruff passes and 17 tests pass; two optional runtime
tests skip in the local Python 3.13 environment because MyoSuite/Drake are
absent. The actual MyoSuite runtime test passed in the isolated, task-owned
Python 3.12 environment on DeskComputer (`MyoSuite 3.0.0`, `myo-sim 0.2.3`,
Gymnasium 1.2.3, MuJoCo 3.6.0): 1 passed with two overlay-only unknown pytest
mark warnings. The fixture is the public elbow pose demo and does not qualify a
production golf model. F01's required MyoSuite driver and iron rows remain
unqualified, and all six engine rows remain represented. The generic legacy
Gym four-tuple path was not changed because this provider avoids `env.step()`.
The wrapper-state refactor adds one contract test; its focused bundle-contract
and runtime suite passes 7 tests with one optional runtime skip locally.

The callback/clock/full-state hardening is newer than the cited DeskComputer
runtime execution. Synthetic MuJoCo regressions and local affected tests cover
the new checks; native MyoSuite revalidation of this exact source revision is
pending. Keep the prior runtime receipt bound to its executed source and do not
promote it to evidence for the current revision.

See `docs/development/feedback_controls/F09F-MYOSUITE-NATIVE-EXCITATION.md`
for exact validation and limits, and
`manuals/upstreamdrift/chapters/31-myosuite-native-excitation-replay.qmd` for
the canonical boundary note. Keep the pull request stacked/unarmed until its
F09c, T01, F09e marker handling, and runtime-provider dependencies are in the
required ancestry; none of this work closes F09 or F10. The integrated branch
also contains the actual F09c ancestry merge and the F09e q/v-only Pinocchio
state check. Combined replay, observation, capture, and marker tests pass; two
Drake cases skip on local Windows because its optional `pydrake` bindings are
absent. F09c's lazy Tools seam and bounded replay receipt validators preserve
source behavior.

## F09g Direct Model-Path Replay Kernel (#11945)

Branch `feat/11945-myosuite-direct-model-provider` adds a reusable T01
`ACTUATOR_COMMAND` replay kernel in
`src/engines/physics_engines/myosuite/python/native_direct_model_replay.py`.
The source-bound direct factory receives the unchanged XML model path and
declared resource root; it verifies the declared resource set, compiled
loaded-model identity, ordered `actuator:<name>` mappings, MuJoCo 3.8 actuator
law manifest, contact digest, solver/integrator, full initial integration
state, and actual stepped clock. It checks command readback, warnings, callback
absence, plugin/wrapper exclusion, finite output, and direct-policy
restrictions. The final T01 input row remains integrity-bound as the horizon
endpoint; it is not injected as an extra step and need not repeat the previous
command.

TDD found and removed an over-restriction on the terminal command row: the
public driver/iron histories contain a distinct value at the final sample, so
requiring that it repeat the previous value incorrectly rejected an
otherwise well-defined `N-1` interval replay. The opt-in integration test
replays both prepared public variants in the existing MuJoCo 3.8.0 environment.
All 31 saved integration-state samples match exactly for both models, across
the ordered 100-channel direct command bundle. Synthetic regressions cover
resource tampering, actuator law mismatch, warning/wrapper rejection, and
prevent a synthetic class from claiming the MyoSuite SDK. No private or raw
mocap data was used.

This is underlying MuJoCo execution of pinned MyoSuite-sourced XML, not
official MyoSuite `MujocoEnv` execution. The earlier MyoSuite 3.0/MuJoCo 3.6
elbow smoke is not evidence for this source revision or either golfer model.
No official SDK constructor was inspected or invoked here; no production F09
registration or qualification receipt is emitted, and required driver/iron
and six-engine denominator rows stay open. See
`docs/development/feedback_controls/F09G-DIRECT-MODEL-REPLAY.md` and the
canonical boundary section in
`manuals/upstreamdrift/chapters/31-myosuite-native-excitation-replay.qmd`.
The kernel accepts only `engine_id="mujoco"`; all other engine identities are
rejected before factory invocation. Its bounded MuJoCo 3.8.0 disk-MJCF audit
follows native-tested include resolution and MjSpec mesh/texture/cube-face
metadata, requiring every discovered path and hash to appear in the supplied
inventory before checking the compiled MJB identity. It rejects unreviewed
loaders and formats before native parsing; this is not a general MJCF closure
parser. The exact official SDK constructor/runtime remains uninspected, and no
official MyoSuite or capture-model qualification follows from this route.

An additional unqualified native diagnostic now covers the pinned public
MyoSim `arm/myoarm.xml` donor. MuJoCo 3.8 compiles 63 muscle/tendon actuators
and replays a 60 ms native `ACTUATOR_COMMAND` pulse exactly from the complete
integration state. A one-step `139 x 139` state and `139 x 63` input finite-
difference probe is executable. Its apparent derivative blow-up is now
traced to central perturbations crossing `md3_flexion`'s lower joint limit:
the initial coordinate is exactly `0 rad` at the compiled lower bound. An
in-memory diagnostic model with only limit constraints disabled stabilizes
that derivative; disabling equalities or contacts alone does not. Muscle
force, tendon length and estimated moment arms vary smoothly across the
critical pair, while the constraint-force component changes from zero to
about `0.412 N`. These modified copies are diagnostic only and do not qualify
a plant or alter the donor. The source default also has two penetrating
contacts, including a large humerus/thorax contact force; it is not an
admissible equilibrium or capture pose. This is a muscle-only upper-limb
candidate, not a golfer variant or official MyoSuite SDK run. See
`docs/development/feedback_controls/F09G-MYOARM-DIAGNOSTIC.md` and the
source-hashed `F09G-MYOARM-FD-AUDIT-MJ38.json` receipt for details. F05's
current tangent derivative remains a two-motor torque-only fixture; its
Jacobians do not cover muscle activation dynamics.

## F09h Direct Model and Resource Closure (#11950)

Branch `feat/f09h-native-muscle-replay-11950` extends the direct replay seam
with a bounded MuJoCo 3.8.0 source/resource admission. Hardened XML traversal
uses the native-tested model-root-first include selection, while `MjSpec`
provides the reviewed mesh, texture, compiler-directory and cube-face asset
metadata. The preflight requires every discovered source and resource to be
contained under the declared root and hash-matched to the existing manifest;
it verifies the compiled MJB before model construction and rechecks files after
construction. The closure is restricted to the inspected disk XML, STL/MSH,
and PNG path, not a general MJCF parser or VFS/custom-loader guarantee.

Adversarial TDD covers include priority/fallback, undeclared, missing,
changed/outside-root and symlink resources, texture cube faces, mesh formats,
URI references, compiler directories, unsupported source loaders, and refusal
of process-global callbacks before preflight compilation. On the retained
MuJoCo 3.8.0 runtime, driver/iron closures each discover 49 exact-hash files
from the pinned 310-file public inventory, with compiled MJB identity matching
the receipts; driver, iron and MyoArm replay tests pass. The Windows symlink
test skips because this account lacks symlink privilege. The official MyoSuite
SDK/provider and production physiology/contact/capture gates remain open. See
[`F09H-NATIVE-RESOURCE-CLOSURE.md`](F09H-NATIVE-RESOURCE-CLOSURE.md).

## F01f Compiled Actuator Command Admission (#11955)

Branch `feat/f01f-compiled-command-admission-11955` adds a native-bound
compiled command pathway across the existing T02 structural profile row, F09
execution, and F01 comparison admission. T02 artifact bytes are resolved by
reference and digest, then recomputed against the loaded MuJoCo model and T01
bundle. The typed native receipt binds source/loaded model, inventory and
execution providers, runtime, full initial state, ordered actuator laws,
input/policy/time grid, resource closure and output-state digest. F01's
`feedback-comparison/1.2.0` path admits matched within-engine replay and refuses
muscle-only claims for mixed `ACTUATOR_COMMAND` input. Cross-engine comparison
still needs a reviewed semantic mapping.

The integrity receipt is not a signature: serialized external receipts need
independent lineage verification or native replay. Current tests use an
independent synthetic model on MuJoCo 3.8.0. The native engine remains MuJoCo
even when the inventory source row is MyoSuite; official MyoSuite runtime and
production driver/iron binding remain unqualified. All six-engine and
physiological acceptance gates stay open. See
[`F01F-COMPILED-COMMAND-ADMISSION.md`](F01F-COMPILED-COMMAND-ADMISSION.md) and
[`manuals/upstreamdrift/chapters/37-compiled-actuator-command-admission.qmd`](../../../../manuals/upstreamdrift/chapters/37-compiled-actuator-command-admission.qmd).

The exact Tools feature pin adds nine `sidekick.lab` API files to the
shared-tools divergence inventory. The supported generator also corrected
stale prior last-touch entries: for example, the unchanged Tools root
`src/shared/python/__init__.py` has the same blob at the T01 base pin and this
feature pin, and `git log` reports the same historical commit for both, while
the old generated report incorrectly listed the T01 pin commit as its last
touch. The submodule contains complete history and the old/new feature commits
share their expected merge base. The complete generated JSON/Markdown output
is retained; no generated authorship values were edited manually.

## Exact Tools Main Pin Alignment

This parent-branch repair aligns the `vendor/ud-tools` gitlink, `Cargo.toml` tools-core revision, and `requirements-tools.txt` source pin to the actual merged Tools main commit `86d0f28b1cc5acf61185e07e320c816c2d005512` (PR #5475). It changes dependency identity only; it does not add qualification evidence. Validate the three surfaces with `scripts/shared_tools/check_tools_pins.py`.
