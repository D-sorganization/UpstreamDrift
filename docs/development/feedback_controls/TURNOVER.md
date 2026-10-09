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
The branch also contains the actual F09c parent merge. Combined replay,
observation, capture, and marker tests pass; two Drake cases skip on local
Windows because its optional `pydrake` bindings are absent. F09c's lazy Tools
seam and bounded replay receipt validators preserve source behavior.
