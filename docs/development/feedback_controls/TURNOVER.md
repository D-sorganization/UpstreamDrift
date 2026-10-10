# Feedback Controls Planning Turnover

## F07 Source-Bound Native Moco Numerical Guess Handoff

Issue #11990 adds a byte-bound numerical seed for the maintained native Moco
runner. Actual unchanged 520-muscle driver/iron preparation now has six rather
than seven blockers; only the absent guess-file blocker was removed. This is
not measured or physiologically accepted state. See
`F07_NATIVE_MOCO_GUESS_TURNOVER.md` and canonical chapter 39 for the native
readback, exact clocks, remaining policies and private receipt location.

## F07 Native Offline Muscle Matching Handoff

Issue #11968 adds a maintained OpenSim Moco prepare/solve/T01 export/fresh
replay/physical scoring route. The executable design and exact reproduction
commands are in `F07_NATIVE_MOCO_RUNNER_TURNOVER.md` and canonical chapter 39.
Synthetic two-muscle OpenSim runs pass, while unchanged 520-muscle source plus
frozen private driver/iron reference preparation reports seven blocker kinds
and performs no solve. The earlier planning-only status below remains the
history of the initial packet, not the status of this implementation slice.

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
Tools pin are unchanged. Tests use only generated marker trajectories; no
private capture or acceptance threshold is involved.

The slice does not integrate T01/T02 receipts with an engine, establish replay
reproduction, score collocation feasibility, or qualify any model. Full F09
remains open. Next: consume native replay output from F06/F07 through this
sampling boundary, map the resulting observation receipt into the T02/F01
comparison levels, and preserve every required unavailable/unqualified row.

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

## F07f Native Reference-Conventions Handoff

The scoped read-only OpenSim source audit is tracked under #11962; exact native
TDD, bounded public-source diagnostics, correction of post-initSystem assembly
semantics, reproduction and remaining donor/resource/anatomy gates are in
`F07_REFERENCE_CONVENTIONS_TURNOVER.md`. Canonical calculation detail is in
chapter 38. Neither the previous 520-muscle source nor Pose2Sim is qualified
for capture-matched muscle-driven forward dynamics by this diagnostic.
