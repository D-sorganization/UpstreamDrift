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
Tools pin were unchanged at the F09a slice. F09b below advances the pin to
merged T01 so observation scoring consumes the versioned replay bundle. Tests
use only generated marker trajectories; no private capture or acceptance
threshold is involved.

The slice does not integrate T01/T02 receipts with an engine, establish replay
reproduction, score collocation feasibility, or qualify any model. Full F09
remained open at that point. The next step from F09a was to consume native
replay output, map observations into F01 comparison levels, and preserve each
required unavailable/unqualified row; F09b implements that scoring boundary.

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
