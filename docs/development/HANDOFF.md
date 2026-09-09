# Comparison State Qualification Handoff

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: C:/Users/diete/Repositories/Worktrees/UpstreamDrift-reference-registration
- Branch: fix/reference-comparison-qualification
- Baseline commit: 920a0c881
- Implementation commit: SELF
- Pull request: #9884
- Governing issue: #9879; epics #9863 and #9849

## Changes and Evidence

Nine state/path/loading regressions failed on merged main. The fix preserves
unrelated registration/layer fields and blocks control signals during saved
state restoration. Invalid, oversized and mismatched sidecars fail visibly;
canonical UUID and filename-safe view validation prevent path traversal.
The comparison dialog reuses SwingExportActions with a snapshotted job instead
of duplicating worker lifetime handling. Failed saves and duplicate exports
start no worker; Escape/window close cancels and waits asynchronously for the
worker to finish before closing. The worker and progress dialog are disposed.

Thirty-one focused comparison, swing export and coaching tests pass, including
real-thread deferred-close/cancellation and failed-save tests. Three modified
source modules pass mypy with follow-imports=silent. Ruff and architecture checks pass. DRY/LoD report no growth with existing
baselines unchanged. Design-manual governance passes with release still
blocked-inventory-required (2 QMD sources, no registered calculations).
Protected CI remains pending.

## Remaining Product Work

The other agent merged reference registration/workspace and responsive layout
while this goal was paused. Its implementation is retained. Prior unmerged
registration work remains preserved on feat/9865-reference-registration at
8ad3f5be7; its schema is incompatible and must not be merged wholesale.
Epic #9863 was reopened because its release acceptance is not yet met:
#9881 timing/gaps/calibration identity; #9882 renderer/export parity; #9883
spatial/event alignment controls, unsaved-change handling and visual evidence.
The initial sidecars have hashes, not signatures; a loaded camera alone does
not qualify manual spatial registration. No physical calibration or manual
publication release claim is made. Fleet rollout #1579 is independently open.

## Coordination

Session capture-product-01a08427-comparison-qualification owns #9879 in this
isolated worktree. Shared checkouts, vendor child code and peers remain intact.
Read the central mailbox before scope expansion, commit and handoff. Publish
through a topic PR referencing #9879 and normal branch protections.
