# Shadow Tracker Agent Handoff

## Current State

Epic #10122 now has a tested real-model qualification experiment plus frozen
image-only contracts and dispatch packets. No segmentation, fitter, runnable
product demo or scientific qualification is implemented. There is no newly
committed footage, checkpoint or large model asset.

The planning setup [PR #10136](https://github.com/D-sorganization/UpstreamDrift/pull/10136)
is merged and ST-00 is closed. Image-record implementation remains open.

## Immediate Dispatch

This handoff is published in
[PR #10144](https://github.com/D-sorganization/UpstreamDrift/pull/10144), closing
only handoff preparation #10142. Its source code and contract decisions must
be present in the worker checkout.

Use [Ready Tasks](READY_TASKS.md): Packet A (#10137) and Packet C (#10139)
can start from this handoff independently; Packet B (#10138) waits for A.
[Contract Freeze](CONTRACT_FREEZE.md) supplies exact types and acceptance cases.
[Qualification Findings](QUALIFICATION_FINDINGS.md) records the real-runtime
probe and specialist blockers #10140/#10141. ST-01 remains open; its scientific
gates are not passed. See ST-D7 for why image bookkeeping can proceed separately.

Canonical starting point: [README](README.md). Decisions are in
[Architecture](ARCHITECTURE.md); contracts and tolerances are proposals until
ST-01/ST-02 freeze them. GitHub issues are authoritative for current work status.
Do not copy stale `pending` roadmap state into a completion claim.

## First Pickup

1. Read repository `AGENTS.md`, `CLAUDE.md`, `SPEC.md` and root `AGENT_HANDOFF.md`.
2. Read this file, [Epic](EPIC.md), and the selected
   [Work Package](WORK_PACKAGES.md). Inspect its actual issue and dependencies.
3. Recheck source interfaces listed in [Integration](INTEGRATION.md). The initial
   inspection commit is `f8daa71aa263a60c54c785b1ac2d4bc060eb2a71`.
4. Use the fleet claim checker and lease script from Repository_Management
   before starting; do not touch `do-not-automate` tasks. Use an isolated worktree.
5. Select one small public behavior, write its failing test and capture failure.
6. Implement minimally, run green plus existing boundary tests, and review
   contracts, deep object access and duplicate logic before pushing a focused PR.
7. Update the root handoff and this project's evidence/status pointers. Reference
   the child issue. Close a child only when all of its acceptance criteria land.

ST-01 remains the research task. Lower-cost agents take frozen image-only
ST-02A/C/B packets, not the entire ST-02 or ST-08 work package. The full-body
native/canonical mapping and scientific profile are not frozen by this handoff.

## Original Planning Validation

- Documentation title-case check: 17 changed Markdown documents, zero violations.
- Local-link and roadmap audit: 25 local links resolved; 13 unique child issues;
  dependency graph acyclic and ordered; cited source paths exist.
- Docs governance and ADR numbering checks passed; 50 ADRs, no conflicts.
- Existing title-case and SPEC-integrity tests: 9 passed. Repository conftest
  emitted unrelated optional-dependency collection warnings.
- Tracked Python file-size audit passed. No Python behavior changed; scoped Ruff
  found no Python files, so it is not runtime validation.
- Design-manual governance verified its existing state, with release still
  `blocked-inventory-required`; no scientific approval is claimed.
- The documented `shared_scripts/fleet_hooks.py spec-changelog` entry point is
  absent in this checkout. Existing SPEC-integrity tests provide the available
  local check; preserve this tooling limitation in the PR.

CI results belong to the setup PR. This baseline did not exercise a tracker.
Current diagnostic TDD, live-model test and evidence details are recorded in
[Qualification Findings](QUALIFICATION_FINDINGS.md).

## Small-Agent Task Prompt

Copy and fill the following; attach only the relevant source/test excerpts.

```text
Implement one slice of Shadow Tracker ST-<ID>, GitHub #<child>, under epic #10122.
Read repository rules and docs/plans/shadow_tracker/AGENT_HANDOFF.md first.
Behavior: <one externally observable outcome>.
Allowed files: <explicit source and test paths>.
Reuse: <existing public interfaces and their current signatures>.
Contract: <units, frame, dimensions, unknowns, errors, ownership, invariants>.
First failing test: <test name, input and expected result>.
Acceptance: <exact cases and versioned gate profile if applicable>.
Out of scope: <adjacent work packages and protected interfaces>.
Validation: <focused commands and existing consumer tests>.
Deliver: red/green evidence, focused diff, limitations and handoff update.
Stop and report concrete evidence if a required interface or model capability
is absent. Never substitute mock success, invent measurements or loosen gates.
```

## Escalation Boundaries

Small agents can implement frozen record validation, serialization, deterministic
loss fixtures, provenance/correction storage, UI presentation and status rules.
They must bring schema changes, engine conventions, contact/actuation assumptions,
new control bases, uncertainty claims, benchmark thresholds and new dependencies
to the package reviewer. Continue independent authorized work when possible;
do not silently decide a scientific question by making tests pass.

Keep work bounded by one behavior and ideally one or two production modules.
Use fixed tiny fixtures for iteration; reserve GPU/full-engine/long-swing jobs
for their integration gate. Record evaluation budgets before launching them.
One implementation attempt plus one focused correction is a useful small-agent
handoff boundary; repeated conceptual failures go to a reviewer, not a longer prompt.

## Required PR Evidence

- Child issue, exact scope, dependency state and changed public behavior.
- Red test command/failure, green commands/results and affected consumer checks.
- DbC pre/postconditions and invalid-input tests; LoD/DRY reuse rationale.
- Source/model/data/config hashes for any numerical evidence.
- Distinct software-test and scientific-gate results; skipped engines identified.
- Any changed decision, migration, capability registry or manual boundary.
- Remaining work, exact next action and reproducible failure if blocked.

Never use `Closes #10122` until the complete epic meets its definition of done.
Use `Refs #10122` on implementation PRs and close only completed child issues.
No direct pushes to main, no rewriting other agents' work, and no edits to
fleet-managed instruction blocks.

## Turnover Template

```text
Task / Child Issue / PR:
Branch and Commit:
Completed Behavior:
Files and Public Interfaces Changed:
Red Test Evidence:
Green Validation and Known Skips:
Scientific Profile / Result / Receipt Path:
Input, Model, Engine and Configuration Hashes:
Decisions Changed and Reviewer:
Remaining Risks or Blockers:
Next Small Action and Exact Command:
Claims Explicitly Not Established:
```

Use factual present-tense state. Preserve historical receipts under versioned
evidence paths; do not overwrite old evidence with a new model's results.
