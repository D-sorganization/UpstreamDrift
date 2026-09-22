# Club-Only Motion Matching Turnover

## Current State

Governing epic: [#10602](https://github.com/D-sorganization/UpstreamDrift/issues/10602).
Companion epic: [#10603](https://github.com/D-sorganization/UpstreamDrift/issues/10603).
**CO-00 (#10604)** shipped via PR #10667. **CO-01 (#10605)** shipped via PR #10670.
**CO-02 (#10606)** shipped via PR #10675. **CO-03 (#10607)** shipped via PR #10678.
**CO-04 (#10608)** shipped via PR #10680. **CO-05 (#10609)** shipped via PR
#10681. **CO-06 (#10610)** shipped via PR #10687. **CO-07 (#10611)** shipped via
PR #10700. **CO-08 (#10612)** shipped via PR #10703. **CO-09 (#10613)** is in
progress: integrate club-only matching into existing FitSwingProvider / pipeline
/ ledger / ResultsBrowser / Motion Matching GUI surfaces without parallel
frameworks (software-contract UI only; named native G1 blockers; no invented
native verified claims).
Evidence:
[club_workbook_identity.json](evidence/club_workbook_identity.json),
[club_observation_contracts.json](evidence/club_observation_contracts.json),
[club_plausibility_acceptance.json](evidence/club_plausibility_acceptance.json),
[club_starting_guesses.json](evidence/club_starting_guesses.json),
[club_pendulum_match.json](evidence/club_pendulum_match.json),
[club_body_candidates.json](evidence/club_body_candidates.json),
[club_control_replay.json](evidence/club_control_replay.json),
[club_fast_matching.json](evidence/club_fast_matching.json),
[club_matrix_qualification.json](evidence/club_matrix_qualification.json),
[club_ui_integration.json](evidence/club_ui_integration.json).
Existing reference epic
[#10584](https://github.com/D-sorganization/UpstreamDrift/issues/10584) and
native program [#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363)
remain authoritative for their model and physical contracts. Read the companion
[Review](../club_neural_review/REVIEW.md) and [Workbook
Audit](../club_neural_review/excel_audit.json).

## Objective

Create accurate observed club motion with the actual existing models while exposing
alternative plausible body/control solutions and their assumptions. Cover all four
unique workbook trials and the
[#10585](https://github.com/D-sorganization/UpstreamDrift/issues/10585) model roster.

## Issue Index

| Issue                                                                   | Work Package                                                         | Prerequisites       |
| ----------------------------------------------------------------------- | -------------------------------------------------------------------- | ------------------- |
| [#10604](https://github.com/D-sorganization/UpstreamDrift/issues/10604) | CO-00: Freeze Workbook Identity, Units, Events and Trial Lineage     | First Dispatch      |
| [#10605](https://github.com/D-sorganization/UpstreamDrift/issues/10605) | CO-01: Extend Canonical Club Observation Contracts and Calibration   | CO-00               |
| [#10606](https://github.com/D-sorganization/UpstreamDrift/issues/10606) | CO-02: Define Golf Plausibility Priors, Ambiguity and Acceptance     | CO-01               |
| [#10607](https://github.com/D-sorganization/UpstreamDrift/issues/10607) | CO-03: Build Retrieval and Constrained IK Starting Guesses           | CO-01, CO-02        |
| [#10608](https://github.com/D-sorganization/UpstreamDrift/issues/10608) | CO-04: Match Club-Only Motion With Double and Triple Pendulums       | CO-02, CO-03        |
| [#10609](https://github.com/D-sorganization/UpstreamDrift/issues/10609) | CO-05: Generate Plausible Upper-Body and Full-Body Candidates        | CO-02, CO-03        |
| [#10610](https://github.com/D-sorganization/UpstreamDrift/issues/10610) | CO-06: Recover Feasible Controls and Independently Replay Candidates | CO-04, CO-05        |
| [#10611](https://github.com/D-sorganization/UpstreamDrift/issues/10611) | CO-07: Optimize Fast Matching and Expose Candidate Diversity         | CO-03, CO-06        |
| [#10612](https://github.com/D-sorganization/UpstreamDrift/issues/10612) | CO-08: Qualify the Club-Only Matrix and Plausibility Tradeoffs       | CO-06, CO-07        |
| [#10613](https://github.com/D-sorganization/UpstreamDrift/issues/10613) | CO-09: Integrate Club-Only Matching Into Existing UI and Results     | CO-01, CO-02, CO-07 |
| [#10614](https://github.com/D-sorganization/UpstreamDrift/issues/10614) | CO-10: Publish Reproduction Guide and Final Club-Only Turnover       | CO-08, CO-09        |

## Execution Order and Gates

CO-00 -> CO-01 -> CO-02 -> CO-03 -> CO-04/CO-05 -> CO-06 -> CO-07 -> CO-08. CO-09
follows the stable service contract; CO-10 is final acceptance.

The immediate task is
**[#10613](https://github.com/D-sorganization/UpstreamDrift/issues/10613) (CO-09)
only** (CO-08/#10703 shipped). This is a dispatch-ready plan, not
authorization to run every expensive
experiment at once. Lower-cost agents handle bounded schema, adapter, fixture, CLI and
UI work. An experienced reviewer checks model/observation semantics, force
identifiability, contact, physical feasibility and scientific promotion. Dependencies
are real contract gates: require merged implementation plus relevant tests/receipts, not
merely a closed issue. Do not start CO-10 until CO-09 lands.

External prerequisites:
[#10585](https://github.com/D-sorganization/UpstreamDrift/issues/10585) model
identities/Tools ownership;
[#10587](https://github.com/D-sorganization/UpstreamDrift/issues/10587) baseline
contracts;
[#10589](https://github.com/D-sorganization/UpstreamDrift/issues/10589)/#10590/#10591
actual reduced models;
[#10378](https://github.com/D-sorganization/UpstreamDrift/issues/10378) native full-body
coverage; [#10379](https://github.com/D-sorganization/UpstreamDrift/issues/10379)
jobs/results;
[#10430](https://github.com/D-sorganization/UpstreamDrift/issues/10430)/#10440 current
physical fitting and strategy integration. The two new epics share
observation/model/receipt contracts and later proposal integration. Their first audits
can run independently; club-only classical matching does not wait for neural training.

## Copyable First-Worker Prompt

Read https://github.com/D-sorganization/UpstreamDrift/issues/10602 and
https://github.com/D-sorganization/UpstreamDrift/issues/10604. Read this turnover, the
shared review, and current repo/nearest instructions. Check and post a lease for issue
10604, register presence and use an isolated topic worktree. Implement only CO-00; do
not start sibling campaigns. Follow that issue's input/output contract, named reuse
boundaries and test-first acceptance. Use DbC validation that survives python -O, typed
adapter/service boundaries for LoD, and shared implementations for DRY. Return the
focused PR, exact RED/GREEN and native outcomes, artifact hashes, unresolved blockers
and one next executable task.

## Universal Worker and Validation Contract

## Copyable Worker Prompt

Work only on the selected child (this issue) in D-sorganization/UpstreamDrift. Read the
parent epic and its linked turnover document, then current AGENTS.md, CLAUDE.md,
SPEC.md, AGENT_HANDOFF.md and nearest directory instructions. Verify prerequisite PRs
and contracts, not only closed issue status. From Repository_Management run `python3 -m
scripts.check_agent_claim --repo UpstreamDrift --issue <selected-child-number>`; honor
other leases/do-not-automate. Post your lease, register presence and inspect your inbox
before implementation. Use a fresh topic worktree and preserve unrelated work.

Use TDD: write the behavioral/negative tests named here, demonstrate RED, implement
minimally, show GREEN. DbC: reject nonfinite, wrong dimensions, incompatible
units/clocks/models and absent required evidence, including under python -O. LoD: shared
code receives typed contracts; engine SDK calls remain in the owning adapter and UI uses
services. DRY: reuse the listed public facades, targets, datasets, training scheduler,
FitSwingProvider, physical acceptance, ledger and replay tools. Tools-owned changes land
upstream first, then update the consumer pin; never edit vendor copies. Do not implement
a sibling solver or parallel training/result store.

No fake physical success, hidden target-state resets, invented measured body labels,
silent missing-as-zero data, per-frame geometry/registration changes or relaxed
acceptance. Neural proposals and synthetic fixtures are not native evidence. MATLAB
qualification requires R2025b. Freeze experiment seeds, splits, thresholds and compute
limits before fitting/training; save failed attempts and exact hashes. Expert review is
required for mechanics, contact/force allocation, observation semantics and scientific
promotion; routine implementation should continue under reviewed contracts.

Run repo-standard scoped Ruff check/format, mypy, behavioral tests and required
architecture/file/governance checks. Typical test command: `python3 -m pytest
<named-test-paths> -q -n 0 --no-cov --timeout=60`; discover the actual existing path
first. Native qualification uses real model/runtime tests; a skip leaves it unqualified.
Update current HANDOFF/DEVELOPMENT_LOG/SPEC entries as policy requires. Open a focused
ready-for-review PR using gh with `Fixes #<selected-child-number>` and parent reference.
Follow protected checks, no bypasses. Do not close an epic or claim acceleration from
mock-only tests.

Handoff: PR and state, commit/model/data/checkpoint hashes, changed files, exact
commands/outcomes, numeric quality/coverage and runtime when relevant, failure cases,
artifacts, unresolved assumptions and exactly one executable next action. Stop compute
at the saved budget, diagnose the first failing gate, and retain a reproducible rejected
result instead of running indefinite sweeps.

## Scientific Boundaries

Club-only motion constrains observations, not a unique whole-body movement. Plausibility
is explicitly prior-dependent. A neural proposal, IK preview, synthetic fixture or
closed historical issue is not physically validated motion. Keep geometry and source
frame fixed during a fit, preserve actual timestamps, and qualify continuous native
replay from one initial state. Do not fabricate missing torques/reactions, overwrite
masks with zeros, or raise thresholds to pass.

Native full-body qualification remains in
[#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363)/#10378.
Reduced-model and club-only profiles do not satisfy G3. MATLAB R2025b is required.
Data/model/provider/checkpoint/control-basis changes invalidate old qualification until
rechecked. Trained artifacts per model and acceleration claims require separate
evidence; retain negative benchmark results and open unmet model cells.

## Completion Receipt

Every worker reports issue/PR/SHA, changed files, exact tests/runtime, input/model/data/checkpoint hashes as applicable, replay/quality/coverage and latency when changed, storage location and limitations. Update canonical handoff and governing DEVELOPMENT_LOG entry in place. Final program completion requires the issue-specific actual outcomes, not merely this planning document or green software tests.
