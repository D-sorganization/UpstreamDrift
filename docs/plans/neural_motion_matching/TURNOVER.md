# Neural Motion Matching Turnover

## Current State

Governing epic: [#10603](https://github.com/D-sorganization/UpstreamDrift/issues/10603).
Companion epic: [#10602](https://github.com/D-sorganization/UpstreamDrift/issues/10602).
Planning and source/workbook review only; no new neural training or fitting campaign was
executed in this task. Existing reference epic
[#10584](https://github.com/D-sorganization/UpstreamDrift/issues/10584) and native
program [#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363) remain
authoritative for their model and physical contracts. Read the companion
[Review](../club_neural_review/REVIEW.md) and [Workbook
Audit](../club_neural_review/excel_audit.json).

## Objective

Generate trustworthy native simulation data and train a versioned neural artifact for
each model to accelerate matching of full or club-only observations, with actual
physical correction/replay and a quality-matched comparison against current methods.

## Issue Index

| Issue                                                                   | Work Package                                                               | Prerequisites       |
| ----------------------------------------------------------------------- | -------------------------------------------------------------------------- | ------------------- |
| [#10615](https://github.com/D-sorganization/UpstreamDrift/issues/10615) | NM-00: Audit Existing Datasets, Checkpoints and Training Claims            | First Dispatch      |
| [#10616](https://github.com/D-sorganization/UpstreamDrift/issues/10616) | NM-01: Freeze Learning Tasks, Model Roster and Benefit Experiment          | NM-00               |
| [#10617](https://github.com/D-sorganization/UpstreamDrift/issues/10617) | NM-02: Make Native Dataset Labels Complete and Semantically Correct        | NM-00, NM-01        |
| [#10618](https://github.com/D-sorganization/UpstreamDrift/issues/10618) | NM-03: Version Efficient Episode Storage, Splits and Dataset Views         | NM-01, NM-02        |
| [#10619](https://github.com/D-sorganization/UpstreamDrift/issues/10619) | NM-04: Generate Feasible Teacher Episodes and Active-Learning Candidates   | NM-02, NM-03        |
| [#10620](https://github.com/D-sorganization/UpstreamDrift/issues/10620) | NM-05: Train Classical and Small Neural Dynamics Baselines                 | NM-03, NM-04        |
| [#10621](https://github.com/D-sorganization/UpstreamDrift/issues/10621) | NM-06: Train Masked Trajectory-to-Control Proposals With Native Refinement | NM-03, NM-04, NM-05 |
| [#10622](https://github.com/D-sorganization/UpstreamDrift/issues/10622) | NM-07: Compare Forward Surrogates and Physics-Structured Alternatives      | NM-05, NM-06        |
| [#10623](https://github.com/D-sorganization/UpstreamDrift/issues/10623) | NM-08: Add Native Verification, Distribution Checks and Safe Fallback      | NM-06, NM-07        |
| [#10624](https://github.com/D-sorganization/UpstreamDrift/issues/10624) | NM-09: Train and Qualify a Checkpoint for Every Physical Model             | NM-04, NM-08        |
| [#10625](https://github.com/D-sorganization/UpstreamDrift/issues/10625) | NM-10: Benchmark Accepted-Match Speed, Data Efficiency and Break-Even      | NM-08, NM-09        |
| [#10626](https://github.com/D-sorganization/UpstreamDrift/issues/10626) | NM-11: Integrate Model-Specific Training and Inference With Existing Tools | NM-03, NM-08        |
| [#10627](https://github.com/D-sorganization/UpstreamDrift/issues/10627) | NM-12: Publish Model Cards, Reproduction Commands and Final Turnover       | NM-09, NM-10, NM-11 |

## Execution Order and Gates

NM-00 -> NM-01 -> NM-02 -> NM-03 -> NM-04 -> NM-05 -> NM-06 -> NM-07 -> NM-08 -> NM-09
-> NM-10. NM-11 follows stable service contracts; NM-12 is final acceptance.

NM-00 ([#10615](https://github.com/D-sorganization/UpstreamDrift/issues/10615)),
NM-01 ([#10616](https://github.com/D-sorganization/UpstreamDrift/issues/10616)),
NM-02 ([#10617](https://github.com/D-sorganization/UpstreamDrift/issues/10617) /
[#10679](https://github.com/D-sorganization/UpstreamDrift/pull/10679)),
NM-03 ([#10618](https://github.com/D-sorganization/UpstreamDrift/issues/10618) /
[#10686](https://github.com/D-sorganization/UpstreamDrift/pull/10686)),
NM-04 ([#10619](https://github.com/D-sorganization/UpstreamDrift/issues/10619) /
[#10698](https://github.com/D-sorganization/UpstreamDrift/pull/10698)),
NM-05 ([#10620](https://github.com/D-sorganization/UpstreamDrift/issues/10620) /
[#10701](https://github.com/D-sorganization/UpstreamDrift/pull/10701)),
NM-06 ([#10621](https://github.com/D-sorganization/UpstreamDrift/issues/10621) /
[#10709](https://github.com/D-sorganization/UpstreamDrift/pull/10709)),
NM-07 ([#10622](https://github.com/D-sorganization/UpstreamDrift/issues/10622) /
[#10768](https://github.com/D-sorganization/UpstreamDrift/pull/10768)),
NM-08 ([#10623](https://github.com/D-sorganization/UpstreamDrift/issues/10623) /
[#10770](https://github.com/D-sorganization/UpstreamDrift/pull/10770)),
NM-09 ([#10624](https://github.com/D-sorganization/UpstreamDrift/issues/10624) /
[#10777](https://github.com/D-sorganization/UpstreamDrift/pull/10777)),
NM-10 ([#10625](https://github.com/D-sorganization/UpstreamDrift/issues/10625) /
[#10778](https://github.com/D-sorganization/UpstreamDrift/pull/10778)), and
NM-11 ([#10626](https://github.com/D-sorganization/UpstreamDrift/issues/10626)) are implemented.
The immediate task is
**[#10627](https://github.com/D-sorganization/UpstreamDrift/issues/10627) (NM-12)**
(publish model cards, reproduction commands and final turnover).

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

Read https://github.com/D-sorganization/UpstreamDrift/issues/10603 and
https://github.com/D-sorganization/UpstreamDrift/issues/10615. Read this turnover, the
shared review, and current repo/nearest instructions. Check and post a lease for issue
10615, register presence and use an isolated topic worktree. Implement only NM-00; do
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
