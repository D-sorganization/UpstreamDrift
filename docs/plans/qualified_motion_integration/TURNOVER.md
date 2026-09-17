# Crocoddyl and Pink Implementation Turnover

Execution companion to [epic #10254](https://github.com/D-sorganization/UpstreamDrift/issues/10254)
and the [numerical contracts](README.md). Updated 2026-09-16. This is a dispatch
plan, not evidence that the remaining product or physical gates pass.

## Authority and Dispatch Order

Review `origin/main` at dispatch; the reviewed revision is
`0ec64e45f1f259c11e26e36315ee370a5c51c7d9`. Pipeline consolidation #10251
and anthropometry refresh #10250 are closed. Their old pre-merge instructions
are superseded. Receipt-chain integrity remains open in #10271; do not
backfill hashes into historical evidence or call it regenerated evidence.

Before coding, check and acquire the issue lease from Repository_Management.
Use one topic worktree per issue, and record the exact base SHA and dependency
PR heads. A different worker's open claim is a stop condition for that slice.
Do not cherry-pick entire stacks independently into multiple downstream PRs.
The integration owner resolves dependency order and shared-file conflicts.

| Slice                   | Existing Issue / PR    | Dispatch State                               | Required Boundary                                  |
| ----------------------- | ---------------------- | -------------------------------------------- | -------------------------------------------------- |
| Contact Derivatives     | #10255 / #10259        | Implemented; Review and CI Pending           | Real constrained-plant derivatives                 |
| Finite Weld Derivatives | #10260 / #10263        | Implemented; Review and CI Pending           | Six-dimensional finite-pose derivative             |
| Viewer Lifecycle        | #10256 / #10266        | Adapter Implemented; Gepetto Runtime Corrupt | Persistent owned scene and cleanup                 |
| Pink Adapter            | #10257 / #10267        | Implemented; CI Triage Pending               | Configuration, constraints, limits, geometry       |
| Runtime                 | #10262 / #10268        | Native Probes Pass; CI Pending               | Isolated ABI and actual QP execution               |
| Polynomial Step         | #10265 / #10270        | Implemented; CI Pending                      | Exact RK4 state/coefficient sensitivities          |
| Crocoddyl Actions       | #10269                 | Local Checkpoint; Integration Review Pending | Lift, flow, terminal and replay diagnostics        |
| Pink Pipeline           | #10276, #10277, #10278 | Contract Review Before Dispatch              | See the Pink packet; adapter alone is insufficient |
| Evidence Integrity      | #10271                 | Bounded Work After Producer Review           | Input-to-output provenance chain                   |
| Plant Parity            | #10112                 | Integration Owner Design Required            | Matched model, ground, controls and weld           |
| Objective and Benchmark | Epic W5                | Blocked on Actions and Objective Review      | Same full horizon and independent replay           |
| Product Wiring          | Epic W8                | Blocked on Stable Backend Results            | One service for CLI and UI                         |

Do not create duplicate issues for these scopes. For an epic-only row, first
search existing children; reuse one or create one scoped implementation issue,
then acquire its lease. Dispatch readiness is separate from merge readiness.

## Common Worker Contract

Each assignment must include the issue, exact base/dependency SHAs, owned files,
public input/output signature, prerequisites, exclusion list and acceptance
commands. A worker may implement and test the assigned boundary. Changes to
plant, objective denominator, frame convention, physical thresholds, control
family, default backend or time semantics return to the integration owner.

1. **TDD:** Add a minimal regression and record its expected failure before
   implementation. Cover one normal case, one boundary and one invalid case
   for every public operation. Keep independent finite differences independent
   of the derivative under test. Never count skipped native tests as passes.
2. **DbC:** Validate dimensions, names, units, finite values and time grids at
   entry. Document mutable ownership and cache lifetime. Assert meaningful
   postconditions: finite results, unchanged caller inputs, root effort zero,
   constraints audited after integration and structured failure on infeasibility.
   Do not use assertions as the only runtime validation of user inputs.
3. **LoD:** Consume a typed public facade. No engine-private model traversal or
   multi-object attribute chains in orchestration; let the owning adapter
   expose the operation. Run the repository LoD check without inflating its
   baseline to hide new violations.
4. **DRY:** Search shared modules and read their public exports first. Reuse
   canonical coordinate mapping, contact law, capture contract, polynomial
   actuation, RK4 step, runtime probe and receipt producer. UI, CLI and viewers
   must not contain independent solver or calibration implementations.
5. **Return Evidence:** Supply source diff, red/green commands and outcomes,
   native interpreter/version, JUnit counts including skips, seed, input hashes,
   wall time, memory, remaining failures and exact commit SHA. No source edits
   in shared handoff/SPEC/log files until coordinated with the integration owner.

Run focused tests during development. Before publication run required Ruff
lint/format, typing, file/module-size, LoD and design-manual governance checks,
then repository-required CI. A known main failure is a documented blocker,
not permission to remove a gate or declare the branch green.

The main-derived reference-report LoD violation was reproduced and corrected
without changing the baseline: resolve the offset mapping before serialization.
Seven reference-stage tests and the 3,221-file no-growth scan pass locally.
Remote CI remains authoritative for merge readiness.

## Runtime and Resource Procedure

On OGLAPTOP use the existing Ubuntu-24.04 environments. Do not reinstall or
mix pip Pinocchio into the conda Crocoddyl environment. The preserved runtimes
are local conveniences; reproducible repository pins remain authoritative.

| Runtime                                    | Local Location                                            | Verified Use                                     |
| ------------------------------------------ | --------------------------------------------------------- | ------------------------------------------------ |
| Pinocchio 3.8 / Pink 4.4                   | `/home/dieterolson/.venvs/upstream-motion-10254`          | IK and native derivative tests                   |
| Crocoddyl 3.2.1 / Pinocchio 4.1 / Pink 4.4 | `/home/dieterolson/.venvs/upstream-crocoddyl-conda-10254` | Crocoddyl and QP probes                          |
| Gepetto / Pinocchio 3.8                    | `/home/dieterolson/.venvs/upstream-gepetto-10254`         | Corrupt Prefix; Repair Before Live Qualification |

After #10268 is present in the selected checkout, run from that checkout in WSL:

```bash
/home/dieterolson/.local/bin/micromamba run \
  -p /home/dieterolson/.venvs/upstream-crocoddyl-conda-10254 \
  python3 scripts/ci/check_motion_runtime.py --receipt /tmp/motion-runtime.json
```

After storage recovery this command passed all three required subprocess
probes: Crocoddyl ABI/dynamics derivatives, Pink hard equality and Pink
infeasible QP. Checker checkout was `e171759e47cd8947add859fe79e6224439cacf34`.
The Windows worktree pointer prevents WSL Git freshness inspection; its receipt
correctly reports freshness **unknown**. Record Windows Git status/SHA alongside
the receipt instead of claiming the source was verified clean by the checker.

Run native integration tests in a separate process from `tests/unit`: the unit
test configuration mocks optional engines. Save JUnit XML, require a positive
executed count and zero skips for the assigned qualification. Do not use
`-n auto` for native benchmarks while local CI runners are active. Limit large
native work to one job at a time; coordinate with the other worker before
full-horizon allocation. OGLAPTOP policy is four daytime runners, up to six
overnight. Do not stop jobs, WSL or other workers for this project.

## Gepetto Assessment Result

The bounded live spike found truncated files left by the earlier disk-full
installation: the existing Gepetto prefix has zero-byte GUI/Qt/viewer libraries
and a Pinocchio extension that fails with `file too short`. No live rendering
claim is possible. Owned Weston/Xwayland probe processes were cleaned up; no
shared display or runner processes were stopped. The strengthened heavy
integration test checks moved geometry, foreign-node preservation and repeated
cleanup in a fresh native process. Windows skip results do not qualify it.

Decision: keep Gepetto optional and experimental. Before a later live retry,
repair the existing prefix from reproducible package metadata after checking
disk capacity; do not copy or trust truncated cache artifacts. Its repair is
not a prerequisite for Pink/Crocoddyl work. Live pass plus a demonstrated
review workflow is required before adoption; otherwise record deferral and
retain native/MeshCat review.

## Crocoddyl Action Packet (#10269)

Local implementation checkpoint: `7caf4bc6f` on
`feat/10269-crocoddyl-polynomial-actions`, in
`C:/Users/diete/Repositories/Worktrees/UpstreamDrift-10269-crocoddyl-actions`.
It is committed but not pushed. Its base remains `722540e707`; the next worker
must integrate current #10270 (at least `f489853f5`) before publication.
Nineteen unit tests and six fresh-process native cases pass, including FDDP,
BoxFDDP and active/offground full-body derivatives with reordered coordinates.
The harness enforces six native executions with zero failures/errors/skips.
Parent review added initial-boundary defects and shooting-node/replay cost gaps;
regressions fail before those additions and pass afterward. Focused typing,
Ruff, pre-commit and function/file budgets pass. Broad regression, governance,
remote CI and full-horizon performance/physical qualification remain pending.

Resume from the recorded commit; do not reimplement these actions. Preserve
untracked `vendor/ud-tools.partial-10269-diskfull/` recovery material. The next
bounded assignment is integration/review and publication of this checkpoint,
not another action-model implementation.

Owned implementation: new `motion_matching/crocoddyl_polynomial.py`, optionally
a focused diagnostics module, and corresponding unit/integration tests.
Consume `NativeFullBodyStep`; do not alter the plant or stepper to accommodate
the optimizer. Issue #10269 freezes the full API and derivative equations.

The zero-time lift sets global coefficients once; subsequent flow controls have
zero dimension. Flow state is `[q, v, p]` and copies `p` exactly. Lift cost is
zero. Running costs are rates multiplied once by physical `dt`; terminal cost
is unscaled. Quadratic gradients/Hessians use augmented state order. Coefficient
bounds apply to the lift. Do not add hidden regularization to repair a singular
lift Hessian; expose solver regularization and raw conditioning diagnostics.

RED cases: wrong lift derivatives, missing RK4 coefficient block, stale reused
data, terminal `u=None`, irregular time-grid scaling, malformed bounds, contact
kinks and an FDDP trajectory whose nodes differ from independent replay.
GREEN requires pure analytic toy checks plus real native action finite
differences, a small solve and immutable diagnostic outputs. Warm-start
feasibility must be measured before passing `is_feasible=True`.

Independent replay always starts at the original physical initial state and
uses `us[0]` as the one global coefficient vector. Never average node copies,
reset states at shooting nodes or substitute feedback. The 327-dimensional
augmented-state design has material dense memory cost: the preliminary 654-node
allocation estimate is about 2.3 GB before custom buffers. Report allocation
and wall time before a full solve; no silent time coarsening.

## Evidence Packet (#10271)

Own the existing receipt producer/validator and focused freshness tests after
checking the consolidated pipeline. Define raw-byte and semantic hash fields
explicitly with a schema version. The canonical validator owns normalization;
consumers call it instead of reimplementing digest rules.

RED: mutate capture, anatomy, marker map, calibrated/scaled model, solver config,
time grid or replay output and verify chain rejection. Missing fields and an
old schema cannot produce qualified status. Distinguish a whitespace-only
serialization change from a physical model change according to the declared
hash type. Preserve historical artifacts unchanged. GREEN: regenerate both
clubs through the actual producer and validate the entire chain, then test
the UI/CLI consumer rejects stale evidence. Regeneration is not fit acceptance.

## Remaining Expert Decisions and Downstream Packets

- **Pink Timing and Task Assembly:** Review the dedicated Pink packet before
  dispatch. Lock projection-time versus capture-time semantics and exact
  six-dimensional closure/marker Jacobians. A passing low-level QP adapter
  does not establish selectable constrained full-capture IK.
- **Matched Plant (#10112):** Freeze identical canonical anatomy, ground plane,
  bilateral weld, actuation names and contact law across engines. Current
  legacy parity compares different plants. Explicit ground must survive replay;
  near-zero ground cannot silently trigger recalibration. Add that regression
  before backend comparisons. Preserve the existing parity threshold.
- **Objective (W5):** Freeze units, mask denominator, full-capture times,
  weights, limits and cost derivatives in a reviewed specification. Only then
  dispatch residual implementations. Mark soft penalties separately from hard
  constraints; retain failures and compare identical compute budgets.
- **Product (W8):** After result signatures stabilize, dispatch backend
  selection/progress/cancel/error/export through the existing pipeline service.
  Test that selecting Pink/Crocoddyl reaches the real backend, unavailable
  dependencies fail clearly, cancellation leaves no process/scene leaks and
  CLI/UI receipts identify the actual backend. Update feature parity and agent
  context through their canonical sources; no second orchestrator.
- **Diagnostics (W6):** Specify constrained force-balance assumptions before
  adding inverse-dynamics/reaction views. Require forward/inverse consistency,
  frame/power checks and sensitivity. Feedback is a separately labeled replay
  mode; physiological forces are not inferred from RNEA alone.

## Qualification and Capability Decisions

Pink tasks, posture/damping, hard equalities and verified configuration/velocity
limits are in scope. Acceleration limits, collision barriers and CoM tasks
require a measured use case and validated geometry/time semantics. Hierarchical
QP is deferred unless weighted tasks fail a documented priority requirement.

Crocoddyl action/residual models, FDDP and bounded-control assessment are in
scope. Feedback and sensitivity follow offline qualification. MPC, code
generation and parallel optimization are deferred until profiling demonstrates
benefit. Gepetto is an optional live capability decision, never a dependency
of core fitting. Record adopt/defer/reject with evidence and revisit criteria.

Final acceptance requires both complete captures, named modeled/excluded labels,
unchanged masks, independent open-loop replay, driver <60 mm and 7-iron <95 mm
under the declared metric, plus ROM, contact, closure, root-actuation, effort,
cross-engine and convergence gates. User video review remains pending.
MATLAB comparisons use R2025b and depend on #10110. A backend may be usable
while physical qualification remains blocked; label these states separately.
