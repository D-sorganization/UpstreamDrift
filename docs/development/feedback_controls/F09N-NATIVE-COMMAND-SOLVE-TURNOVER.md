# F09n Native Command Solve Turnover

Issue #12036 implements the next runnable native controller boundary from
[the execution plan](F09N-NATIVE-COMMAND-SOLVE-PLAN.md). Parent F09m is PR #12033
at `ab0a1a37d2846f5da33ffea23b7a4f62c2111c78`. Shared F02 callback kernel and
tests are imported exactly from PR #12024 at
`4f9d2e4b9f741459a2e8ae5950406f9b1129be14`; this is an explicit dependency,
not a second optimizer. No unrelated Euclidean adapter was copied.

## Implementation and Rules

`NativeCommandSolveProblem` freezes initial/fallback command plans, lower/upper
bounds, per-sample increments and criteria provenance. Its parameter digest and
callback identity checks reject changed solve declarations. The declared
criteria digest does not authenticate hidden callback dependencies or establish
physiology. `solve_native_command_plan` uses persistent native predictions and
the frozen tracking objective, then existing SDK promotion and serialized T01
independent replay. It never applies a live command or introduces a new replay
format, native stepping loop, optimizer loop or constant-torque surrogate.

The full fallback is guarded and independently replayed before optimization.
Every knot includes post-mapping bounds and increments, with the first increment
measured from actual observed controls. Complete-state signed margins and a
strict Boolean guarded admission are mandatory. The same margins are recomputed
on guarded SDK states. Numerical tolerance cannot relax exact guarded command
or increment bounds. A selected candidate must improve the recomputed guarded
fallback objective. Cancellation, timeout and numerical/guarded rejection keep
the admitted fallback; source/state/identity execution failures propagate.

Elapsed time covers this function's fallback, numerical search and promotion;
global time-to-accepted-match also includes caller preparation, provider/model
construction, reference mapping, failed work and export. Budgets are cooperative,
not hard deadlines. A future prefix requires fresh current-state admission.
This wave has no private fit, online execution, contact or physiological claim.

## Actual TDD and Validation Evidence

Durable receipts live in the workspace `docs/development/feedback_controls_planning`.
The actual provider is MyoSuite 3.0.0 / MuJoCo 3.6.0 on DeskComputer, reusing the
existing owned source stage and SDK environment.

- `f09n-native-api-red.xml`: seven genuine missing-API failures before implementation.
- `f09n-native-first-executed.xml`: seventeen passes in 74.60 seconds, including
  a real native optimization, guarded recomputation/replay and shared kernel tests.
  A preceding launch failed before tests because the remote test directory was
  absent; the separate zero-test receipt is not behavior evidence.
- `f09n-native-adversarial-first.xml`: twenty-two passes in 88.61 seconds.
  Includes late activation, stale anchor, expired budget and guarded objective
  disagreement tests. Later binding changes have separate evidence.
- `f09n-criteria-provenance-red.xml`: genuine missing declaration API failure.
- `f09n-native-full-campaign.xml`: 169 passes, zero failures/errors/skips in
  227.09 seconds before the final criteria-mutation fix. Twenty source/test
  hashes are retained before and after execution; this historical campaign
  does not certify later edits.
- `f09n-criteria-mutation-red.xml`: genuine failure to reject a criteria
  declaration changed during the solve. Parameter/callback verification fixes
  this behavior.
- `f09n-criteria-binding-green.xml`: all fourteen native controller tests pass
  in 92.94 seconds after frozen parameter/callback verification.
- `f09n-native-final-campaign.xml`: all 170 tests pass with zero failures,
  errors or skips in 250.31 seconds. All twenty before/after executed source/test
  hashes match the current worktree (`f09n-native-final-source-hashes-*.json`).
  The three-step synthetic objective improves from 16.361447408944105 to
  0.0002999297924770022 in 189 numerical evaluations. Measured whole solve time
  is 44.92984030000298 seconds. This is correctness evidence, not a rapid or
  real-time matching claim; profile native/source guards, duplicate evaluations
  and derivative strategy before production matching. JUnit retains metrics and
  parameter/criteria digests despite its nonfatal xunit2 property warning.

The native fixture has a motor and filtered activation actuator, nonzero epoch,
state and previous controls. Its known feasible three-step target is synthetic;
constant finite fixture margins do not qualify production contact or physiology.
Original driver/iron tests remain in the larger affected SDK campaign.

Local Ruff, direct architecture budget, LoD no-growth, DRY, document titles and
file-size checks pass. The exact pinned mypy pre-push hook passes current source.
A cold exploratory global type check was stopped; no pass is claimed for it.
The existing OpenSim environment lacks mypy: its central run passed lint/tests/
import policy but failed missing-type-provider and stale divergence inventory.
The inventory was regenerated through its canonical generator. The subsequent
global central runner passes all five gates, including both source files in
diff-scoped mypy, ten portable kernel tests and fourteen explicit unavailable-
SDK skips. Those skips remain distinct from actual native execution. No new
environment was installed and no hook was bypassed.

Parent PR #12033's inspected CI unit job `114062282519` passed 22,735 tests and
failed two inherited inventory gates: Tools capability acceptance was declared
against `2e766511...` while this stack pins `e775bce8...`, and the monolith
register was stale. The monolith register is regenerated canonically here.
Actual capability probes and served-bundle verification must precede a new
Tools acceptance declaration; no metadata stamp or blanket CI-green claim is
made. Reconcile the admitted private consumer/pin stack before integration.

## Reproduction Commands

Use the actual pinned MyoSuite 3.0.0/MuJoCo 3.6.0 environment, not the portable
environment that skips SDK tests. Point `FEEDBACK_NATIVE_PYTHON` to its existing
interpreter and `FEEDBACK_MYOSUITE_PRODUCTION_ROOT` to the verified unchanged
production model/resources directory. Run from this checkout with the committed
Tools gitlink initialized at `e775bce870690bba1b56f3d6297513003fb7dbac`.
Keep later provider/pin integration evidence separate from this campaign.

```powershell
$taskRoot = (Get-Location).Path
$env:PYTHONPATH = "$taskRoot;$taskRoot/src;$taskRoot/vendor/ud-tools/src/shared/python;$taskRoot/vendor/ud-tools/src"
$env:TOOLS_REPO_PATH = "$taskRoot/vendor/ud-tools"
$env:PYTHONUTF8 = '1'
$nativeTestFiles = @(
    'tests/unit/engines/myosuite/test_project_task_producer.py',
    'tests/unit/engines/myosuite/test_project_task_feedback.py',
    'tests/unit/engines/myosuite/test_project_task_forecast.py',
    'tests/unit/engines/myosuite/test_project_task_artifact_admission.py',
    'tests/unit/engines/myosuite/test_native_model_resource_closure.py',
    'tests/unit/engines/myosuite/test_project_task_native_search.py',
    'tests/unit/engines/myosuite/test_project_task_tracking.py',
    'tests/unit/engines/myosuite/test_project_task_command_solve.py',
    'tests/unit/motion_matching/test_bounded_candidate_search.py'
)
& $env:FEEDBACK_NATIVE_PYTHON -m pytest --noconftest -o addopts='' @nativeTestFiles -q --junitxml=f09n-native-campaign.xml
```

Record actual versions, exact source/resource hashes before and after, JUnit,
full solve metrics and committed-source correspondence. Preserve existing
receipts rather than replacing them. Repeat canonical local checks and normal
hooks before publication. The final fixture's declared criteria/reference and
physical scales are in the committed test source; production criteria are not
inferred from that fixture.

## Remaining Full Epic Work

Qualify model-specific hard criteria and reference mapping, optimize and execute
receding-horizon prefixes through the existing state-dependent recorder, and
benchmark actual complete time-to-accepted-match. First profile the measured
44.93-second synthetic solve and remove redundant prediction/guard work only
with complete cache identity and preserved adversarial admission. Native contact constraints
need their real laws and tolerances; SLSQP smoothness/performance is not assumed.
Reconcile the private Tools consumer/pin stack before production integration.
All seventeen production rows, six ecosystems, full private motion matching
and the muscular OpenSim full-state/full-horizon saved-excitation replay under
qualified own contacts remain required. The derived humerus source-coordinate,
branch, state/domain/conditioning and anatomy gates remain separate and open.
