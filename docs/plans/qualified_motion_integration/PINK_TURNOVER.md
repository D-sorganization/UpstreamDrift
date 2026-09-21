# Pink Full-Body Pipeline Turnover

Read [common dispatch rules](TURNOVER.md) and [numerical boundaries](README.md)
first. This packet implements epic #10254 W3; #10257 / PR #10267 repairs the
low-level adapter only. Audit basis: integration merge `722540e707df0d0134ae32cfc1b227b3fb1e94a7`,
including main `0ec64e45f1f259c11e26e36315ee370a5c51c7d9`.

## Existing Behavior and Reuse

The production ground-support lane constructs MuJoCo kinematics. The separate
`PinocchioFullBodyIK` uses SciPy least squares and position-only soft closure.
Neither selects Pink. `dtack`'s Pink enum and the GUI Solve IK stub are not
production wiring. Reuse calibration, capture masks/times, canonical anatomy,
reference smoothing and receipt stages in `motion_matching.pipeline`.

PR #10267 supplies validated configuration refresh, finite checks, native error
propagation, hard-constraint/limit forwarding and exactly one `pin.integrate`
per QP step. Do not recode these behaviors in the pipeline. Runtime #10262
already probes actual feasible/infeasible Pink solves; extend its shared
capability result rather than creating another independent installation probe.

The qualified Pink 4.4 API accepts `constraints` separately from `limits`.
`RelativeFrameTask` supplies a six-component log error and `6 x nv` Jacobian.
Do not substitute `LinearHolonomicTask` for nonlinear grip closure or pass a
soft task and claim hard equality. Verify sign, frame placement and log-map
Jacobian against the repository's finite-weld contract before reuse.

## Decisions Fixed by the Integration Owner

- Pink becomes an **explicit selectable experimental IK backend**. Existing
  MuJoCo selection remains the default until matched qualification. Requested
  Pink with missing capability is an error; no silent fallback.
- Runtime grip closure is full SE(3), with separate metre/radian residuals.
  Address placement fitting remains calibration and cannot substitute for
  runtime equality. The principal-log branch at pi is rejected consistently
  with #10260. A user must not choose a weaker weld to make a fit pass.
- Coordinate order, attachments, locked coordinates and document bounds come
  from the canonical lane/model contract. Support `nq != nv` in the adapter;
  derive joint configuration/tangent slots instead of slicing by assumption.
- Stage input times are finite and strictly increasing. Every physical
  interval uses its own duration. Existing fixed-rate filtering may reject
  nonuniform samples until its owner supplies a tested resampling contract;
  do not silently apply `RATE_HZ` to irregular data.
- Calibration/frame projection iterations use a separately named numerical
  projection step. These are not physical time. Position limits apply during
  projection; the resulting trajectory must pass a separate capture-time
  velocity/acceleration audit. Do not claim physical velocity limits merely
  because each inner QP obeys a bound over its artificial projection step.
- A mode that claims physical rate limits must use the previous accepted
  physical frame and actual capture interval, integrate exactly once per
  physical step and report failure when one step cannot satisfy the declared
  residuals. Do not run multiple full-interval updates per capture frame.
- Smoothing belongs to the existing reference stage. Re-audit/reproject
  constraints afterward through the same backend. Smoothing cannot confer
  feasibility or repair an infeasible result without a new solver result.

## Packet P1: Task and Kinematics Translation (#10276)

**Ready for bounded dispatch after #10257 and #10260 are available at recorded
dependency SHAs.** Own a focused engine-local Pink full-body task module beside
`src/engines/physics_engines/pinocchio/python/full_body_ik.py`, plus its unit
and real-engine integration tests. No CLI, GUI, lane factory or receipt edits.

Provide a typed facade with at most three public arguments:

```python
class FullBodyPinkTasks:
    def build(self, request: FrameTaskRequest) -> FrameTaskBundle: ...
    def audit(self, configuration: ConfigurationState) -> FrameResiduals: ...
```

`FrameTaskRequest` contains canonical named marker targets and validity mask,
posture target, declared stance/closure policy and immutable options.
`FrameTaskBundle` separates soft tasks, hard equalities and limits.
`FrameResiduals` reports named marker errors, translation/rotation closure,
bound violations and stance errors after integration. Native Pink objects
remain engine-local; shared callers receive plain typed arrays/diagnostics.
Reuse an existing equivalent type if discovery finds one; do not introduce
parallel canonical model or pose types.

RED tests must cover marker-offset Jacobians under nontrivial rotation,
canonical versus permuted coordinate order, displaced six-dimensional weld,
near-pi rejection, valid free-flyer `nq != nv`, bound-unit conversion, locked
coordinate semantics, dropout masks, unknown/duplicate names and invalid
transforms. Invalid unmasked targets fail before native calls; masked targets
must never enter QP residuals. Zero observed targets produces an explicit
insufficient-data result, not a successful posture-only fit.

GREEN requires independent directional finite differences and a real Pink QP
whose closure/limits are checked after manifold integration. Freeze exact
test tolerances in the tests based on derivative scale and #10260's contract;
do not treat those numerical tolerances as physical fit thresholds.

## Packet P2: Trajectory Service and Timing (#10277)

Dispatch after P1 review. Own one shared trajectory protocol/options/results
module, its engine-local Pink implementation and tests. Coordinate any change
to `pipeline/lane.py` with the integration owner; do not rewrite calibration.

```python
class ConstrainedIKBackend(Protocol):
    def solve_trajectory(
        self, request: IKTrajectoryRequest, options: IKOptions
    ) -> IKTrajectoryResult: ...
```

The immutable request owns initial state, named targets/masks, capture times
and the canonical model/calibration identity. Options explicitly identify
projection or physical-step mode, QP solver, iteration budget, tolerances and
limit policy. Result contains configurations, per-frame convergence/failure,
post-step residuals, rate audits, timings and actual backend identity.
Physical failures cannot be replaced by the previous frame under a success
status. Partial results retain a failed frame index and cannot qualify a run.

RED: iteration count cannot multiply physical elapsed time or allowed motion;
two different capture intervals change rate audits correctly; cache refresh
from a distant initial state matches a fresh solver; conflicting limits or
closure return structured infeasibility; dropout recovery is deterministic;
smoothing requires a new constraint audit; cancellation is honored at frame
boundaries without modifying caller arrays. Preserve the complete time grid.

GREEN: real multi-frame Pink execution with named marker tasks and hard closure,
independent residual/rate recomputation and exact backend identity. Compare
with least squares using identical model/calibration/masks; report accuracy,
temporal continuity, failures and wall time. Full physical qualification stays
separate. Dynamics replay remains the selected canonical dynamics backend;
Pink is an IK library, not a new forward-dynamics engine.

## Packet P3: Existing Product Surface (#10278)

Dispatch after P2 signature freeze. Own the existing `MatchRequest` in
`src/tools/motion_matching/pipeline.py`, its CLI/GUI consumers and the current
receipt schema/producer. No solver math in this packet.

Forward a backend enum and typed options through the existing command and
lane service. Reuse the runtime capability facade; defer native imports until
selection. Export a versioned receipt block with backend, solver/runtime
versions, source/model/capture identities, actual task/constraint/limit policy,
time semantics, complete per-frame status and qualification state. Coordinate
with #10271's chain validator; one producer owns provenance normalization.

RED: default remains unchanged, explicit Pink reaches a spy service, then a
real native backend; unavailable dependency has a useful failure; no fallback
mislabel; failed/cancelled runs cannot produce qualified receipts; serialized
options round-trip; UI and CLI consume the same service and diagnostics.
GREEN requires both-club user journeys, progress/cancel/export, parity ledger
and agent-context updates through their canonical generators.

## Validation Commands and Stop Conditions

Run unit tests separately from native tests. Example adapter regression commands
after the prerequisite PR is present, from the selected repository root:

```bash
python3 -m pytest -q tests/unit/engines/pinocchio/test_pink_tasks.py tests/unit/engines/pinocchio/dtack --timeout=60
/home/dieterolson/.local/bin/micromamba run \
  -p /home/dieterolson/.venvs/upstream-crocoddyl-conda-10254 \
  python3 -m pytest -q tests/integration/engines/pinocchio/test_pink_tasks.py \
  tests/integration/engines/pinocchio/test_pink_adapter_contract.py \
  --timeout=60 --junitxml=/tmp/pink-tasks.xml
```

Check that the named integration file exists on the recorded prerequisite head
before running it. Add the new P1/P2 test paths explicitly to the appropriate
separate commands; require real imports, executed tests and zero skips.
Use the common turnover's lint/type/architecture/governance gates.

Do not dispatch P2 while closure frame conventions or physical-time semantics
remain ambiguous. Do not dispatch P3 while result/failure contracts are changing.
Do not widen bounds, choose new physical thresholds, suppress infeasibility or
modify legacy parity gates to obtain a pass. Return those decisions to the
integration owner with a minimal reproducible test.
