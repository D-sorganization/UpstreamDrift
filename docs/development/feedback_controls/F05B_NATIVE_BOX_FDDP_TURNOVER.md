# F05b Derivative-Aware Native Candidate Turnover (#11910)

This child is a restricted executable BoxFDDP candidate for F05 #11789. It
uses the existing Crocoddyl 3.2.1/Pinocchio 4.1.0 runtime and the pinned
Tools T01 replay contract; acados was surveyed but not installed. No package
or base Crocoddyl environment was changed. A task-owned WSL overlay adds
only supported MuJoCo 3.8.0 for the paired native test.

`native_box_fddp_hinge.py` admits exactly a contact-free one-hinge RK4 plant
with one direct unit motor. Its analytic discrete $A,B$ are checked against
native steps and independent finite differences. `box_fddp_tracking.py`
augments nominal/perturbed scenario states, bounds the shared motor input,
uses Crocoddyl `SolverBoxFDDP`, and independently checks max-scenario cost,
state/input bounds, full observation clock and fallback safety before applying
the first command. Loaded native models are checked again before execution.
The existing F05a native adapter drives the plant, records actual post-limit
ZOH torques, then independently replays a Tools T01 bundle from the complete
initial state. It compares q, v and full integration state to 1e-12. Solver
exceptions, timeouts, stale/cancelled observations and invalid candidates
retain labeled fallback; no warm plan survives an unaccepted call. The
cooperative wall budget is not a hard deadline.

The test-first sequence observed missing-module RED, then native derivative
and independent replay GREEN. A provider exception test went RED on the
unhandled Crocoddyl exception and GREEN after explicit fail-closed fallback.
A loaded-model derivative mismatch test went RED before the native wrapper
existed and GREEN after model-boundary validation. Strict unsupported
integrator, non-unit actuator, gravity and unsafe fallback cases pass.

The source-controlled first and repeat receipts
`F05B_NATIVE_PAIRED_RECEIPT_MJ38.json` and
`F05B_NATIVE_PAIRED_RECEIPT_MJ38_REPEAT.json` are generated from flagged JUnit
metrics. It requires both predeclared 0.4/0.7 rad cases, all companion tests
green, source hashes, exact Tools pin, software/runtime identity, native
model/initial-state/policy/input hashes, all command statuses and measured
timings. It charges common T01 bootstrap to both alternatives, then reports
Box derivative setup and both methods' controlled execution plus input export
and independent replay. Python launch/pytest collection are outside that
timing; hardware load and provider versions can change results. BoxFDDP's
internal average-scenario cost and SciPy's max-scenario optimization objective
are different, though post-solve acceptance uses the same max-scenario score.

In both supported WSL runs, BoxFDDP and SciPy each admitted 8/8 steps in both
trials. BoxFDDP median controller-call latency was 0.38–0.68 ms, versus SciPy
29–42 ms, while q RMSE was effectively tied within the two predeclared starts.
Cold common T01 bootstrap varied from 1.93 to 4.15 s. Controlled execution,
export and replay varied 0.14–0.31 s for Box and 0.40–0.59 s for SciPy;
Box ran first in each paired test, so cache order remains a measurement limit.
This supports keeping BoxFDDP as a one-hinge candidate,
not selecting it for full-body production. The existing F02 TVLQR remains the
broader fallback/default, including F05a's separately measured Windows
native comparison. F05 #11789 remains open; contact/grip, manifold,
multi-engine, muscle, private capture, held-out full swings and real-time
deadline evidence are still absent.

Reproduction after entering the checkout in the task-owned WSL overlay
(substitute the local overlay interpreter if it lives elsewhere):

```bash
F05B_BENCHMARK_RECEIPT=1 python -m pytest -q tests/unit/motion_matching/test_box_fddp_hinge.py -o junit_family=legacy --junitxml=docs/development/feedback_controls/F05B_BENCH_JUNIT.xml
python scripts/f05b_native_paired_receipt.py --junit docs/development/feedback_controls/F05B_BENCH_JUNIT.xml --host-alias local-supported-runtime --output docs/development/feedback_controls/F05B_NATIVE_PAIRED_RECEIPT_MJ38.json
```

The temporary JUnit file is not a source artifact. The canonical design-manual
chapter is `manuals/upstreamdrift/chapters/24-native-box-fddp-candidate.qmd`;
its calculation-registry row remains blocked.
