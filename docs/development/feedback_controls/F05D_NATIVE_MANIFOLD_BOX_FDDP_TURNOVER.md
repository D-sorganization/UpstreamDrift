# F05d Native Multi-DOF Manifold BoxFDDP Turnover (#11932)

This child extends F05c's actual MuJoCo Euler tangent derivatives into a
receding-horizon Crocoddyl BoxFDDP controller for the same floating-root,
two-hinge synthetic plant. It is stacked on F05c PR #11922 and keeps F05
parent #11789 open. The physical state is 17 values, its tangent is 16,
and only the two compiled hinge motors can receive torque. Native manifold
`diff`/`integrate` handle the quaternion; the action uses actual native
`mj_step` and F05c `mjd_transitionFD`, with a Gauss–Newton tracking-cost
Hessian. The manual calculation and units are in provisional chapter 26.

The controller accepts a BoxFDDP proposal only after the same nonlinear
native plan, reference, torque bounds and effort cost beat a verified
zero-torque fallback. It checks complete finite state, compiled-model
identity, actuator order, hidden callbacks, contact-free policy, exact
observation clock and cooperative solve budget. A rejected plan clears its
warm start and records a specific fallback status. The execution path
reuses the existing generic direct-torque runner: it records actual
post-limit ZOH commands, freezes a pinned Tools T01 bundle from complete
`mjSTATE_INTEGRATION`, then a fresh MuJoCo instance reproduces every full
integration-state row within $10^{-12}$. The solver and observation data
are absent from that replay. The generic runner retains all F05a/F05b
one-hinge behavior; it did not add a second replay protocol.

The test-first sequence saw missing-module RED, then six manifold/action/
native-control tests GREEN. A missing multi-DOF runner test was RED before
the common executor was extracted and GREEN afterward. Additional
negative tests cover unsupported gravity assistance, hidden callback,
actuator-order/model mismatch, model mutation, false solver success with
out-of-bound torques, stale observation, and malformed receipt evidence.
Twelve native F05d cases and four receipt-parser cases pass on the
task-owned supported WSL overlay; 13 F05b/F05c native regressions pass.
The broader F05a native test module cannot collect in this sparse overlay
because unrelated `structlog` is absent. No dependency was installed or
runtime changed for this child.

The source-controlled
`F05D_NATIVE_MANIFOLD_RECEIPT_MJ38.json` binds the exact source files,
test XML, Tools pin, provider versions, compiled/source model, initial
state, policy, time grid, applied-input hashes and the two predeclared
0.4/0.7 rad starts. Crocoddyl BoxFDDP and SciPy SLSQP native shooting use
the same truth plant, native-generated reachable moving reference,
four-step horizon, 15-iteration
cap, torque bounds, nonlinear objective and 0.2 s cooperative call
budget. Solver order reverses between starts. A common T01 preflight is
charged to both; each run also charges controller setup, every failed or
accepted call, execution, export and replay. BoxFDDP accepts 3/8 steps;
SciPy accepts 0/8. Every other step applies the recorded zero-torque
fallback. The realized native costs are 31.93 versus 254.35 at 0.4 rad
and 61.80 versus 201.83 at 0.7 rad (BoxFDDP versus SciPy), with all
solver statuses and tracking errors in the receipt. Neither method has
a fully accepted four-step run under the 0.2 s budget. Load and warmup
vary. This does not certify a hard 10 ms deadline or a production winner.

To reproduce from this checkout in an environment with MuJoCo 3.8.0,
Crocoddyl 3.2.1, Pinocchio 4.1.0 and the pinned Tools T01 submodule:

```bash
F05D_BENCHMARK_RECEIPT=1 python -m pytest -q tests/unit/motion_matching/test_native_manifold_box_fddp.py -o junit_family=legacy --junitxml=/tmp/f05d-native-manifold.xml
python -m pytest -q tests/unit/motion_matching/test_f05d_native_receipt.py
python -m scripts.f05d_native_manifold_receipt --junit /tmp/f05d-native-manifold.xml --host-alias wsl-task-overlay --output docs/development/feedback_controls/F05D_NATIVE_MANIFOLD_RECEIPT_MJ38.json
```

The receipt generator rejects absent, skipped, failed or partially
recorded native trials. JUnit XML is temporary and not committed. The
synthetic plant has no measured capture, anatomy, grip/ground contact,
muscle excitation or uncertain contact dynamics. State limits beyond
finite/contact-free native integration are not certified. F02 TVLQR
remains the broader default; F05/F09/F10 qualification remains open.
