# F05c Native Multi-DOF Tangent Turnover (#11920)

This child extends the native control boundary from one hinge to a small
floating-root, two-hinge plant with $n_q=9$, $n_v=8$ and two bounded direct
unit motors. It is a derivative and frozen-input replay fixture, not a
qualified multi-DOF BoxFDDP controller. The unactuated root's response is
native plant dynamics; no generalized/root effort is synthesized or injected.

`native_tangent_derivative.py` admits MuJoCo Euler only, no gravity/contact,
plugin, muscle activation, equality constraint, hidden gravity compensation
or external force. It restores the complete `mjSTATE_INTEGRATION` initial
state, checks quaternion normalization and motor order/limits, obtains
MuJoCo's actual discrete `mjd_transitionFD` matrices, and independently steps
the same native plant with the held command. In tests, selected tangent-state
columns and both motor columns match separately perturbed native steps using
`mj_integratePos`/`mj_differentiatePos`. Reversing the motor declaration and
command order reverses $B$ columns while preserving physical motion.

The test-first sequence observed missing-provider RED, then a native-state
normalization failure that exposed the fixture's rounded quaternion; the
fixture now calls native quaternion normalization. A Tools T01 schema-name
check also failed RED before the test was corrected to the pinned
`state_schema.components[].dimension` contract. Four focused supported
MuJoCo 3.8 tests pass. A separate fixed three-step two-motor torque history
with terminal ZOH sentinel replays from the complete initial native state
through F06/Tools T01 and matches every full integration-state row within
1e-12. Non-Euler integration, out-of-limit or at-limit central probes, and hidden assistance
reject. The exact Tools pin and model/initial-state/policy/input/time-grid
digests are in `F05C_NATIVE_TANGENT_RECEIPT_MJ38.json`.

Ten derivative repeats on the task-owned WSL runtime had median 1.56 ms,
p95 3.23 ms and worst 3.61 ms; first bundle setup took 3.37 s and fresh
replay 0.060 s. Those figures include machine-load and cold-import effects.
The derivative cost alone does not certify any 10 ms end-to-end control
deadline. There are no source observations or private mocap files here.
The next controller slice must linearize along full native rollouts, respect
the moving manifold reference and validate candidate first commands with
the nonlinear native plant and independent replay before comparing with F02.
Contact/grip, muscle excitations, full-body topology, capture accuracy and
six-engine parity remain open under F05/F09/F10; parent #11789 stays open.

Reproduce from this checkout with MuJoCo 3.8.0 and the pinned Tools submodule:

```bash
F05C_BENCHMARK_RECEIPT=1 python -m pytest -q tests/unit/motion_matching/test_native_tangent_derivative.py -o junit_family=legacy --junitxml=docs/development/feedback_controls/F05C_BENCH_JUNIT.xml
python scripts/f05c_native_tangent_receipt.py --junit docs/development/feedback_controls/F05C_BENCH_JUNIT.xml --host-alias local-supported-runtime --output docs/development/feedback_controls/F05C_NATIVE_TANGENT_RECEIPT_MJ38.json
```

The JUnit XML is temporary, and the receipt generator fails on missing,
skipped or failed predeclared tests. The canonical provisional calculation
is chapter 25; its registry row remains blocked.
