# Native Thelen Replay Turnover

## Scope and Executed Evidence

Child #11826 extends early F07 replay to explicitly admitted Thelen2003Muscle
alongside Millard2012EquilibriumMuscle. The test-first commit `f6c04ba5a2`
records two failures: missing law identity and Thelen refusal. Commit
`b00a46222a` adds ignored-mode admission and independent restored-force tests.
Actual OpenSim 4.6-2026-06-22-85aaf64 on Python 3.12.10 executes both compliant
native fixtures. The focused suite plus governance passes 67 tests.

```text
python -m pytest tests/opensim/test_native_muscle_replay.py tests/scripts/test_design_manual_governance_contract.py --override-ini=addopts= -q
```

Use an interpreter with the reported native provider; default CI skips do not
establish native evidence. The shared task environment is local infrastructure,
not a clean-host provider receipt. Both muscle laws retain complete physical
state without equilibration after restoration. The independent native force
at the supplied state agrees with replay's initial force. Refinement, applied
excitation checks and hidden-drive regressions run for both subtypes.

## State-Policy Decisions and Failed Experiments

Native per-instance minimum activation and fiber length are authoritative.
The first expanded suite exposed a Millard-specific test assumption: `1e-8 m`
is below its native fiber minimum but is above the zero-pennation Thelen
minimum of zero. The test now queries the actual native minimum and probes
below it. Positive fiber length remains a separate physical precondition;
no universal Millard numerical floor is imposed on Thelen.

Native Thelen initialization itself throws for ignored activation/tendon
modes. Millard accepts such modes but changes the state/control relationship.
Both therefore fail the adapter's explicit policy before native initialization.
Separate mode/layout tests are required before supporting them. This leaves
the reviewed full-body candidate's 34 rigid-tendon Millard instances pending.

The unit-test lane explicitly fetches the default branch to its remote-tracking
reference. F02's executed CI exposed that a stacked PR otherwise fetches only
its feature base into FETCH_HEAD, leaving ownership guards without origin/main.
This mirrors that scoped repair; no required check or threshold is weakened.

The policy records ordered concrete laws and changes adapter version to
`native-muscle-replay/1.2.0`. Model bytes bind serialized subtype parameters;
external-resource identity remains a separate gate. The contact extension
PR #11820 is a sibling dependency; reconcile its policy builder and recursive
audit when both branches merge, retaining native-law and force-policy fields.

## Remaining Gates

F07 stays open. This work does not qualify the public full-body candidate,
inherited asset licensing, wrist anatomy, muscle path geometry, grip/contact,
physiological recruitment or real capture matching. Native state derivatives
and full-model resource/subtype readiness need further qualification. The
separate private D01 protocol preserves original observation clocks and reports
missing analog/force-platform/event channels instead of inventing evidence.
