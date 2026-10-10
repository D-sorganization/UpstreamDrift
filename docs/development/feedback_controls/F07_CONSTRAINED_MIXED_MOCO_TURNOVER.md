# F07 Constrained Mixed Moco Turnover

Child #12173 composes explicit constrained cold-start admission with the
assistance-explicit mixed Moco/T01 path. The source branch is
`feat/f07-constrained-mixed-native-12173`. The canonical calculation is
`manuals/upstreamdrift/chapters/48-native-constrained-mixed-moco.qmd`.
This is software integration evidence only; it does not close F07, F08, the
17-variant denominator, six-engine parity, or release.

## Behavior Frozen by the Declaration

Integration follow-up retains the parent mixed Moco fixture correction and
explicitly exports its dependency fixtures in the constrained test module.
This removes dependence on collected test-module plugin order. After the
normal parent/main merge, the three affected suites pass 34 actual OpenSim 4.6
tests; the default interpreter with two workers reports 34 SDK skips without
fixture errors. These follow-up checks do not replace the earlier 197 native
regressions or establish full-suite CI/scientific acceptance.

The combined route is opt-in through `DeclaredColdStart`. The caller declares
exact source and named continuous state, clock, lock targets, constraint
enforcement, coordinate charts, optional linear chart rules and residual
tolerance. The implementation leaves native constraints in place and audits
their actual enforcement and residuals at every replay sample. Moco
preparation requires each state optimizer box to fit inside its declared
coordinate interval. For a linear rule it computes endpoint extrema for each
weighted coordinate interval and rejects if the implied sum interval is not
contained in the declared range. It does not infer lock targets or disable
constraints. Without the opt-in declaration, the original strict defaults
remain unchanged.

Replay creates a fresh native source model. Initialization of the one
`Manager` must preserve every requested named state value and exact initial
time; hidden projection fails before output. The replay retains exact saved
input knots, compares physical native time at every knot, and emits separate
applied controls and force/torque outputs. Constraint digests plus position
and velocity residual observations accompany the mixed result.

The mixed profile binds ordered scalar actuator paths, native law, role,
bounds, force scale, coordinate and units. Muscle excitation remains a
dimensionless input whose output is nonlinear force in N. Coordinate actuator
input is dimensionless and scales to N for translation or N\*m for rotation.
Root-residual, upper-assistance and leg-reserve are explicit role labels;
root topology is checked against the actual native parent frame. Role labels
and correct units do not establish production assistance validity.

## TDD and Validation

All five central pre-PR gates passed: lint/format, diff-scoped mypy (eight
production files), affected tests, Semgrep/import policy, and policy/fragment
validators. The central Python 3.13 run had 14 portable passes and 184 explicit
SDK/source-fixture skips; those skips are separate from the actual OpenSim
4.6 runtime evidence below. The architecture budget and design-manual
governance checks passed; governance remains release-blocked with zero
registered calculations.

The new native test `tests/opensim/test_native_constrained_mixed.py` covers a
two-Millard, rotational Pin and explicit coordinate coupler. It checks fresh
replay, exact applied mechanical command and physical N\*m output, per-knot native enforcement and residuals, immutable
state samples, changed-future assistance with an identical common prefix,
source refusal without declaration, policy mismatch, command/controller
conflict, and Manager projection rejection for both value and speed seeds in
both mixed replay and mechanical prepared-state paths (four cases).
It also executes maintained mixed Moco preparation, solve, exact-knot T01
export, fresh combined replay and scoring. Portable regressions cover chart
interval proof and scalar executor policy contracts.

The frozen combined affected regression completed with **73 passed, 1
skipped** in 63.60 seconds. The single skip is unavailable optional provider
coverage. A separate successful synthetic Moco v2 run took 20.49 seconds with
a 40 ms horizon and marker weight $10^4$. Earlier hidden-projection tests
failed before the Manager seed check was added. The fixture's declared
residual tolerance and replay differences are software test bounds, not
clinical or physiological tolerances.

The disjoint existing pure-muscle/contact and Moco-runner regression completed
with 124 passed and two skips in 66.39 seconds. Together, the selected combined
and existing regressions total 197 passed and three skips with no overlapping
tests.

Run the focused tests serially in the installed OpenSim 4.6 Python 3.12
environment with matching `CASADIPATH`:

```powershell
& <sdk-python> -m pytest tests/opensim/test_native_constrained_mixed.py tests/opensim/test_native_mixed_moco.py tests/opensim/test_native_prepared_state.py tests/opensim/test_native_constrained_muscle.py tests/opensim/test_native_mixed_actuation.py tests/opensim/test_native_muscle_bundle.py tests/opensim/test_native_muscle_replay.py tests/opensim/test_native_moco_runner.py tests/unit/engines/opensim/test_native_moco_chart_bounds.py tests/unit/engines/opensim/test_native_scalar_policy.py -q -o addopts=''
```

Do not treat the 40 ms fixture as a private capture or full-horizon result.
Production driver/iron source anatomy, marker calibration, passive forces,
complete native initial-state evidence, external-resource closure,
contact/grip, assistance policy, physiology and all-variant/six-engine parity
remain release blockers. The design-manual calculation registry still has
zero registered calculations and remains blocked-inventory-required.
