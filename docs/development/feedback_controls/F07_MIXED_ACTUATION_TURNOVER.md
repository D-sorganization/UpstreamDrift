# F07 Mixed Actuation Replay Turnover

## Ownership and Scope

Replay prerequisite #12151 of parent #12147, child of #11791 and epic #11784. Implementation is a separate
assistance-explicit scalar T01 profile. The original muscle-only bundle still
rejects every non-muscle actuator. The restricted fixed-path Slider/Pin subset
does not yet admit either full-body production model. Parent #12147 composes
the constrained-muscle prerequisites locally and adds optional mixed Moco
request/export/replay dispatch. The statement that combined constrained and
mixed replay remains rejected records the earlier #12151/#12147 snapshot;
child #12173 now exercises that composition only under an explicit declared
constraint policy and narrow synthetic fixture. See
`F07_CONSTRAINED_MIXED_MOCO_TURNOVER.md` and canonical chapter48. Production
models remain unqualified.

## Reproduction

Use an existing OpenSim SDK environment; no new environment is required.
Set `PYTHONPATH` to the checkout. On Windows, set `CASADIPATH` to that
SDK's `Lib/site-packages/opensim` directory before native Moco tests.

```powershell
& <sdk-python> -m pytest tests/opensim/test_native_mixed_actuation.py tests/opensim/test_native_muscle_bundle.py tests/opensim/test_native_muscle_replay.py tests/opensim/test_native_moco_runner.py tests/opensim/test_native_moco_guess.py tests/opensim/test_moco_initial_bindings.py -q -o addopts=''
```

The synthetic model has two antagonistic Millard muscles, a ground Slider
coordinate and a relative Pin coordinate. Mechanical gains are 7 N and
13 N m per dimensionless control. Native excitation minimum is 0.01; no
artificial zero-excitation bounds override it. The positive replay is 40 ms,
not a private capture trial or standing equilibrium. A zero-assistance case
retains both mechanical channels with exactly zero bounds and sampled work.

## Validation and Failure Lineage

Planning receipts are retained under the fleet workspace
`docs/development/feedback_controls_planning`, with native RED/repair/GREEN
logs named `12147-*`. Initial missing-module tests preceded implementation.
Native probes exposed and corrected default-versus-live coordinate restriction
checks, ground-root classification by names, and rotational override units.
Actual player mismatch and mislabeled initial native time each failed before
the corresponding executor checks. Binary extension discovery now rejects an
empty native binary set even with multiple Python helper artifacts.

The broader OpenSim4.6 regression v2 passed 214 tests with two inapplicable
contact-fixture skips. Initial regression retained six CasADi DLL loading
failures because CASADIPATH was absent; the configured SDK rerun passed.
Later result-lineage and exact-time changes have their own focused receipts.
Final native regression after exact-time, source-read and result-lineage changes also passes 214 tests with the same two inapplicable fixture skips; all 23 new mixed-profile tests pass. OpenSim runtime is 4.6-2026-06-22-85aaf64 on Python 3.12.10. Record exact final commit and PR validation after publication.

## Next Implementation Gates

Integrate constrained and cold-start prerequisites using normal protected PR
flow. Reuse the existing all-scalar Moco bindings rather than weakening the
muscle-only endpoint. The constrained-plus-mixed integration is exercised only
for an explicit source-bound declaration and synthetic fixture; it does not
broaden factory-model admission. Broaden native topology through explicitly
tested profiles; a fixed-path two-DOF fixture is not factory-model admission.
Astra's derived Rajagopal route and #12150 MTP reduction remain separate
source/model qualification work. Ground support, grips, bilateral/trunk/upper
muscle anatomy, passive-force policy, full capture horizon, independent saved
excitation replay and 17-variant/six-engine parity remain open. Store only
truthfully labeled preview videos on Desktop; no matching video was generated
from these synthetic fixtures. Preserve private captures and retained failure
receipts, and remove only clean merged owned worktrees.

## Mixed Moco Link Validation

Full-suite CI at parent head `2c636a7336` exposed two setup errors: the
`mixed_model` fixture was unavailable through a collected test-module plugin.
Import the shared fixture explicitly into the mixed Moco test module rather
than relying on collection order. After this correction the actual OpenSim 4.6
mixed-actuation/Moco suites pass 25 tests; the default interpreter with two
xdist workers reports 25 SDK skips and no fixture errors. These are distinct
runtime results. Full-suite CI remains the final integration check; no native
or scientific acceptance threshold changes.

Run `tests/opensim/test_native_mixed_moco.py` with the same native SDK and
CASADIPATH. Requests bind all channel bounds and roles; JSON accepts only
explicit typed channel declarations. Export preserves absolute mixed paths and
every original control knot. Independent replay NPZ distinguishes physical
actuations, dimensionless controls, roles and output units, native power and
sampled work. Pure muscle evidence retains its original field names.

Initial TDD failed on the missing mixed request API. An unscaled tiny target
cost allowed an almost motionless optimizer success; a predeclared marker
weight of 1e6 resolves this numerical stopping problem. The first ten-interval
solve then replayed with about 0.69 micrometre marker RMSE but 0.0034 maximum
raw mixed-unit state error, prompting finer-mesh per-state checks. Retain these
failed runs under planning `12147-mixed-moco-*`; no production or capture
qualification follows from this synthetic exercise.

The twenty-interval diagnostic at constraint tolerance $10^{-7}$ and optimizer
convergence tolerance $10^{-6}$ passes separate synthetic replay limits:
coordinate value $10^{-7}$ m or rad, speed $10^{-5}$ m/s or rad/s, activation
$10^{-3}$, and fiber length $10^{-5}$ m. Observed maxima are about
$9.97\times10^{-9}$ m translation, $3.85\times10^{-7}$ m/s speed,
$6.07\times10^{-4}$ activation and $3.99\times10^{-6}$ m fiber length;
marker RMSE is about $7.58\times10^{-7}$ m. These are fixture numerical
checks, not biomechanical thresholds. A forty-interval attempt with tighter
$10^{-9}/10^{-8}$ tolerances hit the configured test timeout; the failed log
is retained and no faster-solve claim is made.

## Prerequisite Refactor Integration

The preserved constrained stack is composed through parent commit f1c57928e9.
Its bounded request parsing, passive/reference/replay helpers and owned native
input-player refactor are retained; mixed request parsing and explicit admission
are integrated into those helpers. Architecture budget and generated context
checks pass. The post-merge native regression passes 239 tests with three
explicit fixture/opt-in skips (62.78 s).

Pre-PR attempt one exposed a missing change-fragment header and two unrelated
CLI subprocess import failures with Repository_Management ahead of this
checkout on PYTHONPATH. The header and path ordering were corrected; the
standalone four orchestrator CLI tests then passed. Attempt two overlapped
local merge conflict resolution and collected transient conflict-marker syntax
errors; it is not validation evidence for the resolved tree. The third full central run used the frozen resolved source and passes all five
gates: lint/format, diff mypy, affected tests (247 passed, 35 explicit generic
SDK/opt-in skips), Semgrep/import policy and policy/fragment validation. The
separate actual OpenSim regression above supplies native execution evidence.

## Publication Gate Correction

CI on af321fdc93 found one failing generated lifting-report freshness test.
Inherited formatting changed the report bytes while its generator and source
receipt stayed unchanged. Restoring the authoritative main report and excluding
that byte-exact generated artifact from Prettier passes all11 report tests.
The original failed log is retained; this correction changes no dynamics.
