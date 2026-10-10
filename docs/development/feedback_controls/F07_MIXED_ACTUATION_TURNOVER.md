# F07 Mixed Actuation Replay Turnover

## Ownership and Scope

Replay prerequisite #12151 of parent #12147, child of #11791 and epic #11784. Implementation is a separate
assistance-explicit scalar T01 profile. The original muscle-only bundle still
rejects every non-muscle actuator. The restricted fixed-path Slider/Pin subset
does not yet admit either full-body production model. Moco mixed dispatch is
follow-on work after the constrained-muscle stack #12142/#12149 integrates.

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
muscle-only endpoint. Broaden native topology through explicitly tested
profiles; a fixed-path two-DOF fixture is not factory-model admission.
Astra's derived Rajagopal route and #12150 MTP reduction remain separate
source/model qualification work. Ground support, grips, bilateral/trunk/upper
muscle anatomy, passive-force policy, full capture horizon, independent saved
excitation replay and 17-variant/six-engine parity remain open. Store only
truthfully labeled preview videos on Desktop; no matching video was generated
from these synthetic fixtures. Preserve private captures and retained failure
receipts, and remove only clean merged owned worktrees.
