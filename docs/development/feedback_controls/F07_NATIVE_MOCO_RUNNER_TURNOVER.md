# Native Offline Muscle Matching Runner Turnover

## Scope and Source

Issue #11968 adds the maintained `tour_matching.cli moco-native` route. It
loads a strict request JSON and invokes the existing Moco builder, complete
named-state binding, native passive/constraint/reference observers, T01
serialization, independent native replay and marker geometry. The CLI's
`--prepare-only` stage retains blockers without constructing a solver. The
full route writes `preparation.json`, numbered `solve_attempts/*.json`,
`solve.json`, `solution.sto`, `native_replay_bundle.json`, `export.json`,
`native_replay.npz`, `replay.json` and `score.json`. `score.json` is a physical
replay diagnostic, not an acceptance certificate.

The branch is stacked on #11956 and includes genuine source merges of the
#11966 reference observer and #11903 native marker geometry. It is unarmed
until the parent dependencies land. Relevant source modules are
`native_moco_request.py`, `native_moco_runner.py`, `native_moco_replay.py`,
`moco_tracking.py`, `trc.py` and `cli.py` in the OpenSim `tour_matching` package.
The canonical calculation and limitations are in chapter 39. This is a new
software entry point, not a new optimizer, parser, anatomy model or evidence
schema.

## Executed Synthetic Evidence

OpenSim `4.6-2026-06-22-85aaf64` with native Millard and Thelen laws executed
file-driven solves, exact-knot T01 export/reload, fresh observation-free replay
and masked scoring. The two-DOF/two-muscle case uses source-native ordered
channels `zeta`, `alpha` and reversed guess-state order. Both 2- and 4-mesh
intervals use the same 11 observation times over 10 ms. A retained local
synthetic run reported:

| Law     | Mesh Intervals | Native Knots | Solve Wall (s) | Export Wall (s) | Replay Wall (s) | Marker RMSE (m) | Optimized vs. Replay State Max |
| ------- | -------------: | -----------: | -------------: | --------------: | --------------: | --------------: | -----------------------------: |
| Millard |              2 |            5 |          0.358 |           0.077 |           0.225 |       2.3072e-5 |                       0.004381 |
| Millard |              4 |            9 |          0.291 |           0.065 |           0.143 |       2.3211e-5 |                       0.000861 |
| Thelen  |              2 |            5 |          0.266 |           0.063 |           0.268 |       1.6992e-5 |                       0.001705 |
| Thelen  |              4 |            9 |          0.345 |           0.077 |           0.120 |       1.7066e-5 |                       0.000269 |

These are single local diagnostic runs, not latency distributions or an
algorithm-winner claim. The table omits fixture creation, Python/OpenSim
initial import, preparation and scoring, so it is not total time to an
accepted match. Each actual receipt also retains CPU time; the runner now
records preparation wall/CPU as well. A separate same-input native diagnostic
compared RK-Merson accuracy $10^{-8}$ (the T01 policy) with $10^{-9}$ on the
same synthetic source, clock and excitation values. Across the four runs,
maximum continuous-state deviations were 0.69–1.53e-9 and force deviations
2.01–9.67e-8 N. Because accuracy differs, those are **different numerical
policies**, not a same-policy reproduction claim. The marker error did not
decrease monotonically with mesh. No scientific threshold was set from these
outputs.

TDD captured the absent CLI, improper marker-placement admission, stale
registered reference, loss of original 360 Hz clock under nine-decimal TRC
formatting, missing unused-reference and assistance blockers, and overwritten
failed solver attempts before corrections. The first attempted 3-frame native
solve lacked Moco's required six spline data points and was abandoned;
retained 11-frame tests then solved. Injected native construction failure now
remains as numbered attempt 1 after successful attempt 2. Synthetic replay
tests poison the observation provider and prove the fresh replay does not
access it.

Run the focused tests from the repository root with the qualified OpenSim 4.6
Python 3.12 environment, repository root on `PYTHONPATH`, and `CASADIPATH`
pointing to that environment's OpenSim package directory:

```text
python -m pytest tests/opensim/test_native_moco_runner.py tests/opensim/test_trc_export.py -q -o addopts= --timeout=50
```

The broader native dependency regression of Moco bindings, reference,
passive, muscle bundle/replay, marker geometry, constraint state and TRC
completed with **250 passed, 2 skipped** in the owned OpenSim 4.6 runtime.
That environment needed the repository-pinned `opencv-python-headless`
4.13.0.92 wheel for the marker-geometry test module to collect; it was
installed without upgrading NumPy/OpenSim. The two skips remain explicit and
are not counted as native acceptance.

The central five-gate pre-PR run passed lint/format, diff Mypy, Semgrep/import
policy and policy/fragment checks. Its generic affected-test mapping ran 337
tests: 298 passed, 38 skipped, and one unrelated SG optimizer CLI test failed.
That test supplies legacy bare flags where its unchanged parser requires the
`run` subcommand. The same test failed with `SystemExit: 2` in the unchanged
main checkout under the same interpreter and `PYTHONPATH`; both the SG CLI
source and test have identical Git blobs in main and this branch. The new
OpenSim CLI entry point remains part of this change, and this baseline failure
is not counted as a passing central gate.

Baseline reproduction used main checkout `50784017c607f764e9c28db2d3c66b8211211f7a`,
`UpstreamDrift/.venv/Scripts/python.exe`, `PYTHONPATH=<main checkout>;Repository_Management`,
and `python -m pytest -q tests/integration/sg_optimizer/test_cli.py
tests/unit/motion_pipeline/orchestrator/test_cli.py --tb=line --junitxml=...`.
The private planning JUnit artifact is `moco_11968_baseline_main_cli.xml`
(SHA-256 `8d2ef8a32b3f1d994ece676b7312c03844d7ddeeeeb3251e866bec0b543fd53ed`):
one SG failure and four orchestrator passes. Both SG source and test Git blobs
match this branch (`705f8bee8fbd66f85cea8d4691d7a95cf53d7446` and
`c30792cba4ebde726b6a371872b2e44146673600`).

For a reviewed request JSON, use:

```text
python -m src.engines.physics_engines.opensim.python.tour_matching.cli moco-native --request REQUEST.json --output-dir OUT --prepare-only
python -m src.engines.physics_engines.opensim.python.tour_matching.cli moco-native --request REQUEST.json --output-dir OUT
```

The input JSON includes exact source/TRC/guess SHA-256, `bindings`
(`state_bounds`, `initial_state`, `control_bounds`), Moco `config`, positive
`marker_weights`, absolute-frame `marker_bindings`, capture `registration`,
absolute `reference_frame_path`, reviewed `passive_policy` and explicit
`excluded_markers` with reasons. Unknown fields fail closed. The full command
must only be used for a scientifically reviewed request; software readiness
alone is necessary but insufficient.

## Unchanged Source and Private References

The original 3,261,648-byte 520-muscle XML was hashed against the existing
pinned source SHA-256 `a55c64341680551fb5a41be254bdfdb3b2be0ac789336ea44902b22fc9a83913`.
Frozen private driver and iron capture providers were checked against their
private manifest, converted to owned diagnostic TRCs and verified for exact
original binary observation clocks and validity masks. Driver retained 654
frames, 38 channels and 24,135 valid samples; iron retained 657 frames, 38
channels and 24,219 valid samples. The private aggregate reports have the
same seven blockers: unavailable source-valid guess, registration, marker
placements, passive policy, nonmuscle-assistance policy, native-constraint
policy and complete prepared native state. No optimizer was constructed and
no raw private marker arrays, identities or file paths are included here.

The diagnostic intentionally supplies no invented state/control bounds or
capture-to-model marker aliases. Earlier independent F07 observations also
found 29 nonmuscle actuators, 54 locked coordinates, two couplers, unresolved
muscle-path/anatomy and no qualified own-contact/grip model. The present
software route must remain blocked until reviewed placements, physiological
limits, source closure, native initialization/replay and private calibration
protocol exist. It does not close F07, F08 or the six-engine/17-row denominator.

## Continuation and Limits

1. Resolve the reviewed model/reference correspondence and passive/contact
   policies without altering original muscle law or hiding assistive actuation.
2. Supply a complete model-valid initial named state, scientific bounds and
   native guess for the unchanged source; retain discrete/numerical-state
   limits of cold-start replay explicitly.
3. Run a source-admissible full-horizon native solve and independent exact
   excitation replay with the original driver/iron clocks, masks and all
   required markers. Compare source-level execution and refined integration
   on common physical times before any qualification decision.
4. Keep different control policies' applied-input digests separate. Require
   exact channel order, knots, interpolation and injection between each
   policy's execution and its own independent replay.

The privately staged diagnostic has source/clock/mask receipts and its own
local reproduction script. Its raw TRCs are deliberately outside this public
repository and are not a public artifact or approved capture release.
