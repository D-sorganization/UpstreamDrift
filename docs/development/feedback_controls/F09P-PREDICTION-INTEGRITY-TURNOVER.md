# Prediction Integrity Consolidation Turnover — F09p #12055

## Change and Contract

Starts at F09o `98784e8374f3adf31f3b22d6bb9e54c0bfa3d081`, with admitted
Tools `86d0f28b1cc5acf61185e07e320c816c2d005512`. Reuses the existing direct
native execution validator, complete provider verification and guarded SDK
promotion/canonical replay. No second stepping loop, optimizer or replay format.

Preparation previously verified owned identity, built the bundle, and verified
provider/native interpretation again. Full owned identity now follows all bundle
preparation. Its provider check is reused by the native execution validator.
Entry and exit checks remain. Every prediction still reads complete provider
bytes, including repeated commands; no timestamp or cross-invocation cache.

Final preparation also verifies complete live preservation before owned
execution. Handle replacement during bundle construction previously reached two
native steps before exit rejection; now it must fail before any step. Tests
cover model, source, handles, callbacks and live-state preparation mutation.
The numerical result stays provisional. Independent promotion/replay and all
physical admission obligations remain unchanged.

## TDD and Controlled Measurement

The local OpenSim/MuJoCo 3.8 environment skipped all five initial cases because
this lane requires actual MyoSuite 3.0/MuJoCo 3.6. Those skips are not evidence.
The actual SDK RED receipt has two failures and three passes: four provider
checks instead of three, and the late handle-mutation rejection. Retain
`f09p-boundaries-red-native.xml` in the workspace planning evidence folder.
The final test also covers live-state mutation after preparation.

The existing owned DeskComputer SDK environment ran baseline/candidate/baseline/
candidate in fresh processes, switching only the native-search source inside
its owned stage. Each run made fourteen two-step predictions; the first two
were excluded from timing statistics. Complete native-state hashes agree for
all fourteen inputs across all four runs. Before/after source manifests match
within each run; only native-search source differs across baseline/candidate.

| Run         | Median Prediction | Provider Checks | Native Library Bytes Read |
| ----------- | ----------------: | --------------: | ------------------------: |
| Baseline 1  |          97.91 ms |              56 |               623,816,704 |
| Candidate 1 |          86.51 ms |              42 |               467,862,528 |
| Baseline 2  |         108.58 ms |              56 |               623,816,704 |
| Candidate 2 |          67.61 ms |              42 |               467,862,528 |

The exact 25 percent byte-work reduction is supported directly. Wall-clock
observations are machine-load dependent, with only two repetitions; do not
claim statistical production-speed improvement or hard real-time execution.
Receipts `f09p-benchmark-{baseline,candidate}-{1,2}.json`, retained benchmark
scripts and `inspect_f09p_benchmarks.py` permit independent evidence inspection.
No private captures were used by this two-channel fixture.

## Native Regression and Publication

The full ten-module actual MyoSuite 3.0/MuJoCo 3.6 campaign passed **176 tests**
with zero failures, errors or skips in 107.86 seconds (JUnit 107.652 s).
All 21 declared executed source/test files and the complete 8,122-file tracked
Tools archive matched exactly before and after. This includes original
production driver/iron twenty-step forecasting, scoring, guarded promotion
and independent replay; those short probes do not qualify full captures.
Receipts are `f09p-native-full-campaign.xml` and matching
`f09p-native-source-{before,after}.json`. Local normal hooks and GitHub CI
remain separate. No SDK was installed or modified; no new environment or
worktree was created, and private data and existing videos were unchanged.

Ruff, the normal mypy hook, architecture/file-size budgets, document titles,
manual governance, monolith freshness, LoD and DRY no-growth checks pass without
new exceptions. Context check and all twelve navigation tasks pass. The fleet
pre-PR runner passes all five gates with `MYPYPATH` explicitly set to the
repository root: its default root-plus-src search initially reported duplicate
`engines`/`src.engines` module names. Its local affected tests skipped six cases
because MyoSuite is absent; actual execution evidence is the separate 176-test
SDK campaign, not those skips. Normal commit/push and GitHub results must be
recorded separately in the publication receipt.

## Reproduction and Remaining Work

Use the F09o native environment/resource variables, adding
`tests/unit/engines/myosuite/test_prediction_integrity_boundaries.py` to its
nine-module campaign. Preserve original receipts; use new evidence filenames.
The benchmark controller alternates retained baseline/candidate source only in
the owned stage and restores the candidate. Reusing its output names is refused.
Compare full native-state hashes and source manifests before interpreting time.

Next advance measured full-horizon private-reference fitting, model-specific
hard margins and qualified contacts/grip, with derivative strategy chosen from
native evidence. All seventeen production model/variant rows, six ecosystems
and the ultimate muscular OpenSim saved-excitation replay endpoint remain open.
Preserve Astra's prepared-state/domain/conditioning, source-coordinate/branch
and anatomy gates. This change adds no capture fit or physiological admission.
