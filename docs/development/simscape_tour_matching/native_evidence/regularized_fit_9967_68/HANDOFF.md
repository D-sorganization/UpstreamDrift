# Budgeted Expanded Sextic Trial 68 — Terminal Gate Failure

Run68 terminated with exit 1 after 425.444 s and nine forward evaluations:
`Sensitivity primal marker replay exceeds numerical agreement bound`.
Eight preceding Jacobian checks passed, each using 82,871–82,967 sensitivity
evaluations within its 100,000 cap. The terminal exception is an agreement
failure, not a budget failure. No retry, fallback, threshold change or
optimizer return occurred. The failed comparison's exact discrepancy is not
provided by the existing exception and cannot be inferred from passing checks.

Last unqualified candidate:
`c8885ace0b1021dea6e2fbd824286fd37eeff8cb91498147d2d585cb64bed039`.
Its preceding forward metrics are whole RMS 28.138 mm, early RMS 10.859 mm,
terminal RMS 65.565 mm and terminal club-cluster RMS 30.120 mm, over only
0–0.85 s. Reported near-bound count is four. Compared with restart62's
28.635 mm whole / 66.713 mm terminal errors, the improvement is modest and
does not qualify the failed derivative or candidate. `best-unqualified.json`
preserves this checkpoint; no returned-candidate exists. Run62 remains the
latest cleanly returned exploratory candidate, itself numerically rejected.

## Exact Configuration

Original model, baseline candidate19, capture and initial state are preserved,
with explicit restart62. Arguments: `--shaping bernstein456 --max-nfev 10
--analytic-jacobian --effort-penalty-weight 0.01 --max-step 0.000125
--max-sensitivity-evaluations 100000`. Default scales/amplitude/bounds remain
unchanged. This exposes 81 B4/B5/B6 controls within one global degree-six
polynomial. Sensitivity cap applies per augmented integration, not to all
forward/audit work or total optimizer work. Exact launch vector/environment,
source/input hashes and PID/terminal receipt are in `launch.json`.

Runtime `/home/dieterolson/native-regularized-fit-9967-68` clones immutable65
and overlays the exact updated runner and its tests. Qualification: 32 focused
tests passed in 0.83 s, including 13 runner-control tests and nine real Pinocchio
tests; `--help` import passed and exposes both new flags. The four unknown
pytest-unit-mark warnings reflect the isolated runtime configuration. Inherited
outer namespace scaffolding remains as previously documented; no physics or
numerical provider was shimmed. Runtime68 remains unchanged after launch.

## Evidence and Next Decision

`runtime-receipt.json`, `overlay-hashes.json`, `source-overlay.zip` and
`runtime-source.zip` preserve qualification and source. Runtime archive SHA256:
`26a76f7df628348e58464ef7636cb81e1d2bff284d520057d6defedf8d4d2fef`.
`summary.json` preserves all nine forward checkpoints and eight successful
Jacobian receipts. `raw-run.zip` contains every raw output, exact traceback,
launch/log files, original inputs and runtime source. Raw archive SHA256:
`93e4d44d3d78162fc1d7115f995344343d2773ee4418bb2135b0e0e8651368b5`.
All overlay/input/output hashes and qualification-versus-execution source
identity were verified after download. Force-add ignored ZIPs when committing.

The smaller step that passed fixed-candidate diagnostic65 is not sufficient
for every subsequent candidate. Root owns the next reliability decision;
do not resume this checkpoint by suppressing its failed gate. All processes
from this assignment are terminal. No production edits, commits or further
experiments were made after68.
