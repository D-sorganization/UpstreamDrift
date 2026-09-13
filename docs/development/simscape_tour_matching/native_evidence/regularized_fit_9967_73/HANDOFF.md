# Grouped Expanded Sextic Trial 73 — Returned, Still Rejected

Run73 terminated cleanly (exit 0) after 989.414 s. It reached max-nfev 10;
optimizer convergence and numerical acceptance are both false. All ten
sensitivity checks passed, using 164,267–164,315 evaluations each within the
explicit 200,000 cap. Twelve forward records include final replay audits.

Returned errors over only 0–0.85 s: whole marker RMS **28.105 mm**, early RMS
**10.860 mm**, terminal RMS **65.398 mm**, terminal club RMS **30.041 mm**.
Reported near-bound count is seven. Compared with the supplied run68 checkpoint
(28.138 mm whole / 65.565 mm terminal), this is only a small improvement.
Do not interpret successful numerical checks as a reasonable full-swing match.

The exact candidate is `returned-candidate.json` beside this file and at
`/mnt/c/Users/diete/native-regularized-fit-9967-73/returned-candidate.json`.
Canonical SHA256:
`786522cd5380a9f62602920b2cff6e6b404f7ceb99bfc6783f1f359b98a6457a`.
`returned.json` and `summary.json` preserve all metrics, parameters and checks.

## Exact Configuration and Restart Identity

Original model, capture and baseline candidate19 were retained. The supplied
restart was extracted from run68's checkpoint and its canonical hash verified
as `c8885ace0b1021dea6e2fbd824286fd37eeff8cb91498147d2d585cb64bed039`.
The existing runner recovers/reapplies Bernstein controls; its first actual
candidate hash is `6f0be375a9690b198d0d094272faa7bfe7d6a73cabbde1b79db24021022b92b7`.
Both identities and source checkpoint bytes are preserved. Initial metrics
closely reproduce the checkpoint, but do not claim byte-identical reconstruction.

Arguments: B4/B5/B6 shaping, max-nfev 10, analytic Jacobian, separate error
control, max_step 0.0000625, sensitivity cap 200000, penalty weight 0.01.
Default amplitude 10, bounds 0.8–1.2 and physical penalty scales are unchanged.
The original state is not reset. No threshold was weakened or automatic retry
performed. Exact command/environment and hashes are in `launch.json`.

Runtime `/home/dieterolson/native-regularized-fit-9967-73` clones70 and overlays
the exact committed d8fbfdaff runner, current native provider and runner tests.
Qualification passed 62 tests (49 numeric/real Pinocchio plus 13 runner-control)
and `--help`. Inherited namespace scaffolding remains; no numerical provider
was shimmed. Qualification and execution source hashes match exactly.

## Actual State and Visual Evidence

One authorized post-return scalar replay used rtol 1e-11, atol 1e-13 and
max_step 0.0000625. It took 13.385 s integration time / 164,231 integration
evaluations and exactly reproduced final whole/terminal RMS metrics.
`sampled-markers-state.npz` contains measured predictions, target XYZ, validity,
labels, clock, coordinate names and **native_state with shape (307,54)**:
27 actual q followed by 27 actual qd per sample. These are state values, not
sensitivity Jacobians. No qdd is saved.

`marker-comparison.png` uses valid-marker Euclidean RMS and selected marker
errors/world X–Z paths. It is explicitly labeled rejected/exploratory and
0–0.85 s. The run19 baseline is reused from the existing run62 visual's actual
sample arrays (max_step 0.00025); it was not reintegrated. The returned73 curve
uses the new half-step replay. Capture validity and finite XYZ mask all metrics.
`replay_returned.py`, `replay-launch.json`, `replay-receipt.json`,
`render_samples.py` and `visual-receipt.json` preserve exact sources, commands,
inputs, timings and hashes. Local matplotlib 3.10.3 rendered the PNG, which was
visually inspected. No second optimizer ran.

## Archive and Next Decision

`raw-run.zip` contains every raw remote output, the actual-state evidence,
launch/log files, inputs, original checkpoint, extracted restart and runtime source.
SHA256: `3764032e6ca8f749b01464125a7efb9a4627b962c2b93611b656c16e0b159fb8`.
`runtime-source.zip` SHA256:
`2a42bb15a891517330ee9f91507549021c8cf35c5ca6d3ed96274b6f2558e324`.
All input/output/archive hashes and overlay/execution identity were verified.
Force-add ignored ZIP, NPZ and PNG artifacts. Raw archived JSON preserves
original bytes if convenience copies are reformatted.

Root's next work is derivative/predicted-reduction and multiple-shooting
conditioning review, not an automatic increase in this run's evaluation count.
All processes from73 and its evidence replay are terminal. No production edits,
commits or further fitting were performed by this execution agent.
