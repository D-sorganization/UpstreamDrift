# Regularized Native Continuation 62

Run62 terminated cleanly (exit 0) after 468.965 s. It remains numerically
rejected and the optimizer did not converge within max-nfev 20. Whole marker
RMS is 28.635 mm, early RMS 10.825 mm, terminal RMS 66.713 mm and terminal
club-cluster RMS 31.062 mm. The original run19 values were 30.791, 10.667,
99.989 and 56.557 mm respectively. This is only the 0–0.85 s prefix.

The exact returned candidate is `returned-candidate.json` beside this file,
also at `/mnt/c/Users/diete/native-regularized-fit-9967-62/returned-candidate.json`.
Canonical SHA256:
`7467c5d82817251858255bdf0e560a712f0d20a366ee0094c364a6c14481e1d5`.
`returned.json` and `summary.json` preserve complete metrics and parameters.
The reported near-bound count is four; inspect all final parameters before
interpreting that narrow numerical proximity threshold as remaining freedom.

## Runtime and Scope

Run62 uses immutable runtime61, original baseline candidate19 and explicit
restart61, with `--shaping sixth --max-nfev 20 --analytic-jacobian
--effort-penalty-weight 0.01`. Bounds and scales were unchanged. Exact command,
environment, source/input hashes and terminal PID receipt are in `launch.json`.
Its 22 forward records include output audits; max-nfev is not an RHS budget.
No state resets or numerical provider changes were introduced. Runtime61's
outer namespace qualification limits remain those documented in its handoff.

The authorized next bounded trial63 uses the same original baseline/runtime,
restart62, `bernstein456` and max-nfev 5. It frees B4/B5 while retaining a single
degree-six polynomial and the same bounds/scales. It is a separate experiment;
run62 does not establish that extension will converge or pass acceptance.

## Visual Evidence and Archive

`marker-comparison.png` plots measured forward predictions against the capture.
Top-left is RMS of Euclidean XYZ errors over valid markers at each time.
Selected marker errors and world X–Z trajectories occupy the other panels.
`sampled-markers.npz` preserves all predicted and target XYZ values, labels,
time and validity. Invalid or nonfinite capture markers are excluded from
metrics. The replay metrics exactly reproduce recorded run19/run62 metrics.

`visualize_replay.py` ran both actual replays on ControlTower and saved the NPZ,
then failed to import matplotlib. No simulation was repeated: `render_samples.py`
rendered the saved arrays locally with matplotlib 3.10.3. The PNG was visually
inspected. `visual-receipt.json` preserves this limitation and source/input,
sample and PNG hashes. It is not evidence of successful remote plotting.

`raw-run.zip` preserves every remote run output, launch/log files, executed
visual source, sampled NPZ, inputs and complete runtime source archive. Its
SHA256 is `3a4ecacc2b4b618d5b78cebc7113ac65e52561098b9e41b0fd801f38bf2e83f4`.
The archive and every output hash were verified after download. Preserve raw
bytes there if formatting convenience JSON. Force-add ignored ZIP, NPZ and
PNG files when committing. No production source or thresholds were changed.
