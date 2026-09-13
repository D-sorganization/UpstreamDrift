# Grouped Tolerance Diagnostic 71 — Terminal Agreement Failure

One fixed-candidate call on frozen runtime70 failed the unchanged agreement
gate: maximum absolute discrepancy 1.02308684e-7 m at 0.85 s, WaistRBack axis
1 (Y), against limit 1e-7 m. Run70's discrepancy was 1.02308673e-7 m.
Moderately tightening the augmented tolerances did not materially change the
measured discrepancy. The call took 34.025 s; launch-to-terminal took 38.104 s
with exit 1. This was not an evaluation-budget exception. Failure-path
evaluation counts are not exposed by the existing provider; none are invented.

Exact candidate:
`c8885ace0b1021dea6e2fbd824286fd37eeff8cb91498147d2d585cb64bed039`.
Original model, initial state and all 307 capture times remain unchanged.
Arguments: first_control 4, separate_error_control true, rtol 3e-11, atol
3e-13, max_step 0.000125, max_sensitivity_evaluations 150000. Independent
primal replay and all closure/agreement gates remain unchanged. No validated
result/Jacobian arrays were returned. No optimization or automatic retry ran.

`diagnose_sensitivity.py`, `receipt.json`, `summary.json` and `raw-run.zip`
preserve exact driver, inputs, runtime source, terminal traceback and hashes.
Raw archive SHA256:
`f656a09d04fabad97bc2f2b9653c987cb8361eca7b822fa8c1d51646be545c80`.
Every input/output hash and identity with qualified frozen70 were checked.
Runtime70's 49 passing tests and package-scaffolding limitations remain the
qualification basis. Force-add the ZIP when committing.

The unchanged discrepancy alone does not isolate whether step size or the
independent primal contributes most. Root authorized a separately recorded
single half-step diagnostic72, restoring70 tolerances, to test that question.
No numerical providers or thresholds were altered here. Output
`/mnt/c/Users/diete/native-sensitivity-9967-71` is terminal and immutable.
