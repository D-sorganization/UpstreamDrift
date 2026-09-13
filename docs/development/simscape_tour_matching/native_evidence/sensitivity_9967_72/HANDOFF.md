# Grouped Half-Step Diagnostic 72 — Agreement Passed

One fixed-candidate check passed the unchanged marker agreement gate with
maximum absolute difference **2.506897942e-8 m**. It used **164,339** sensitivity
evaluations within its explicit 200,000 cap. Sensitivity integration took
53.916 s, call/save 71.554 s, and launch-to-terminal 79.522 s (exit 0).

The exact run68 candidate remains
`c8885ace0b1021dea6e2fbd824286fd37eeff8cb91498147d2d585cb64bed039`.
Arguments: first_control 4, separate_error_control true, rtol 1e-10, atol
1e-12, max_step 0.0000625, max_sensitivity_evaluations 200000. Original model,
initial state, capture and 307 sample times over 0–0.85 s are unchanged.
Frozen runtime70 and its independent primal replay are unchanged; the smaller
max_step applies to both primal and augmented integration. The budget counts
augmented linearization calls, not all primal/output work.

Relative to run70, only the step limit changes numerical integration behavior;
the increased cap remains nonbinding. The disagreement drops by about 4.08x.
This is evidence that step refinement helps this candidate, not a proof of
global convergence, derivative-column accuracy or accepted swing fit. All
existing closure and agreement thresholds remain in force.

`diagnose_sensitivity.py`, `receipt.json`, `summary.json` and `raw-run.zip`
preserve exact arguments, launch/environment, source/input hashes, terminal
outcome and complete sampled state/marker Jacobians and primal marker arrays.
Archive SHA256 is recorded in `summary.json`. All archived input/output hashes
and identity with qualified runtime70 were verified after download. Runtime70's
49 tests and documented namespace limitations remain the qualification basis.
Force-add the ignored ZIP when committing. Remote output is
`/mnt/c/Users/diete/native-sensitivity-9967-72`; preserve it unchanged.

Root has separately authorized bounded fit73 using this candidate as restart,
the same original baseline19 and these numerical settings. That fit is a new
experiment; diagnostic72 does not predict acceptance of later candidates.
No production edits or optimizer runs were made as part of72.
