# Inverse Optimal Control of the Allocation Weights (NSX-71)

Research report for issue #12589, child of epic #12593. Status of this file in
git history: the section "Pre-Registration" was committed **before any weight
was fitted** and before any held-out error was computed; later sections were
added in later commits. `git log --follow` on this file shows the order.

## Question

The NSX-23 allocator (`allocate_trajectory`) picks one representative torque
history from the family of feasible histories by minimising

`J = sum_k w_k a_e |S^-1 tau_k|^2 + a_r sum_k |S^-1 (tau_{k+1} - tau_k)|^2 / dt_k
 + a_ref sum_k w_k |S^-1 (tau_k - tau_ref,k)|^2 + a_g (gross power)`.

The weights `a_*` are currently hand-chosen (the anthro acceptance run uses
`a_e = 1`, `a_r = 0.1`). Inverse optimal control asks the reverse question:
for observed torques, which weights make them optimal, are those weights
identifiable, and do weights fitted on some data predict other data?

## Pre-Registration

Written before fitting. Everything in this section is fixed; deviations are
reported in "Deviations From the Pre-Registration" and nowhere else.

### Disclosure of What Was Seen Before Registering

Before writing this section the author (a) read the allocation code and the
anthro support module, and (b) projected the record torques of the 199
decomposable benchmark frames onto each frame's torque family and looked at
the projection residuals (median relative residual 0.31, maximum 1.48) and at
the contiguous feasible runs. No weight was fitted, no allocation was run on
the anthro frames, and no prediction error was computed.

### Data

* **Only one anthro swing record exists**: `dynamics_record.npz` (606 frames,
  0.003 s apart). The benchmark strides it by 3 (0.009 s).
* **No measured joint torques exist.** The record's `tau` is the simulation
  input of a compliant-contact forward simulation, not a measurement. The
  fitted quantity is therefore "weights that explain simulation torques", and
  the Non-Claim of the issue applies: weights fitted to IK-derived or
  simulation-derived data do not validate hand loads or physiological effort.
* **Observed torque.** The record torque of the 35 actuated coordinates is
  projected, per frame, by least squares onto the affine torque family
  `tau = T0 + N_a z` of that frame's decomposition. The projection is the
  observation used by the fit and by the primary metric. The projection
  residual is reported as a descriptive statistic and is a limitation (the
  record was generated with compliant contact, the decomposition uses rigid
  contact).
* **Hypotheses.** Feasible sets are the NSX-60 anthro feasible sets used by the
  NSX-23 acceptance test (ASSUMED capacity prior plus friction cones).

### Split

The split is by record time, contiguous blocks, fixed now. "Run" means a
maximal contiguous stretch of benchmark frames with non-empty feasible sets
(the allocator requires a feasible set in every frame of a window).

| Block | Record frames | Role |
| ----- | ------------- | ---- |
| Run B, positions 0 to 40 (41 frames, record 96 to 216) | fit | weights are fitted here only |
| Run B, positions 41 to 43 | buffer, unused | removes rate-term coupling between fit and test |
| Run B, positions 44 to 81 (38 frames, record 228 to 339) | held-out window H1 | later part of the same swing phase |
| Run A (31 frames, record 0 to 90) | held-out window H2 | earlier part of the same swing |

**This is not a held-out swing.** All blocks come from one swing of one model.
The windows are adjacent in time and share the model, the capacity prior and
the controller that produced the record, so the held-out error is optimistic
for any claim beyond "this swing, other time windows". No generalisation
across swings, subjects or models is claimed or testable.

### Models Fitted

* **Primary (M1):** effort and rate terms. Scale invariance of the objective
  is fixed by reporting `rho = a_r / a_e` (weights are only identified up to a
  positive common factor).
* **Secondary (M2, not used for the verdict):** adds a gross-power term with
  `c+ = c- = 1` (per watt), reported for completeness.
* The reference term is not fitted to anthro data (no reference exists).

### Estimator

KKT-residual (stationarity) estimator. For the observed torque `tau*`, with
inactive constraints, optimality of `tau*` under weights `a` means
`sum_j a_j N_k^T grad_j(tau*) = 0` for every frame `k`. Each column `j` is the
stacked reduced gradient of one unit-weight term, so the residual is `G a`.
Frames at which any polyhedral row or cone is active at `tau*` (relative
slack at most 1e-6) are dropped from the stationarity rows and counted.
Columns are scaled to unit norm, then `min |G u|` is solved over the simplex
`u >= 0, sum u = 1`; weights are `a_j = u_j / |G_j|`. A bilevel fit was
rejected (see "Choice of Method"). Weights are always evaluated by replaying
`allocate_trajectory`.

### Metric

For a window, replay `allocate_trajectory` with the weights on the window's
feasible sets and report the relative error

`E = |S^-1 (tau_alloc - tau_obs)|_F / |S^-1 tau_obs|_F`

(`S` the capacity torque scale, `tau_obs` the projected record torque). Three
weight sets are compared on each held-out window: **fitted**, **default**
(`a_e = 1`, `a_r = 0.1`) and **effort-only** (`a_e = 1`, `a_r = 0`).

### Gates and Success Threshold

The verdict is "the objective family explains the data and its weights
generalise to held-out windows of this swing" only if **all** of G1 to G3 hold.

* **G1, adequacy.** On the fit window the relative stationarity residual
  `rho_KKT = |G~ u*|` (unit-norm columns, simplex weights, so the value lies
  in `[0, 1]`) is at most **0.10**. If G1 fails, the verdict is "the
  effort-plus-rate objective family is rejected by these data".
* **G2, identifiability.** The second-smallest singular value of the
  unit-column matrix `G~` is at least **1e-2** times the largest, and the
  90 % block-bootstrap interval (blocks of 8 frames, 200 resamples of the fit
  window, `numpy.random.default_rng(12589)`) of `log10(a_r / a_e)` is at most
  **1.0** wide. If G2 fails, the verdict is "weights are not identifiable
  from these data" regardless of G3.
* **G3, held-out prediction.** On **both** H1 and H2, `E_fitted <= 0.9 x
  E_default` (at least a 10 % reduction against the current hand-chosen
  weights) **and** `E_fitted <= 0.25`.

No other threshold, window, metric or model is used for the verdict. If an
analysis is added after seeing results it is labelled exploratory.

### Reproduction

`tests/integration/test_loop_nullspace_inverse_allocation.py` runs the
pre-registered protocol and writes the measured record when
`LOOP_NULLSPACE_INVERSE_JSON` names a file path. The constants above are
asserted against the module constants in the same test.

## Results

Added after the pre-registration commit.
