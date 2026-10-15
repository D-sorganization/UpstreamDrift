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

## Choice of Method

The objective is linear in its weights and the inner problem is convex with
weight-independent constraints. A **bilevel** fit (outer search over weights,
inner `allocate_trajectory`) is non-convex, needs one allocator solve per
evaluation, and its failure modes (local minima, solver `NUMERICAL` statuses
at the allocator's fixed 1e-11 tolerance, which occurred here) are hard to
separate from non-identifiability. The **KKT-residual** estimator turns the
question into a linear system `G a = 0`, whose solution set, rank and
near-null directions are exact statements about identifiability, and which
recovers the generating weights to machine precision in the noise-free case.
Its price is that it is an equation-error estimator (biased under noise,
see below) and cannot use frames where a constraint is binding. The bilevel
check is kept as the **scoring** step: every fitted weight set is replayed
through `allocate_trajectory`, so the held-out metric never depends on the
estimator's algebra.

Implementation: `src/shared/python/loop_nullspace/inverse_allocation.py`. No new
solver is written; the module calls `allocate_trajectory` for replay and
`solve_socp` for two small convex programs (the simplex fit and the
interior-margin test).

## Identifiability

Scale invariance: `J(c a) = c J(a)`, so only ratios of weights can be
identified; the module fixes the gauge by `sum u = 1` and reports ratios.
Beyond that gauge the module reports, never silently resolves:

* a stationarity column that is numerically zero (the data carry no
  information on that weight) is listed in `unidentified_features` and its
  weight is `NaN`;
* directions along which the fit criterion is flat on the normalised simplex
  are returned as `null_directions` (weight coordinates). Example: a reference
  of zero makes the reference gradient equal the effort gradient, and the
  reported direction is exactly `(1, 0, -1)` (effort versus reference), while
  their sum relative to the rate weight remains identified.

Fixture lesson: the shared test trajectory (one torque vector scaled in time,
fixed `N`) makes the effort optimum a fixed linear projection that commutes
with time differences, so the rate weight has no effect on the optimum at all.
All stationarity columns vanish and the module reports every weight as
unidentified. The synthetic fixture used for recovery therefore varies the
torque direction and `N` in time. The same mechanism is a warning for real
data: weights are identifiable only to the extent that the family changes
shape from frame to frame.

Observations must be interior points of the feasible torque set. For each frame
the module solves a small SOCP for the largest uniform slack over all
torque-preserving choices of the gauge coordinates, and drops frames with
slack at most 1e-6 (binding) or below minus 1e-6 (the observation violates the
declared hypotheses). Dropped frames are counted in the result.

## Synthetic Recovery

Data generated by `allocate_trajectory` with known weights on the planar golfer
fixture (`tests/unit/loop_nullspace/test_inverse_allocation.py`, 22 tests).
Measured relative errors of the recovered weight ratios:

| Case | Error in recovered ratios | `rho_KKT` |
| ---- | ------------------------- | --------- |
| Effort and rate (0.7, 2.5) | 2e-16 | 8e-14 |
| Effort, rate and reference (0.7, 2.5, 1.3) | 6e-14 and 8e-13 | 2e-13 |
| Effort, rate and gross power (1.0, 0.5, 0.4), 3 of 8 smooth frames | 1e-10 | 7e-12 |
| Scaling all true weights by 5 | same ratios; data identical to 1e-6 | |
| Relative observation noise 1e-4 | 2e-4 | 0.02 |
| Relative observation noise 1e-3 | 6.3e-2 | 0.19 |

Tests also cover finite-difference checks of the four term gradients, the
dropping of binding and infeasible frames, collinear terms (reported, not
fixed), a zero column, the degenerate fixture above, the seeded bootstrap and
the preconditions. The noise rows show the estimator's bias: 1e-3 relative
noise already moves the rate to effort ratio by 6 %.

## Results

Measured by `tests/integration/test_loop_nullspace_inverse_allocation.py` on
host Brick (4 cores, Python 3.12) with `LOOP_NULLSPACE_INVERSE_JSON` set.
Everything here follows the pre-registered protocol unless a subsection says
"exploratory".

### Observation Model

The record torque of the 35 actuated coordinates is projected onto each frame's
torque family `T0 + N_a z`. In the capacity-scaled norm the relative residual
of that projection is large everywhere:

| Set of frames | Median residual | Frames with residual at most 0.10 |
| ------------- | --------------- | --------------------------------- |
| Fit window (41) | 0.69 | 0 |
| Held-out H1 (38) | 0.77 | 0 |
| Held-out H2 (31) | 0.50 | 0 |
| All 199 decomposable benchmark frames (exploratory) | 0.81 (minimum 0.44) | 0 |

(The figures in the disclosure above, median 0.31, used the unscaled norm;
the scaled norm used for every other number is the one defined in "Metric".)
**No frame of the record has a torque that is a member of its own rigid-contact
torque family**, so the recorded torques are not reproducible by the model that
the allocator optimises over. In addition, the projected torque violates the
declared hypotheses (ASSUMED capacity prior or friction cones) in 15 of 41
fit frames, 19 of 38 H1 frames and 10 of 31 H2 frames. Candidate causes, not
separated here: compliant versus rigid contact, passive joint forces and
limits inside the simulation plant that the torque-only balance omits, and the
closure projection of the state. This is the first result: the inverse
problem is posed on an observation that the forward model cannot represent.

### Primary Fit M1 (Effort and Rate)

| Quantity | Value |
| -------- | ----- |
| Frames used / infeasible / binding | 26 / 15 / 0 (250 stationarity rows) |
| Fitted `a_r / a_e` | 1.1e-4 |
| `rho_KKT` | 0.72 |
| Singular values of the unit-column matrix | 1.017, 0.983 |
| Curvature of the criterion on the simplex (relative) | 0.93 |
| 90 % block-bootstrap interval of `log10(a_r / a_e)` | [-4.02, -3.84], width 0.18 |

The fit is sharply determined (the bootstrap interval is narrow) but
**inadequate**: the two gradient columns are almost orthogonal after
normalisation (singular values near 1), so no positive combination makes the
gradient small, and the minimiser merely puts almost all weight on the term
with the smaller gradient norm. A precise number that fits badly is not a
finding about the golfer.

### Held-Out Evaluation

Relative capacity-scaled error of the replayed allocation against the projected
record torque (the quantity the pre-registration defines); in parentheses the
error against the raw record torque (exploratory).

| Window | Fitted | Default (1, 0.1) | Effort only |
| ------ | ------ | ---------------- | ----------- |
| Fit window (in sample) | 0.605 (0.807) | not evaluable | 0.605 (0.807) |
| H1, 38 frames | 0.696 (0.994) | not evaluable | 0.696 (0.994) |
| H2, 31 frames | 0.735 (0.722) | not evaluable | 0.736 (0.723) |

"Not evaluable": `allocate_trajectory` with the default weights returns
`NUMERICAL: AlmostSolved` on every one of these windows at its fixed solver
tolerance and therefore no torque; the tolerance is not loosened. The fitted
weights are practically effort-only, and the fitted and effort-only errors agree
to three digits. Errors of 0.6 to 0.7 mean the allocation explains little of
the projected record torque even in sample.

### Secondary Fit M2 (Adding Gross Power)

Not informative. With `c+ = c- = 1`, 25 of the 26 interior frames sit at a
gross-power kink (a joint with nonzero velocity and a projected torque of
zero), leaving one frame (`rho_KKT` 0.51, weights not interpretable). Reported
for completeness; it plays no part in the verdict.

### Gates

| Gate | Outcome | Evidence |
| ---- | ------- | -------- |
| G1, adequacy (`rho_KKT <= 0.10`) | **Failed** | 0.72 |
| G2, identifiability | Passed as registered | bootstrap width 0.18 at most 1.0; see note |
| G3, held-out prediction | Failed | default replay not evaluable (fail closed); fitted equals effort-only to 3 digits, no gain |

**Verdict (pre-registered rule): the effort-plus-rate objective family is
rejected by these data. Nothing here supports the claim that fitted weights
generalise, and nothing identifies the golfer's weights.** G2's first criterion
is vacuous with two columns (the second-smallest of two singular values is the
largest), so G2 rests on the bootstrap interval and on the absence of
unidentified features; the curvature measure in the table is the more
informative identifiability diagnostic and also shows no degeneracy. G1 is the
binding failure.

What the failure does and does not say. It does not say that golfers have no
effort or smoothness cost, and it does not rank the default weights: the
default replay did not converge. It says that stationarity of the record
torques under any non-negative effort and rate weights fails by a wide margin,
and that this confounds two causes that these data cannot separate: a wrong
objective, and a record whose torques are not in the allocator's feasible
family at all (previous subsection). The second cause is sufficient to explain
the failure.

## Deviations From the Pre-Registration

None of these changes the verdict, which is fixed by G1.

* The disclosure of residuals before registration quoted the unscaled norm
  (median 0.31); the scaled norm used throughout is stricter (median 0.81 over
  all frames). Both were computed before any fit.
* G3 was implemented fail-closed: if a replay returns no torque (status not
  OPTIMAL) the comparison fails. The registration did not state this case.
* The "relative slack at most 1e-6" test was implemented as the largest uniform
  slack over torque-preserving gauge coordinates (a gauge-aware version of the
  same idea), threshold 1e-6 on row-normalised slack.
* Exploratory additions, labelled as such: the all-frames residual table, the
  raw-record error in parentheses and the secondary gross-power fit.

## Limitations and Non-Claims

* **One swing record, no measured torques.** The record's torques are the
  input of a compliant-contact simulation. Held-out windows are other time
  windows of the same swing; they share the model, the capacity prior and the
  controller. No statement about other swings, subjects or models is made or
  possible. The one-swing split is optimistic for any claim beyond this record.
* Weights fitted to inverse-dynamics or simulation torques do not validate hand
  loads, physiological effort or EMG, and the allocation remains a
  hypothesis-conditioned representative (NSX-23 non-claim).
* The capacity prior is ASSUMED for 23 of 35 coordinates; both the scale `S`
  and the feasibility of observations depend on it.
* The KKT-residual estimator is biased under observation noise (6 % at 1e-3
  relative noise in synthetic data) and ignores frames with binding
  constraints; with constraints binding on most frames of a real swing (not the
  case here only because the observation is infeasible instead) the usable data
  would shrink further.
* The allocator's fixed solver tolerance returned `NUMERICAL` for the default
  weights; the replay metric is undefined where that happens.
* No design-manual or scientific qualification is implied; this is software
  evidence only.

## Recommended Follow-Up

1. Establish why recorded torques fall outside the torque family (compliant
   versus rigid contact, passive joint forces) before any weight fit on this
   record; a forward-dynamics-consistent observation (the family projection
   residual at most 0.10 on a contiguous window) is a precondition for
   re-running this protocol unchanged.
2. Re-run the registered protocol on several swings and, ideally, measured
   torques or an instrumented-grip record (NSX-05, deferred); only then is a
   cross-swing held-out claim possible.
3. Investigate the `AlmostSolved` statuses of the default weights on windows of
   this record (the 82-frame acceptance window converges, sub-windows do not).

## Reproduction

```bash
export QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl
python3 -m pytest tests/unit/loop_nullspace/test_inverse_allocation.py -n 0 -q
LOOP_NULLSPACE_INVERSE_JSON=/tmp/inverse.json python3 -m pytest \
  tests/integration/test_loop_nullspace_inverse_allocation.py -n 0 \
  -m "integration or slow or unit" -q   # about 45 s
```
