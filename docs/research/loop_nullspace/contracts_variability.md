# Per-Child Contract Appendix: Variability and Sensitivity Contracts

This file continues `contracts.md` with the variability, sensitivity and
inertial-ensemble contracts from NSX-80 through NSX-98 and NSX-62.

## NSX-80 Variability Sensitivity (#12709)

Specification only; no production module. Section numbers in
`loop_nullspace.tex` (Variability Sensitivity, `sec:variability`) are the
citable contract for every variability child (epic section 5A, V1 to V8):

| Section | Content | Golden keys |
|---|---|---|
| `sec:var-lin` | KKT linearization on the tangent bundle, reactions re-solved, `K_geom(λ) = Σ_r λ_r ∇²Φ_r`; `G_t N = 0` | `linearization`, `golfer_preload` |
| `sec:var-salt` | Event projection `Π` and saltation matrix `S`, `S f⁻ = f⁺` | `saltation` |
| `sec:var-cov` | Covariance recursion and adjoint form | `min_variance.model` |
| `sec:var-mechanisms` | M1 signal-dependent noise, M2 geometric preload, M3 muscle impedance (MyoSuite/OpenSim only); `OBJECTIVE_FLAT_INVARIANT` when none is declared | `min_variance`, `additive_noise_flat` |
| `sec:var-face` | Wrapped face-angle error, bias, SD, IQR, exceedance, miss probability; coefficient of variation banned for zero-centred outcomes | none |

Registry additions (`src/config/loop_nullspace_registry.json`): hypotheses
`H-NOISE-ADDITIVE`, `H-NOISE-SIGNAL-DEPENDENT`, `H-GEOMETRIC-PRELOAD`,
`H-DECLARED-JOINT-IMPEDANCE`, `H-SENSORIMOTOR-DELAY`, `H-TIMING-JITTER`,
`H-MUSCLE-SHORT-RANGE-STIFFNESS`; eight citations, all `VERIFIED` against
retrieved abstracts (Harris and Wolpert 1998, Jones et al. 2002, Todorov and
Jordan 2002, Scholz and Schoener 1999, Mueller and Sternad 2004, Burdet et al.
2001, Cusumano and Dingwell 2013, Cui et al. 2008). Partial-support caveats are
kept in each `claim_checked`: Scholz and Schoener (variance partition not in the
abstract), Burdet (impedance geometry, not co-contraction) and Cui (feline,
force-dependent rather than activation). Hypothesis statements claim only what
the abstract supports; impedance values and noise amplitudes are declared by the
user, not measured.

Preconditions for consumers: use the golden values as oracles, never as
inputs. Postconditions: the `z`-dependence of `F_t` appears only through
`λ(z)`; an additive-noise objective with no M2 has gradient exactly zero.
Non-claims: noise amplitudes are declared hypotheses, not measurements; no claim
that any golfer uses a minimum-variance allocation.

## NSX-87 MuJoCo and MyoSuite Derivatives (#12716)

`MujocoForceAdapter.constrained_acceleration_derivatives(q, v, tau)` (and the
MyoSuite subclass) drive the single NSX-84 `FiniteDifferenceDerivatives`
routine with the adapter's canonical `M`, `h`, `J_c`, `Jdot_v` through
`loop_nullspace/adapter_derivatives.py`. The derivatives are those of the rigid
hard-KKT solve with the reaction re-solved; capability is `PARTIAL` with
`derivative_method=finite_difference` (central differences along retractions,
step `1e-5`).

* MyoSuite refuses (`DerivativeCapabilityUnavailable`) and reports `NONE`
  unless `kkt_regularization == 0`.
* Muscle activation states are unsupported (`UNSUPPORTED_STATE_COMPONENTS`).
  NSX-94 adds quasi-static muscle stiffness on top of the coordinate
  derivatives (`apply_muscle_stiffness`); activation and fibre-length dynamics
  remain undifferentiated.
* Non-claim: `mjd_transitionFD` differentiates MuJoCo's soft stepping
  (solref/solimp compliance). It is a separately characterised registry entry
  and its sensitivities are not conformance evidence here.
* Pinocchio agreement (same specification, state and load; basis-independent
  operators `D P^T`): the bound is the registered NSX-56 Track N tolerance
  propagated through the differencing step, `max(1e-11, 1e-11, 1e-12, 1e-8) /
  1e-5 = 1e-3` (scaled). Measured: acceleration `7e-12`, reaction `1e-10`,
  `d a/d tau` `4e-12`, `d a/d xi` `9e-7`, `d a/d eta` `6e-5`, `d lambda/d xi`
  `1e-5`, `d lambda/d eta` `7e-5`. Agreement between engines is not physical
  validation.

## NSX-90 Variance Propagation (#12719)

Modules (all under `src/shared/python/loop_nullspace/`):

- `variance_propagation.py`: `propagate_covariance(dyn, p0, sigma)` (forward
  `P_{t+1} = A P Aᵀ + B Σ Bᵀ`), `adjoint_kernels(dyn, C)` ->
  `AdjointKernels` with `s_y_x0` (`S_{y,x0}`) and `s_y_tau(t)`
  (`S_{y,τ}(t)`), `adjoint_covariance`, `variance_budget` (by input class and
  step), `augment_persistent` (per-trial parameter states),
  `euler_dynamics` (`A = I + dt F`, `B = dt G`, any tangent dimension),
  `step_covariances` (from `DisturbanceSpec`, timestep consistent) and
  `event_outcome_rows` (`C Π`, NSX-83). `LinearizedDynamics` is reused from
  `estimation/mosaic/local_policy.py`.
- `nonlinearity_gate.py`: `assess_nonlinearity(rollout, directions, *,
  amplitude, gate_threshold)`. **`gate_threshold` has no default**: NSX-80
  specifies a gate but no number, so the caller supplies it.
  `LinearValidity` is `LINEAR_VALID`, `NOT_LINEAR_VALID` or `NOT_ASSESSED`.
- `variance_records.py`: `assess_variance(problem, declaration, *,
  outcome_units, gate=None)` returning typed `QuantityRecord`s. Kernels are
  `INVARIANT` unless M2 or M3 is declared; variance is `INVARIANT` with the
  noise hypothesis in `conditioning` when no mechanism is declared and
  `FAMILY_DEPENDENT` otherwise.

Preconditions: symmetric covariances, conformal shapes, finite inputs
(`ValueError`). Postconditions: adjoint and forward variance agree to 1e-12
(tests) and 1e-9 (runtime `ensure`); the budget parts sum to the total.
Non-claims: linear predictions do not cover saturation, slip or missed impacts.

## NSX-91 Paired Nonlinear Monte Carlo (#12720)

Modules (all under `src/shared/python/loop_nullspace/`):

- `paired_mc_spec.py`: `CampaignSpec` (disturbances, grid, trial counts, paired
  and hold-out seeds, `OutcomeChannel`s, optional `GeneratorSpec`),
  `TrialDraw` (each draw is a pure function of seed set, trial index and
  disturbance), `IntegratorSignature`, `require_independent_integrator`
  (R13: a replay sharing the generator's integrator name or step is an inverse
  crime and raises) and `compare_noise` (identical generator noise is circular
  and raises; different noise is flagged `noise_mismatched_to_generator`).
- `paired_monte_carlo.py`: `run_campaign(spec, strategies, *, batch_size=1)`
  and `run_seed_set`. Every `Strategy` sees the same draws (common random
  numbers); results are identical for any `batch_size`. Rollouts are never
  dropped: `TrialStatus` is `OK`, `FAILED`, `MISSED` (`TrialMiss`) or
  `NON_ADMISSIBLE` (`TrialNonAdmissible`); `tally` counts them and gives a
  Wilson interval on the failure rate. Programming errors propagate.
- `paired_statistics.py`: `summarise` (wrapped `OutcomeDispersion` for angle
  outcomes, `ScalarDispersion` otherwise, seeded bootstrap intervals),
  `compare_strategies` (paired differences resampled jointly; open-loop versus
  feedback needs `allow_cross_mode=True` and is recorded), `holdout_consistent`,
  `cross_check_linear` (MC SD versus the NSX-90 prediction; the gap is always
  reported, `AGREES` only if the gate ran and passed) and `sd_records`
  (`QuantityRecord`s, `UNVALIDATED`, mode label as a convention).
- `replay_campaign.py`: `replay_strategy(spec, name, ReplayCase, TorqueNoise,
  outcome_fn)` runs NSX-31 `replay()` under held torque noise; the strategy
  mode is the replay mode (R10), the integrator is RK4 at `dt / substeps`, and
  a terminated replay is a retained failure.

Preconditions raise `ValueError`/`TypeError`. Postconditions: every trial of
every strategy appears in its tally; draws are independent of the strategy,
order and batching; hold-out seeds differ from the paired seed.
Declared choices (the specification fixes none): percentile bootstrap
intervals, population SD, Wilson interval for failure rates, right-continuous
zero-order-hold noise inside RK4 stages. Non-claims: the MC is a statement about
the model under the declared noise; a failure rate is not a physical
probability; replay is not physical acceptance. NSX-95 comparisons must use this
runner.


## NSX-93 Declared Impedance and Delayed Feedback Layer (#12722)

Modules (all under `src/shared/python/loop_nullspace/`):

- `impedance_spec.py`: `JointImpedance` of `ImpedanceWindow`s,
  `H-DECLARED-JOINT-IMPEDANCE`. Each window is a symmetric PSD stiffness `K`
  [N m/rad] and damping `D` [N m s/rad] over `[t_start, t_end]` with a
  raised-cosine ramp; off-diagonal entries are biarticular coupling;
  overlapping windows add. `noise_scale = 1 + kappa_i K_ii(t)` is the declared
  noise-inflation penalty (`kappa` is an input with no source, a declared
  hypothesis). The impedance is declared, never inferred from net torque (V5).
- `feedback_runtime.py` and `feedback_layer.py`: `FeedbackLayer` is a
  run-scoped feedback law that `replay()` starts per run through
  `begin_run`. The impedance acts on the current error `K e + D de`; the
  `GainSchedule` feedback acts on the delayed error (`DelayedFeedback`,
  whole-step delay, Hermite history, initial state held before the start;
  `0 < delay < max_step` is rejected). The replay reports the layer's torque on
  the feedback line, separate from the allocated torque (R10).
- `minimum_intervention.py`: `minimum_intervention_design` is TVLQR with
  `Q = 0` and terminal weight `C_imp^T W_y C_imp` (`face_only_weights` for the
  face-only case). The gain annihilates `ker(W^(1/2) C Phi)`, so it corrects
  only outcome-relevant deviations; `task_irrelevant_gain_norm` reports the
  residual (1e-15 observed, tested to 1e-10). `joint_space_gains` maps the
  tangent gain to joint coordinates with `Kq = (K_xi - K_eta P^T O) P^T`,
  `Kv = K_eta P^T`; `tangent_velocity_offset` (NSX-84) supplies `(P, O)`.
- `layer_linear.py`: `LayerLinearization` folds the impedance into the
  continuous closed-loop `F - G_u K_imp` before the (zero-order-hold) matrix
  exponential and appends `m` delayed copies of the state for the feedback;
  `layer_variance` reuses NSX-90 `assess_variance` unchanged and puts the
  layer hypotheses on the `conditioning` axis (NSX-81). `frozen_delay_margin`
  and `LayerLinearization.delay_margins` give the first destabilising delay.
  `variance_ratio_record` is `UNAVAILABLE` (reason: co-contraction is not
  identifiable from net torque) unless the impedance is declared.
- `layer_accounting.py`: `account_layer` splits a replay into the allocated,
  impedance, feedback and total lines. Each carries its `FeedbackBudget`
  (peak, energy and absolute-work ratios), an `EffortCost` (`int tau^2 dt`,
  positive and negative joint work) and, for the impedance, the dissipated and
  peak elastic energy. Effort is bookkeeping of the declared model, not a
  physiological cost.
- `replay_campaign.py`: `TorqueNoise.scale` `(n_steps, n_channels)` multiplies
  the draw interval by interval (the noise-inflation penalty); `None` and ones
  are the same trial exactly.

Equations and frames. Tangent state `x = (xi, eta)`, `dq = P xi`,
`dv = P eta + O xi`. Actuated errors `e = S P xi`, `de = S P eta + S O xi`.
Impedance `K_imp = K S P + D S O` (position block), `D S P` (velocity block);
`u = -K_imp dx`. Delayed feedback `u_t = -K_fb,t dx_(t-m)` with
`z_t = (x_t, ..., x_(t-m))`, `P0_aug = T P0 T^T`, `T = [I; ...; I]`. Units: N m,
rad, s; outcomes in the units of their rows.

Tolerances (derived, none tuned). The oscillator oracle `var = S/(2 d k)`
differs from held noise by a sinc factor of order `(omega_n dt)^2 / 12`, so the
test tolerance is `(10 dt)^2 / 6`. The PD delay margin on a double integrator is
`tau_c = atan(kd w_c / kp) / w_c` with `w_c^2 = (a + sqrt(a^2 + 4 b)) / 2`,
`a = (kd/m)^2`, `b = (kp/m)^2`; the discrete margin is tested to +-1.5 dt and
the replay is stable at 0.7 and unstable at 1.3 times it. The Monte Carlo
cross-check uses the SD sampling half-width `Z99 / sqrt(2 (N - 1))` of NSX-91.
The zero-layer negative control is exact (`assert_array_equal`) in replay and
`rtol 1e-13` in the linear chain.

Failed or revised experiments. A threshold of 0.01 times the open-loop variance
for a stiff joint failed (ratio 0.15) and was replaced by the analytic base
variance; the first stationary-variance tolerance was loosened to the observed
`-(omega_n dt)^2/12` bias and then tightened to the derived bound; a Monte
Carlo of the delay margin over 100 steps was too slow (eigenvalues of a
`202`-square companion per step), so the default search is 20 steps.

Reproduce:
`python3 -m pytest tests/unit/loop_nullspace/test_impedance_spec.py
test_feedback_layer.py test_minimum_intervention.py test_layer_linear.py
test_layer_nonlinear.py test_layer_seam.py test_layer_accounting.py -n 0
-o addopts=""`.

Non-claims. Impedance, noise inflation and delay are declared hypotheses; no
stiffness is inferred from net torque and no co-contraction effect is claimed
without the declaration (`UNAVAILABLE`). Burdet 2001 supports selective control
of impedance geometry, not co-contraction. The feedback is uncompensated delayed
state feedback (no predictor). The frozen delay margin is a first-loss
diagnostic, not a time-varying stability proof. The tangent chain is `k = 1`
(NSX-84). `H-TIMING-JITTER` is not implemented here. Engine coverage is the
analytic point mass, pendulum and planar-golfer oracles through the provider
seam, not every engine adapter.


## NSX-95 Consistency Strategy Catalogue and Comparison Campaigns (#12724)

The owner's question is which strategies make a golfer more consistent, under
which disturbances and at what cost. The primary outcome is the face-angle
variance. The catalogue is data; the evaluators *compose* NSX-91/92/93/94/25 and
add no new dynamics. Modules (all under `src/shared/python/loop_nullspace/`):

- `strategy_spec.py`: frozen registry `CATALOGUE` of S1-S12. Each entry declares
  its mechanism, the mechanism ids that must be in force for any effect, a
  representation per model class (`rigid_joint_torque`, `muscle_driven`), its cost
  basis and its evaluator (dotted paths, resolved by `resolve_evaluators`).
  Availability is `IMPLEMENTED`, `NOT_IMPLEMENTED` or `UNAVAILABLE`; the
  constructor refuses an implemented entry without an evaluator, an unimplemented
  entry with one or without a reason, and an entry whose availability disagrees
  with its representations.
- `strategy_arms.py`: `ArmSpec` (a strategy at a declared setting),
  `ComparisonContext` (one swing in up to three representations) and `ArmResult`.
- `strategy_evaluators.py`, `strategy_layer_eval.py`: the adapters.
- `strategy_campaign.py`: `run_comparison` (matched-CHS, matched-effort,
  unconstrained fronts), ranking, failure rate and `variance_budget_table`.
- `strategy_stability.py`: `rank_under_priors` (reversals between priors) and
  `heldout_stability` (linear ranking against the NSX-91 paired and hold-out
  Monte Carlo).

Catalogue status.

| Id  | Strategy                    | Status          | Evaluator composed                                   |
| --- | --------------------------- | --------------- | ---------------------------------------------------- |
| S1  | Null-space redistribution   | implemented     | NSX-92 `minimise_face_variance`, M1 channel only     |
| S2  | Co-contraction / impedance  | implemented     | NSX-93 `layer_variance`; NSX-94 co-contraction curve |
| S3  | Distal stabilization        | implemented     | NSX-93 impedance window ending at impact             |
| S4  | Proximal-distal effort      | not implemented | no per-joint effort-share parameterisation           |
| S5  | Grip / hand-split preload   | implemented     | NSX-92, M2 (geometric preload) channel only          |
| S6  | Kinematic redundancy (UCM)  | not implemented | no UCM computation exists in the repository          |
| S7  | Delayed feedback            | implemented     | NSX-93 delayed feedback and frozen delay margin      |
| S8  | Speed-accuracy              | implemented     | NSX-92 `chs_variance_front`                          |
| S9  | Biarticular coupling        | not implemented | no biarticular comparison on the NSX-94 frame        |
| S10 | Shaft-flex phase matching   | unavailable     | rigid club has no shaft flexibility                  |
| S11 | Timing / sequencing         | not implemented | no timing-disturbance variance evaluator            |
| S12 | Address-state consistency   | not implemented | no address-consistency cost model                    |

A not-implemented or unavailable arm returns its reason and no numbers; the
campaign lists it under `not_attempted` and excludes it from the failure rate.

Declared choices (ambiguities resolved conservatively).

- One mechanism at a time. S1 keeps the signal-dependent noise (M1) and freezes
  the geometric preload at the nominal motion; S5 keeps the preload (M2) with
  additive noise. Effects are attributable and not additive when both channels
  act (that combined arm is the NSX-92 problem itself).
- Ranking basis. Arms are ordered by the face-angle variance ratio to the
  strategy-off baseline of their own representation; S8 cases use the reference
  case's baseline. Absolute variances are reported but not compared across
  representations. An arm whose ratio is not below one cannot win.
- Matched CHS. Clubhead speed is an input of each case. An arm outside
  `chs_rel_tol` of the reference (slower *or* faster, or of unknown speed) is
  excluded with `SPEED_MISMATCH`; the speed-for-consistency trade belongs to the
  unconstrained fronts. Matched effort adds a common cap *within one effort
  basis* (allocated torque, expected layer torque, muscle activation are separate
  lines, R10); an arm over the cap is excluded, never relaxed.
- Negative controls. A flat NSX-92 objective, a zero impedance, a zero feedback
  gain, a missing preload for S5 or a curve with no declared mechanism returns
  `NO_MECHANISM` with ratio exactly one and effect zero.
- Co-contraction baseline is the lowest level of the NSX-94 sweep (the least
  co-contraction that still produces the net torque), not zero co-contraction.
- Delay verdict. A feedback arm is `UNSTABLE` (a failure) when the frozen delay
  margin of any step is at or below the declared delay.
- Expected layer effort is `sum_t dt tr(G_t P_t G_t^T)` of the declared torque
  `u = -G z`; it is an accounting of the declared model, not a physiological cost.
- Priors are whole contexts. A reversal is a pair strictly ordered one way in one
  prior and the other way in another. Hold-out status is `CONFIRMED` only when the
  hold-out SD difference reproduces the order with an interval excluding zero,
  `REVERSED` when it contradicts it the same way, else `INCONCLUSIVE`.

Fixtures used in the tests (invented demonstrations, none a golfer measurement):
the NSX-92 pendulum for S1/S5/S8, the NSX-93 double integrator for S2/S3/S7 and
the NSX-94 antagonist pair for co-contraction. Observed results of those
fixtures are not claims about golfers; a late impedance window helped less than
full-horizon stiffness and needed *more* expected effort, because early
regulation keeps the feedback torque small.

Non-claims. Rankings are conditional on the declared priors, mechanisms and the
linear tangent model at the nominal motion; they compare mechanisms inside a
model, not golfers, and are not coaching advice until externally validated. The
ledger records software correctness only; scientific qualification stays in the
design-manual governance pathway.

## NSX-98 Sensitivity and Variance Conformance (#12727)

`tests/integration/cross_engine/test_loop_nullspace_derivative_conformance.py`
extends the NSX-56 harness from the snapshot arrays to the derivatives of the
rigid closed-loop solve. The registry check is
`closed_loop_derivative_conformance` in
`tests/integration/cross_engine/divergence_registry.yaml` (40 rows: four engine
pairs against the Pinocchio ANALYTIC reference, ten metrics each).

* **Track N** compares `d a/d tau`, `d a/d xi`, `d a/d eta` and the three
  `d lambda` blocks as basis-independent operators (`D P^T` for state blocks) in
  the Pinocchio coordinate order, at the NSX-56 projected states (frames 0 and
  400), one seeded actuated load and one closure tangent space.
* **Track S** (scientific comparison, not a measured noise level) compares the
  first-order covariance maps the sensitivities induce under a *declared* unit
  isotropic noise: `acc_load_variance = A S A^T`, `acc_state_variance = D_xi
  D_xi^T + D_eta D_eta^T` and the two reaction analogues. Face-angle kernels and
  minimum-variance allocations (NSX-89 to NSX-92) are not in this stack, so they
  are **not** compared here.
* **Tolerances are derived, then registered.** Load blocks: `kkt_bound` with a
  zero right-hand side times `sqrt(nv)`. State blocks of a finite-difference
  engine (MuJoCo, MyoSuite, OpenSim): the Track N differences propagated with
  `kkt_bound`, divided by the step, times `sqrt(k)` (NSX-88). State blocks of an
  exact engine (Drake): the spread of the engine-free NSX-84 finite-difference
  reference under step x0.5 and x2 (NSX-86). Covariances: `d (2|B| + d)`. The
  registry value is the larger bound over the two frames rounded up to two
  significant digits; a test requires every registered value to cover the
  per-run derived bound, and another that the measured difference stays inside it.
* **Capability matrix.** Pinocchio ANALYTIC and Drake AUTODIFF are `FULL`;
  MuJoCo, MyoSuite and OpenSim are `PARTIAL` (finite difference). Simscape has no
  native adapter and is declared `NONE` in `default_registry()` with its reason.
* **Planted errors fail.** A sign flip, a swap of the force and torque halves of
  the grip wrench, a two-name coordinate swap and a 0.1 percent load-block error
  are each caught. Derivatives whose scheme marks soft stepping
  (`mjd_transition`, `solref`, `solimp`) are refused: soft and hard dynamics are
  never mixed.
* **Measured versus bound** (scaled, worst of the two frames; the registered
  value is the bound). All measurements are inside their bounds; no divergence
  is registered. Load blocks: measured 2e-11 or better against bounds near 1e-9
  (`d a`) and 6e-7 (`d lambda`). State blocks, finite-difference engines:
  measured up to 7e-3 (`d a/d eta`, OpenSim, frame 0) against bounds up to 0.7
  (`d lambda/d eta` up to 1.8): the propagated bound is sound but loose by two
  to four orders for the state blocks. Drake: measured 7.5e-11 or better against
  resolution-limited bounds from below 1e-9 up to 3.4e-4.
* **Non-claims.** The MyoSuite rows inherit the MuJoCo derivation (the adapter
  drives the same routine) and were not measured on the authoring host, where
  the installed `myosuite` exposes no supported runtime layout (#12635); the
  `requires_myosuite` test enforces the per-run bound wherever it loads. The
  variance rows use declared, not measured, noise. Numerical conformance is not
  physical validation.
Anthro ensemble report (NSX-61 acceptance;
`tests/integration/test_loop_nullspace_anthro_ensemble.py`, `slow`). Members are
rebuilt with the NSX-60 foot-contact rows (`foot_contact_tol_m = 0`), decomposed,
given the unchanged `H-TORQUE-CAPACITY` prior and Coulomb cones, and enveloped
(35 joint torques and the six closure-wrench components, 41 quantities) with
`static_envelopes`; the nominal trajectory is member 0. Sampling is the declared
`H-KINEMATIC-ENSEMBLE-PRIOR` with seed 12586; the host is a shared 4-core, so
the sample is modest: three windows of three consecutive record frames (0 to 2,
150 to 152, 300 to 302), 12 samples each, at `sigma_q = 0.001` (1 mm, 1 mrad).

- Admission: all 36 samples are admitted (none rejected) at both sigmas.
- The hypothesis perturbs the root too, so a sample can lift a foot or leave
  fewer than three contacts. The root rows cannot then be balanced (the
  acceleration is held at its nominal value) and `decompose` refuses the sample,
  which is counted, not repaired. At `sigma_q = 0.001` the refused counts are 7,
  8 and 4 of 12 (19 of 36), leaving 5, 4 and 8 enveloped members. At the
  kinematic default `sigma_q = 0.02` only 3 of 12 samples in window 0 can be
  balanced, which is why the envelope report uses the smaller sigma.
- Union (interval hull, nominal included): the median width over the nominal
  width is 1.0008, 1.0005 and 1.0008 per window (maximum 1.016, 1.486 and
  1.026): a kinematic perturbation this small widens the envelope by well under
  1 percent for the typical quantity and by up to 49 percent for the worst cell.
  The union contains every member interval in every cell (asserted).
- Intersection: empty in 15 of 123 (frame, quantity) cells in windows 0 and 150
  (members with disjoint intervals; elsewhere the median width over nominal is
  0.9994 and 0.9992, minimum 0.14 and 0.81). In window 300 three member frames
  are INFEASIBLE under the capacity prior (the perturbation pushes the recorded
  effort outside it), so the intersection is empty in all 123 cells there; the
  union skips those members.
- Sign classes: members disagree on the sign class in 4 of 123 cells (3 frames)
  in window 0, none in window 150 and 9 cells (3 frames) in window 300; no decided
  nominal sign becomes INDETERMINATE in the union in any window (0 cells, 0
  frames). In total 13 cells in 6 of the 9 frames change class across members.
  An infeasible member frame has no class and is reported separately.
- Non-claims: the sampling distribution is a declared hypothesis, not measured
  uncertainty; the hull is not a confidence set; three short windows do not
  characterise the swing; the capacity prior is ASSUMED and unchanged.

## NSX-62 Inertial Parameter Ensemble (#12587)

`parameter_ensemble.py` propagates the inertial parameters that the data do not
identify, using the MOSAIC pseudo-inertia algebra (`estimation.mosaic.inertial`)
rather than a learner. Model: `Y(q, v, a) pi = tau`, `pi` the ten parameters per
body (Pinocchio layout `m, h, Ixx, Ixy, Iyy, Ixz, Iyz, Izz` about the joint
origin; the MOSAIC LMI layout swaps `Iyy` and `Ixz`, converted by
`swap_inertia_layout`).

- Identification. The unactuated root rows of `Y`, stacked over the window
  (`stacked_unactuated_regressor`; checked against `M a + h` of every snapshot),
  contain no unknown input once the ground reaction and the club wrench are
  observed, so their row space is what the window identifies. Columns are scaled
  by `(m, m r, m r^2)` (`r` the nominal radius of gyration about the joint
  origin). The rank is `min(SVD rank, column-pivoted QR rank)` at relative
  tolerance `1e-8`; the QR pivots are the surviving base columns. An
  identification claim holds only for combinations in this row space.
- Sampled parameters. Bodies with positive mass other than the club carrier
  (which holds the hand solids and the club, known from the specification) are
  sampled; massless bodies and the carrier stay fixed but remain in `Y pi`.
- Sampling (`H-INERTIAL-PARAMETER-ENSEMBLE-PRIOR`). Per body
  `J' = J^(1/2) (I + Delta) J^(1/2)`, `Delta` Gaussian with entries of standard
  deviation `sigma_rel`, orthogonally projected in the `Delta` metric onto the
  parameter changes `D N z` (`N` the null basis of the scaled stacked
  regressor), so the identified combinations are fixed by construction and the
  whitening keeps thin bodies inside the cone for small `sigma_rel`. A draw whose
  sampled bodies are not all positive semidefinite is rejected and counted
  (`pseudo_inertia_not_psd`, with the offending bodies); it is never projected
  onto the cone, because that would move the identified combinations.
- Envelopes. Each accepted sample is written to the model, its snapshots,
  decomposition, feasible sets (NSX-60 foot contacts, unchanged
  `H-TORQUE-CAPACITY`, Coulomb cones) and `static_envelopes` are rebuilt at the
  same `(q, v, a)`, and the union uses `envelope_union_intersection` of NSX-61.
  Sensitivity per quantity: union width over nominal width, largest bound shift
  (absolute and relative to the nominal width), sign-class disagreement and
  decided signs lost in the union. A sample whose root rows cannot be balanced is
  counted as refused, never repaired.

Anthro report (`tests/integration/test_loop_nullspace_anthro_parameter_ensemble.py`,
`slow`; seed 12587, `sigma_rel = 0.05`, 10 samples per window, three windows of
three consecutive record frames 0 to 2, 150 to 152 and 300 to 302, 41
quantities: 35 joint torques and the six closure-wrench components). Set
`LOOP_NULLSPACE_PARAMETER_ENSEMBLE_JSON` to write the record.

- Identification: 18 stacked root rows give rank 18 (SVD and QR agree) on 220
  sampled parameters (22 bodies), so 202 directions are unidentified. The rank
  is limited by the three-frame window; a longer window identifies more.
- Samples: 10 of 10 requested lie inside the LMI in every window, none refused as
  unbalanced, no nominal or member frame infeasible under the capacity prior. At
  `sigma_rel = 0.5` (no envelopes) 0 of 100 draws per window are inside the LMI,
  all counted as `pseudo_inertia_not_psd`.
- Identified combinations: the largest change of any scaled identified
  combination is `9.1e-17`, and the largest change of any unactuated row of
  `M a + h` is `4.5e-13 N m`, over all samples.
- Sensitivity: the actuated rows of `M a + h` change by up to 0.78, 0.67 and
  0.76 N m. The median over quantities of the union width over the nominal width
  is 1.0008, 1.0018 and 1.0014; over the 36 quantities with a nonzero nominal
  width the maximum is 1.010, 1.034 and 1.019, the largest relative bound shift
  1.4 to 2.0 percent of the nominal width and the largest absolute bound shift
  0.78, 0.97 and 2.27 (N m, or N for the closure force). No sign class differs
  between members and no decided nominal sign is lost in the union.

Non-claims: no subject-specific inertia; the sampling law is a declared
hypothesis, not a population or measurement distribution; identification is
conditional on the observed ground reaction and club wrench (the fixture's
reaction is a simulation, not a force plate); the hull is not a confidence set;
three short windows do not characterise the swing; the capacity prior is ASSUMED
and unchanged. Bi-level inertia learning is out of scope.

