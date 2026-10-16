# Per-Child Contract Appendix: Trajectory, Replay and Engine Contracts

This file continues `contracts.md` with the trajectory, replay, engine and
ensemble contracts from NSX-23 through the NSX-61 and NSX-62 Ensembles.

## NSX-23 Trajectory Allocation (#12566, Strong)

`allocate_trajectory(feasible_sets, objective, *, reference=None)`. Golden
`c_pm_identity`: with uniform `c_plus`, `c_minus` the cost is
`1/2 (c_plus + c_minus) sum|p| + 1/2 (c_plus - c_minus) P_net` and `P_net` is
constant on `J_c v = 0`; the uniform-weight minimiser is unique and equal for
all `c_minus / c_plus`; heterogeneous weights give a different minimiser.
Curvature (golden `planar_*.curvature`): the retained-space Gram matrix
eigenvalues are positive.

Implemented in `loop_nullspace/allocation.py` (types in `allocation_spec.py`):

- Unknown `w = (z_1..z_T, s)`; torques are affine, `tau = T0 + Ntil z`. Each
  quadratic term is `|A tau - a|^2` with `S = diag(torque_scale)` (convention
  `C-METRIC-CAPACITY`): effort `a_e sum w_k |S^-1 tau_k|^2`, rate
  `a_r sum |S^-1 dtau|^2 / dt_k`, reference `a_ref sum w_k |S^-1 (tau - ref)|^2`
  (`w_k` trapezoid weights over `times_s`). The gross-power epigraph
  `s >= c+ p`, `s >= -c- p` with `p = v tau` is piecewise linear.
- Strict convexity is checked at build time on the retained space (the
  z-directions that change a torque): the quadratic Gram matrix must have rank
  equal to `sum_k rank N_a,k`. Torque-invisible directions (multiplier / `f_g`
  gauge) are reported as `gauge_dimension`, not penalised; the optimal torque
  is unique, the gauge coordinates are not. A failure raises
  `NotStrictlyConvexError` naming the missing term(s) (effort or reference).
- Reduction report: `ABS_POWER_ONLY` (`c+ = c-` per joint), `UNIFORM_NET_PLUS_ABS`
  (uniform weights and `v^T dtau/dz = 0`, so `c-/c+` is a dead parameter at fixed
  mean), `NONE` (heterogeneous). The reduced `1/2 (c+ + c-) |p|` form is used
  when detected (`allow_reduction=False` forces the unreduced epigraph).
- Result: `QuantityRecord(FAMILY_DEPENDENT)` with every hypothesis (feasible-set
  ids plus `H-EFFORT-CAPACITY`, `H-SMOOTH-TORQUE-RATE`, `H-GROSS-POWER-COST`
  when active), row and cone duals, Clarabel KKT residuals (scaled), solve time.
  Exact cones are used; linearised cone rows are ignored. The reference-tracking
  term has no registry hypothesis; it is flagged by `reference_used`.
- Solver: `solve_socp` gained an optional quadratic term `P` (objective
  `1/2 x'Px + c'x`, stationarity residual includes `Px`) and `tol`; the
  `P=None` behaviour is unchanged.
- Anthro acceptance (`tests/integration/test_loop_nullspace_anthro_allocation.py`,
  marked `slow`; run with `-m slow`): the feasible sets are the NSX-60 sets
  (`envelope_only` preset, `H-TORQUE-CAPACITY` prior unchanged, Coulomb foot
  cones). The 199 decomposable benchmark frames split into 124 BOUNDED and 75
  INFEASIBLE (each conflict set `H-TORQUE-CAPACITY` alone), so a trajectory
  allocation over the full window is infeasible by construction and returns
  `INFEASIBLE` after 3 iterations (0.21 s, no record, no relaxation). The
  acceptance window is the longest contiguous run of BOUNDED frames at the
  benchmark stride: record frames 96 to 339, 82 frames (0.009 s apart, so the
  rate term differences strided frames). Objective: capacity-weighted effort
  (weight 1.0) plus torque rate (weight 0.1), no gross-power or reference term.
  Measured on host Brick (shared 4-core, load about 4), Clarabel:

  | Window | Frames | Status | Iterations | Solve time | KKT primal | KKT dual | KKT compl. | Retained rank / gauge |
  |---|---|---|---|---|---|---|---|---|
  | Longest BOUNDED run (96 to 339) | 82 | OPTIMAL | 24 | 0.289 s | 2.6e-14 | 5.7e-10 | 2.2e-13 | 794 / 218 |
  | First 10 frames of that run | 10 | OPTIMAL | 18 | 0.028 s | 1.5e-13 | 7.3e-12 | 9.8e-13 | 87 / 13 |
  | All 199 decomposable frames | 199 | INFEASIBLE | 3 | 0.214 s | none | none | none | 1679 / 350 |

  On the 82-frame run the objective terms are effort 3.00 and rate 424.3, the
  record is `FAMILY_DEPENDENT` with `H-TORQUE-CAPACITY` and `H-FRICTION-CONE`
  among its hypotheses, and every torque lies inside the (ASSUMED) capacity. A
  window straddling one INFEASIBLE frame (for example 90, 93, 96) is
  `INFEASIBLE` too. The solve time excludes building the 199 frames and sets
  (about 36 s, the benchmark's cost). Frame 510 (airborne) is refused before
  the window is chosen, and the window never spans it because the window must be
  contiguous at the benchmark stride.
- Non-claims: the optimum is a hypothesis-conditioned representative, not an
  estimate of true hand loads; no "zero sign flips" criterion is used.
  `force_nullspace/redistribution.py` is untouched (API-compatible); its shim
  onto this core is NSX-24.

## NSX-24 Allocator Shims (#12567)

Preserve public signatures of `ContactForceAllocator` and
`MultiEngineTorqueAllocator`; map `HARD_ZERO_TRAIL` to the NSX-15 trail
`zero_side_solve`. An `INCONSISTENT` frame is reported with its residual.
Golden: `planar_two_arm_full.zero_side`.

Implemented as follows (the first four items are the contract; the last
records what was deliberately not done).

- `loop_nullspace/qp_core.py` is the one place that turns "quadratic, equality
  rows, inequality rows, box, exact Coulomb cones" into `solve_socp`
  (Clarabel). Lower and upper bounds that coincide become equality rows.
- `ContactForceAllocator.allocate` solves the same scaled QP it always built
  (diagonal weights, 8-facet inscribed friction pyramid, actuator and root
  slack penalties) through `solve_qp`. SLSQP agrees with the conic optimum to
  about 1e-13 relative on the parity fixtures; where SLSQP stops short, the
  conic objective is no larger and the result is feasible.
- `redistribute_forces` and `redistribute_trajectory` solve one convex QP in the
  stacked null-space coordinates with exact Coulomb cones (SOC blocks), not
  SLSQP. The rate penalty is evaluated on physical `x`, so the result is
  invariant to basis sign flips.
- A non-optimal solver status (`INFEASIBLE`, `NUMERICAL`) is never repaired:
  the allocator returns the zero iterate with `success=False`; the
  redistribution shims return the particular solution with `converged=False`
  and the independent audit decides `feasible`. A missing `clarabel` raises
  `ImportError` naming the `loop-nullspace` extra (rule R11); there is no
  fallback optimizer.
- Not done: `HARD_ZERO_TRAIL` is not routed through NSX-15 `zero_side_solve`.
  The legacy signature carries only `tau_rnea` and the Jacobians, not a
  `ClosedLoopSnapshot`, so the three-case diagnostic has no inputs. The
  existing slack formulation already reports `INFEASIBLE_HARD_ZERO_TRAIL`
  rather than hiding the inconsistency. The adapter over `allocate_trajectory`
  and `FeasibleSet` requires snapshots and is the next step, together with
  removing the shims.

## NSX-25 Pareto and Probes (#12568)

```python
def pareto_front(feasible_sets, objectives, eps_grid, *, references=(None, None)) -> ParetoFront
def certify(feasible_sets, predicate, window=None, *, coupling=(), aggregation="ALL") -> Certificate
def verify_certificate(cert, feasible_sets, *, coupling=(), predicate=None) -> VerificationReport
def trajectory_envelope(feasible_sets, quantities, *, coupling=()) -> TrajectoryEnvelope
```

Modules: `probes.py` (public API, re-exports), `probe_certificates.py`
(leaf certificates, numpy verifier), `trajectory_system.py` (stacked set, solver
adapter), `trajectory_envelope.py`, `pareto.py`.

**Predicates.** `AffinePredicate(quantity, sense, threshold)` with
`sense in {ge, le}` normalises to `f(z) = s (const + coeff . z) >= s theta`.
Only affine functionals exist: the constructor accepts an `EnvelopeQuantity`,
a sequence of them or a quantity factory; `AffinePredicate.from_function`
evaluates a callable at the origin and the unit vectors and tests the fit at
random points, raising `NonAffinePredicateError` (a `ValueError`) otherwise.
Non-convex predicates are out of scope.

**Trajectory-coupled set.** The frames' sets are stacked in
`z = (z_1, ..., z_T)` and intersected with hard rate bounds
`|q_{k+1} - q_k| <= delta_k` on an affine quantity (`RateBound`, the
`trajectory_coupled` oracle; hypothesis id `H-SMOOTH-TORQUE-RATE`). The
coupled set is a subset of the static one. `certify` keeps Coulomb cones as
exact second-order-cone blocks, so neither verdict depends on the declared
polygon; `trajectory_envelope` uses the declared linearisation like the static
envelopes.

**What "certified" means (strongest standard notion, no new math).** For a
normalised functional `f` and the stacked system `A z <= b` (polyhedral rows
`G z <= h` plus cone blocks `b_j - A_j z in SOC`), a leaf carries

* `LOWER_BOUND`: a primal witness `w` (feasible, value `f(w)`) and a dual `d`
  with `d >= 0` on rows, `d_j in SOC` on cone blocks (self-dual) and
  stationarity `c + A^T d = 0`. Weak duality gives `f(z) >= const - b . d` for
  every feasible `z`; the witness gives `min f <= f(w)`. This is primal
  feasibility plus a zero-duality-gap certificate checked by the verifier, not
  a trust in the solver status.
* `FARKAS`: `d` in the dual cone, `A^T d = 0`, `b . d < 0` (set empty).
* `WITNESS_ONLY`: `f` is unbounded below; a feasible witness with `f` below
  the threshold (found with the violation as an extra row) proves failure.

Verdict, with `atol = SIGN_ATOL_RTOL * scale` (the NSX-22 sign tolerance, never
a bare 0): `HOLDS` iff certified bound `> theta + atol`; `FAILS` iff a feasible
witness has `f < theta - atol`; otherwise `UNDECIDED("BOUNDARY")` (optimum
within `atol` of the threshold, consistent with "a bound that touches zero is
INDETERMINATE"). An empty set gives `UNDECIDED("EMPTY_FEASIBLE_SET")` with its
Farkas leaf, never a vacuous HOLDS. A leaf the verifier rejects is downgraded
to `UNDECIDED("CERTIFICATE_REJECTED: ...")`. `aggregation="ALL"` asks the
predicate of every window frame (one LP each, one persistent model);
`"SUM"` asks it of `sum_k q_k` (one LP).

`verify_certificate` rebuilds the stacked system from the sets, checks that its
SHA-256 digest equals the certificate's, re-checks every leaf with numpy only
(`verify_leaf`) and, if given the predicate, that the leaf functionals are its
own. Tolerances: stationarity, cone membership and stored values to `1e-8`
scaled; witness row violation to `1e-7` scaled (the HiGHS feasibility default).
The certificate is therefore exact up to the reported residuals, not a
rational-arithmetic proof. `Certificate.to_json` / `from_json` round-trip every
array and number exactly (shortest round-trip float text).

**Pareto.** `objectives = (J_a, J_b)` are two `ObjectiveSpec`s over the same
frames (each objective is the sum of the quadratic terms its spec enables;
`J_a` may include gross power, `J_b` may not). The solve is
`min J_a(z) s.t. J_b(z) <= eps`, with `J_b = |M z + m|^2` imposed exactly as the
cone `(sqrt(eps), M z + m) in SOC`; `J_a` must be strictly convex on the
retained torque space (NSX-23), so each OPTIMAL point is unique and Pareto
optimal in the torque objectives. The cone multiplier gives the slope
`d J_a / d eps`. Oracle: two frames, one torque, closed form in
`tests/unit/loop_nullspace/test_pareto.py`. An `eps` below the attainable
`J_b` is reported `INFEASIBLE`. Every point is `FAMILY_DEPENDENT` and lists all
hypotheses of both objectives. `explore_torque_tradeoffs` keeps its return
shape and now emits a `FutureWarning` naming it a weight sweep.

Goldens: `envelope_analytic_lp` (`trajectory_coupled`, `infeasible`,
`unbounded`, `static`).

## NSX-26 Reaction Restriction (#12569)

`restrict(snapshot, kind) -> ClosedLoopSnapshot`. Admissible restrictions are
the nesting `WELD6 -> SHAFT_FREE5 -> BALL3` by row selection in the grip frame:
SHAFT_FREE5 drops the twist moment about the shaft axis `u` (row `tx`); BALL3
keeps the three force rows. The set of admissible reactions for the restricted
kind is a subset of the WELD6 set, so envelopes of the restricted kinds are
contained in the WELD6 envelope for the same fixed welded trajectory (note the
direction: restricting reactions shrinks the feasible set). Row `tx` is the
shaft twist only when the grip x axis is the shaft; `restrict` refuses
SHAFT_FREE5 otherwise unless `shaft_axis` is given.

`restriction_sensitivity(trajectory, kinds, quantities, *, feasible_set_of, ...)
-> RestrictionReport` takes an `AdmittedTrajectory` and snapshot-aware quantity
builders (`snapshot_quantities` lifts an envelope factory;
`restriction_hand_split.hand_split_quantities` gives the six hand-split
coordinates on the embedded multiplier). `kinds[0]` is the reference. Per kind
the report holds feasible, infeasible (with conflict set) and
balance-inconsistent frame counts, width ratios, sign-class changes and
containment checks. The priors live in `feasible_set_of` and are the same for
every kind; `BALANCE_TOL` is unchanged. Output label
`REACTION_RESTRICTION_SENSITIVITY`; the report carries its non-claims. Goldens:
`restriction_counts` (exact sympy). Anthro record:
`records/restriction_anthro_stride1.json` (199 frames).

## NSX-72 Closure Alternatives (#12590)

`closure_alternatives.SpatialGripPlant(model, *, floating_root=False)` for
`model` in `GripModel` (`WELD`, `SHAFT_TWIST_REVOLUTE`, `TRAIL_BALL`,
`LEAD_BALL`); `GRIP_MODEL_SPECS[model]` gives the closure kind, kept rows,
extra coordinates, a summary and the stated limitation. Each plant is a
`ClosurePlant` (`closure_pose_residual`, `closure_rate_matrix`) with the model's
own rows in the shaft-first grip frame. `motion_frames(plant, motion, *,
n_frames, seed, source=None)` samples `Motion.WELDED | TWIST | TILT | OWN` at
the closed posture; `position_probe(plant, motion, *, steps)` takes small
finite steps off it; `admit_model` / `admission_outcome` return a
`ModelAdmission` with the model's own scaled `max_phi`, `max_phi_dot` and
`max_phi_ddot`. A motion sampled in a different coordinate count is
`not representable` (`ValueError`), never mapped. Gates stay at the NSX-00
`1e-6`.

`closure_alternatives_report.evaluate_grip_model(admission, *, quantities,
feasible_set_of) -> GripModelResult` runs the NSX-26 machinery with the model's
single closure kind on an admitted trajectory only (a refused trajectory has
`report=None`). Result fields: kinematic nullity (`n - rank J_c`, the admitted
velocity dimension), allocation nullity, outcome counts, envelope widths.
Output label `CLOSURE_ALTERNATIVE_COMPARISON`; the non-claims travel with it.
Rows-only models (`SHAFT_TWIST_REVOLUTE`, `TRAIL_BALL`) on a welded trajectory
reproduce the NSX-26 restriction exactly (tested); what they change is the
admitted motion. `LEAD_BALL` adds three unactuated coordinates (club on a z-y-x
ball at the lead hand, trail hand welded to the club) and exists only on the
spatial fixture. Counts: the rows-only models match the `restriction_counts`
golden; `LEAD_BALL` counts are structural (generic rank), not an exact golden.
Record: `records/closure_alternatives_spatial.json`,
`records/closure_alternatives_anthro_stride1.json`; report:
`closure_alternatives_report.md`.

## NSX-30 Constrained Counterfactuals (#12570)

```python
def constrained_acceleration(M, h, tau, J_c, Jdot_v, *, baumgarte=None) -> tuple[np.ndarray, np.ndarray]  # (qdd, lambda)
```

KKT projector `M^-1 (I - J_c^T Lambda_c J_c M^-1)` with
`Lambda_c = (J_c M^-1 J_c^T)^-1`. Postconditions: `J_c qdd + Jdot_v = 0` to
`1e-9` scaled; the total control-induced acceleration is independent of the
chosen `lambda` family. Planar goldens provide `M`, `h`, `J_c`, `Jdot_v`.

Implementation (NSX-30): `loop_nullspace/constrained_dynamics.py` holds the single
projector (`constrained_acceleration`, `kkt_projector`, declared `Baumgarte`
terms, `closure_acceleration_residual`); `tau` is the generalised load
`S^T tau_a`. The pendulum solver calls it with
`singular_mass_fallback=True` because its massless club coordinate makes `M`
only semi-definite; the fallback solves the equivalent augmented block system
and is an error without the flag. `simulation_backends.protocol` adds
`ConstrainedEquationsOfMotionProvider` (`closure_jacobian`, `closure_drift`);
`ztcf_acceleration`, `zvcf_acceleration`, `drift_and_control_split` and
`zero_velocity_control_preserved_acceleration` dispatch to the projector for such a
provider. `constrained_counterfactual_report` returns ZTCF, ZVCF, the control
total (`INVARIANT`) and the per-actuator split (`FAMILY_DEPENDENT`).
`loop_nullspace.provider.AdapterConstrainedProvider` exposes any `FULL`
NSX-13 adapter as such a provider. `create_counterfactual_rollout` requires
`dynamics=CounterfactualDynamics(provider, nullspace=...)`, integrates the
provider with RK4 (load held per step), and `NULLSPACE_EXPLORATION` samples `z`
through `NullspaceExploration` (NSX-15 decompositions); the `m_eff = 1.0` mock
and the `5 sin(t)` placeholder are removed.
Without an explicit `dynamics`, `motion_matching/counterfactual_provider.py`
builds the provider from the session's own model specification (Pinocchio
adapter, `AdapterConstrainedProvider`, re-expressed in the session coordinate
order by `ReorderedProvider`) and caches it by `model_sha256`. A specification
missing `bodies`/`joints`/`coordinate_order`, with a mismatched coordinate
order, or without a `closure` fails closed with a `ValueError` naming the
input, which the route returns as 400; the Pinocchio model needs a closure,
so no unconstrained provider is built from a specification. Achieved evidence
on the 41-coordinate anthro model against raw `constraintDynamics`: ZTCF
8.4e-12 and ZVCF 3.9e-13 scaled; ZTCF + control total under a random actuated
load (reference magnitude 9.5e3) 2.5e-12 scaled.
`ztcf_zvcf` keeps its naming: ZTCF removes the control torque, ZVCF sets
`v = 0` and `u = 0`; on a loop both use the projector with the reaction
re-solved, never a held `lambda`.

## NSX-31 Replay (#12571)

`replay(allocation, provider, *, policy, setup, feedback=None, settings=None)
-> ReplayReport` (`loop_nullspace/replay.py`; inputs in `replay_spec.py`,
ground loads in `replay_ground.py`). It integrates the constrained dynamics
forward from the first state only, under the allocated torques, and compares
the result with the reference trajectory. It reuses the NSX-30 KKT projector
(`constrained_acceleration`, with the reaction re-solved at every RK4 stage) and
the counterfactual RK4 (`rk4_step_with_load`, the generalisation of the
existing `_rk4_step` to a time- and state-dependent load). There is no second
integrator or projector.

### Modes and the Mandatory Label

| `ReplayPolicy`           | `ReplayMode` | `evidence_label`                | What it is                                                            |
| ------------------------ | ------------ | ------------------------------- | --------------------------------------------------------------------- |
| `INDEPENDENT_TIME_ONLY`  | `OPEN_LOOP`  | `independent_open_loop_replay`  | Allocated torque applied as is; only time enters.                     |
| `STATE_FEEDBACK`         | `FEEDBACK`   | `feedback_assisted_tracking`    | Allocated torque plus a declared feedback law on the state error.     |

`ReplayReport.__post_init__` refuses a mode that disagrees with its policy, an
open-loop report that carries any feedback, a feedback report without its law,
series and budget, and any state reset. `may_present_as_validation` is always
`False`: a feedback-assisted replay is tracking, never validation, and a UI may
not show it as the latter. The report also carries `NON_CLAIM`.

### Equations

With `S` the actuation selection (coordinate rows), `tau_a(t)` the allocated
torque interpolated from the allocation nodes (`zoh`, `linear` or `cubic`
spline: orders 1, 2 and about 4), `g(t, q) = J_g(q)^T f_g(t)` the known external
ground load and `(a, lambda) = KKT(M, h, S^T tau + g, J_c, Jdot v)`:

- open loop: `tau = tau_a(t)`;
- feedback: `tau = tau_a(t) + tau_fb`, `tau_fb = F(q_ref - q, v_ref - v, q)`.

`tau_fb` is stored as its own series and never added into `allocated_torque`.
Its budget is relative to the allocated series: `peak_norm_ratio = max_t
|tau_fb| / max_t |tau_a|`, `energy_ratio = int |tau_fb|^2 dt / int |tau_a|^2
dt`, `rms_ratio`, absolute mechanical work `int sum_i |tau_i v_i| dt` of each
series, the per-joint RMS, and the fraction of saturated samples (trapezoid
integrals). A zero allocation with non-zero feedback has ratio `inf`, never
`nan`.

Closure drift is reported in both modes: velocity `J_c v` always, position
`Phi(q)` only when the provider exposes `closure_error` (otherwise `None`, not
zero). Any Baumgarte stabilisation (`ReplaySettings.baumgarte`, gains
`alpha`, `beta`) is declared in the report together with the peak
acceleration it added, `stabilisation_acceleration_peak`, and is reported apart
from the physics; the default is none.

Time to divergence is the first replay node at which `max_i |q_i - q_ref,i|`
exceeds `divergence_tolerance` (radians or metres per coordinate; 0.05 for the
anthro runs). A run that blows up (non-finite state, a state above `1e8`, or a
KKT failure) stops there and says so in `terminated_reason`.

### Declared Feedback Laws

`FeedbackLaw(kp, kd)` is a per-joint or symmetric PSD joint-space PD law.
`AccelerationFeedbackLaw` is the state-dependent acceleration-space PD used for
the anthro runs: `tau_fb = pinv(R(q)) (omega^2 e + 2 zeta omega e_dot)` with
`R(q) = S P_c(q) S^T` the actuated acceleration response of the closed loop at
the current configuration and `zeta = 1`. It is scale-free across joints
whose inertia spans `1e-6` to `1e0` kg m^2 and removes directions the loop makes
uncontrollable (`rcond = 1e-6`). Only the 35 actuated coordinates get feedback;
the six root coordinates are unactuated.

### Window Rule (Anthro Fixture)

Frames `0, 3, ..., 597` of the ground-supported anthro record are the NSX-22
benchmark set, 199 frames of which 124 have a non-empty free feasible set under
the ASSUMED capacity prior (75 are INFEASIBLE, as already recorded). The replay
window is the longest run of adjacent (3 record steps apart) usable frames,
earliest on ties. It is chosen by that rule alone, not by outcome: runs are
(frames, count) 0 to 90 (31), **96 to 339 (82)**, 345 to 348 (2), 534 to 537
(2), 543 (1), 549 to 564 (6). The window is record frames 96 to 339, `t =
0.288` to `1.017 s`, 0.729 s long. The initial state is the closure-projected
measured state of frame 96. The allocation (NSX-23, `rate_weight = 0.1`) is
OPTIMAL with KKT residuals primal 2.6e-14, dual 5.7e-10, complementarity 2.2e-13.

Ground loads: the allocated `f_g` is applied as a known external load
`J_g(q)^T f_g` at the replayed configuration on the contact set of the
allocation frame. It is an input, not feedback, and not contact dynamics: no
contact is released or made.

Load-map check (so a divergence is dynamics, not plumbing): at the measured
states, the allocated loads reproduce the measured accelerations to 2.5e-7
(1.8e-9 relative to the peak acceleration).

### Results on the Anthro Window

Open loop (`INDEPENDENT_TIME_ONLY`, RK4 3 sub-steps per 9 ms, linear
interpolation):

- Diverges early: time to divergence **21 ms** (tolerance 0.05); the error is
  0.013 at 10 ms, 0.10 at 30 ms, 0.63 at 50 ms, 8.8 at 100 ms, and the run
  blows up (state above `1e8`) at `t = 0.45 s`. Early log-linear growth rate
  about 105 per second (6 to 30 ms): this is a hard-unstable open loop, not
  noise. Worst coordinates at 50 ms: `mtp_angle_r` 0.68, `mtp_angle_l` 0.41,
  `subtalar_angle_l` 0.24 rad.
- Integrator check: 6 sub-steps give the same 21 ms and the same early errors
  (0.0127 at 10 ms, 0.103 at 30 ms). `zoh` diverges at 12 ms, `cubic` at 21 ms.
  Interpolation changes the amplitude, not the verdict.
- Closure drift stays tiny until divergence (velocity 7.4e-7, position 1.2e-8)
  and is huge only in the blow-up tail (max 1.3 velocity, 1.4e-3 position); do
  not read the tail maxima as a closure statement.

Feedback-assisted (`STATE_FEEDBACK`, acceleration PD, `zeta = 1`). `omega` is the
natural frequency of the declared law; `capacity` is the ASSUMED torque
capacity prior.

| `omega` [rad/s] | sub-steps | time to divergence | max state error | peak ratio | energy ratio | abs-work ratio | peak / capacity | joints over capacity |
| --------------- | --------- | ------------------ | --------------- | ---------- | ------------ | -------------- | --------------- | -------------------- |
| 50              | 3         | 36 ms              | 4.1             | 1.93       | 1.65         | 1.17           | 8.8             | 20                   |
| 100             | 6         | 135 ms             | 2.9             | 1.63       | 0.99         | 1.18           | 6.5             | 19                   |
| 150             | 6         | none               | 0.048           | 0.196      | 0.0076       | 0.12           | 1.24            | 1                    |
| 300             | 12        | none               | 0.047           | 0.229      | 0.0086       | 0.13           | 1.55            | 1                    |

Reading it honestly: the allocated torques alone do not reproduce the motion
beyond about 21 ms. Tracking the whole 0.729 s window needs feedback at
`omega >= 150`; at 150 the feedback has 0.76 percent of the allocated energy,
12 percent of its absolute work and a peak 20 percent of the allocated peak,
but its peak is 1.24 times the ASSUMED capacity on `RFInput`, one joint, and
1.55 times at `omega = 300`. At `omega = 50` and `100` the law does not hold the
state, so their ratios describe the feedback fighting a diverged state and are
not a budget; do not quote them as one. The numbers hold for this law, window,
capacity prior and tolerance only.

Reproduce (about 3.5 min, needs Pinocchio, HiGHS and Clarabel; `-m slow` is
required because the default `addopts` deselects it):

```bash
LOOP_NULLSPACE_REPLAY_JSON=<path>.json python3 -m pytest \
  tests/integration/test_loop_nullspace_anthro_replay.py -n 0 -m slow \
  --timeout=1500 -p no:cacheprovider
python3 -m pytest tests/unit/loop_nullspace/test_replay.py \
  tests/unit/loop_nullspace/test_replay_feedback_laws.py -n 0
```

The record holds window, allocation status and KKT, the open-loop report for
each interpolation, the finer-step run and the feedback sweep.

### Oracle Evidence (Planar Golfer)

The reference is DOP853 at `rtol = 1e-13` with dense output (no inverse crime).
Exact-torque open-loop replay reproduces it within integrator error; RK4 with a
held load is fourth order; interpolation orders are 1, 2 and about 4; feedback is
negligible when tracking is exact (peak under 1e-4 of allocated, energy under
1e-9); a zero-gain law equals the open loop; the feedback budget grows with an
injected parameter mismatch; and a real NSX-23 allocation (a different family
member from the driving torque, differing by more than 1 N m) replays to the
same trajectory (error 5.6e-6).

### Failed Experiments

- Fixed per-joint PD gains (`kp = 300`, `kd = 30`) were unstable under explicit
  RK4 on the oracle (the state exploded within 10 nodes); the oracle test uses
  `kp = 40`, `kd = 1.5`.
- On the anthro chain, diagonal PD and a frozen-`Lambda` matrix PD both blew up:
  inertias span six orders of magnitude and the closure couples them. This is
  why the state-dependent `AccelerationFeedbackLaw` is the declared law.
- A stride-1 allocation of the same window returned `NUMERICAL` with
  `rate_weight = 0.1`; the stride-3 window above is the reported one.

### Limitations and Non-Claims

- Inverse kinematics, feedback-assisted tracking and independently replayed
  open-loop dynamics are different things. Only the open-loop mode is an
  independent replay, and even it shows only that this allocation, with this
  model, does or does not hold; it does not validate or refute the torques.
- The ground load is an applied input. A diverged replay keeps pushing at the
  allocated force through contacts that may have moved.
- The feedback law is one declared choice. Another law gives another budget.
- Capacity, friction and grip hypotheses remain ASSUMED; foot and head
  observability limits are unchanged.
- Software correctness only. Scientific qualification stays in the
  design-manual governance pathway; a successful run is not acceptance.

## NSX-40 Report, Service and Routes (#12572)

`schemas/loop-nullspace-report-v1.json`; `loop_nullspace.analyze(trajectory,
engine, preset, options)`; routes `POST /analysis/loop-nullspace/jobs`,
`GET .../jobs/{id}`, `DELETE .../jobs/{id}`. Cache key is a hash of input
provenance, model provenance, registry version, preset and options.

## NSX-41 Output Policy (#12573, Strong)

`display_decision(record, surface) -> DisplayDecision` per ADR-0054 decision
D7 once the owner accepts an option. Unknown combinations default to withhold.

## NSX-42 Plot Model (#12574)

`build_plot_model(report, display_decisions) -> NSXPlotModel`. Panels:
`invariants`, `envelopes`, `hand_split`, `allocation`, `diagnostics`.
Envelope segments carry `display_class` in `DETERMINED_SIGN`, `INDETERMINATE`,
`UNBOUNDED`. A representative never appears without its band. The golden JSON
plot model is checked in by NSX-42 and consumed by NSX-43 and NSX-44.

## NSX-43 PyQt Tabs (#12575) and NSX-44 React Panel (#12576)

Render only from `NSXPlotModel`; no math in either UI. Parity entry
`loop_nullspace_exploration` in `src/config/feature_parity.json`; regenerate the
matrix with `python3 -m scripts.generate_feature_parity_matrix`. NSX-43 uses
a lazy embed adapter (PyQt6 imports inside `create_main_widget`). Both assert
use of `display_decision`.

## NSX-50 Dynamic Dual Transforms (#12578)

`velocity_map`, `acceleration_map` (with the `Tdot v` term for nonlinear
coordinate maps) and `generalized_force_map = T^T`. Postcondition: power
`tau . v` equal across maps to `1e-12` relative. Closure conversion uses the
golden `closure_convention`.

## NSX-51 to NSX-54 Engine Snapshots (#12579, #12580, #12581, #12582)

Each implements the NSX-13 adapter methods and passes canonical equality with
Pinocchio on states projected by `project_to_closure` to `1e-6` scaled.
MuJoCo (NSX-51): hard-KKT evaluation, drift method recorded in provenance
(finite difference declared). MuJoCo declares `CLOSED_LOOP_SNAPSHOT` FULL from `mj_fullM`,
`qfrc_bias`, the weld site Jacobians and a central-difference drift
(`drift=central_finite_difference` in the model provenance); stock soft-weld
stepping is the separate divergence entry `mujoco-stock-soft-weld-vs-hard-kkt`.
The cross-engine test holds `M`, `h`, `J_c` and `Jdot_v` to 1e-11, 1e-12 and 1e-8
scaled (measured 2e-14, 3e-15, 3e-9). MyoSuite (NSX-52, `MyoSuiteForceAdapter`): subclasses
`MujocoForceAdapter` (no second implementation), records
`kkt_regularization=<value>` in the model provenance (`name=myosuite`), and is
`FULL` only while that value is exactly 0, else `NONE`. Drake (NSX-53): native
adapter, explicit `from pydrake.X import Y` imports, no module-level mocks.
OpenSim (NSX-54): delegates to `OpenSimFullBodyParityAdapter`.

## NSX-55 Simscape (#12583, Strong)

R2025b only (AGENTS.md), versioned export file with release and model checksum;
otherwise a deferred-validation record and capability `NONE`.

## NSX-56 Conformance (#12584)

`tests/integration/cross_engine/test_loop_nullspace_conformance.py` is the one
parametrised harness over every FULL engine (Pinocchio reference; MuJoCo, Drake,
OpenSim). Each state is projected once with `project_to_closure`, each engine
builds its snapshot through `snapshot_from_adapter`, and arrays are permuted to
the reference order by coordinate name.

Track N compares canonical `M`, `h`, `J_c`, `Jdot_v` tightly. Tolerances are the
per-pair, per-metric entries (`check: closed_loop_snapshot_conformance`) in
`tests/integration/cross_engine/divergence_registry.yaml`, with
the rationale and the measured agreement; the grip frame, club inertia split and
loop topology are compared in the same track. Measured: `M`, `h` below 1e-15,
`J_c` below 3e-15, `Jdot_v` 1.8e-12 (Drake, OpenSim, analytic or exact) and
1.2e-10 (MuJoCo central finite difference), all scaled by `max(1, |ref|)`.

Track S compares quantities derived from the snapshot: the rigid KKT
acceleration and grip-wrench multiplier, kinetic energy, power, closure singular
values (rank 6) and the closed-form grip-wrench envelope over a unit actuator
box. Its tolerance is propagated from the Track N differences measured in the
same run (`kkt_bound`: `|K^-1| (|dr| + |dK||x|) / (1 - |K^-1||dK|)`; Weyl for
singular values), never hand-widened. A perturbed-engine control proves both
tracks discriminate.

Capability per engine: Pinocchio, MuJoCo, MyoSuite, Drake and OpenSim are FULL
and are registered in `create_engine_force_adapter`; the harness skips MyoSuite
cleanly when `myosuite` is not importable. Simscape (no native adapter) and the
synthetic fixtures declare NONE and are asserted to refuse a snapshot. MyoSuite
2.12.2 agrees with Pinocchio to the same digits as MuJoCo (the adapter reuses
the MuJoCo code); MuJoCo 3.6.0 and 3.8.0 give identical Track N errors to three
significant digits. Agreement between engines is not physical
validation.

## NSX-60 Lower-Body Loop (#12585, Strong)

Two-foot ground contact is a second unilateral null space: with rigid contacts
the balance is `A x = b`, `A = [S^T J_c^T J_g^T]`, `x = (tau, lambda, f_g)`.
Golden `spatial_counts` variants `floating_root_two_point_feet` and
`floating_root_four_point_feet` fix the counts (two point feet: contact-map
rank 5, nullity 7, gauge 1; four point feet: nullity 12). Golden
`lower_body_observation` (sympy, exact, no repository code) gives
`rank(E_x N)` for five observation maps, and the numeric fixture at zero root
pose (`zero_root_pose=True`) reproduces every entry.

| Observation map | 2 points | 4 points |
|---|---|---|
| Net force and moment (root rows already fix it) | 0 | 0 |
| Net force | 0 | 0 |
| Vertical load of each foot | 0 | 1 |
| Force of each foot | 1 | 3 |
| Every point force | 1 | 6 |

Implementation (`loop_nullspace/ground_contact.py`, `ground_envelopes.py`):

- Seam: `PinocchioForceAdapter.foot_contact_candidates(q)` returns the six
  contact spheres (`ContactCandidate`: `FootContact` at the sphere centre, world
  ground normal, spec static friction, signed gap). `snapshot_from_adapter(...,
  foot_contact_tol_m=T)` keeps candidates with `gap <= T` and stacks their rows
  of `compute_contact_jacobian` as `J_g`; without `T` (default) `J_g` stays
  `None` and no other engine changes. An adapter without the method raises
  `CapabilityUnavailable`.
- Cones: `foot_friction_cones(snapshot)` builds one `ContactCone` per contact over
  its three `f_g` entries, reusing `force_nullspace.constraints.FrictionCone`;
  they enter `FeasibleSet.from_hypotheses(cones=...)` under `H-FRICTION-CONE`.
- Observation: `GroundObservation(selector, values, sigma, labels, source,
  coverage=3)` with mandatory positive per-row `sigma` and constructors
  `from_point_loads`, `from_contact_groups`, `from_normal_load_and_cop`.
  `observation_accounting` reports `rank(E_x N)`, the observed nullity, the
  singular values of `diag(1/sigma) E_x N` and the chi-square disagreement with
  the balance; `restrict_to_observation` is the zero-uncertainty limit (drops the
  visible directions, moves `x_p` to the weighted least-squares member, refuses
  inconsistent data with `ObservationInconsistentError`);
  `FeasibleSet.from_hypotheses(observation=...)` adds the band
  `|E f_g - g| <= k sigma` as `grf` rows under the new hypothesis
  `H-GRF-OBSERVATION` (preset `envelope_grf_observed`), keeping the null
  dimension.
- Quantities: `closure_wrench_quantities()` and `ground_force_quantities(...)`
  feed `static_envelopes` with the multiplier block and the ground loads.

Anthro acceptance report (same 199 frames, same host, `H-TORQUE-CAPACITY` prior
unchanged; observed variants run on the 124 frames whose free set is non-empty,
since observation rows only add constraints):

- The record `dynamics_record.npz` carries `normal_force_n`, `weight_fraction`,
  `cop_m` and sphere heights: simulation output of the compliant contact law
  (not a force plate), net vertical load and centre of pressure only, with no
  per-foot split and no tangential force. Declared uncertainties are hypotheses:
  5 N and 5 mm.
- Record-native observation (3 rows; 180 frames with load): visible rank 0 on
  43 frames, 1 on 10, 2 on 127; nullity median 9 to 7; singular values over
  sigma at most 0.09 (median 0.05), so the data are not resolved at the declared
  sigma. Result: every envelope width equals the free one (median ratio 1.0 to
  `1e-11`) and no sign class changes. The record's GRF adds no information.
- Simulated per-foot plates (force sum per side from the compliant law at the
  projected state; 181 frames): visible rank 0 on 46, 1 on 8, 2 on 45, 3 on 82;
  nullity median 9 to 7; singular values over sigma 0.12 to 1.17 (median 0.25).
  Result: the leg-joint torque envelopes narrow; median width ratio
  (observed over free): hip flexion 0.42 (both legs), hip adduction 0.12 (right)
  and 0.14 (left), hip rotation 0.53 (right) and 0.42 (left), knee 0.47 and
  0.42, ankle 0.81 and 0.80, subtalar 0.42 (both), toe joints pinned in both variants; the
  trunk, shoulder, elbow, forearm and wrist envelopes are unchanged (ratio 1.0).
  Sign classes of the leg joints move from INDETERMINATE to decided on many
  frames (for example left hip adduction on 112 of 123 frames, all to NEGATIVE;
  right hip adduction on 92, to POSITIVE). One further frame (345) becomes
  INFEASIBLE with conflict set `H-TORQUE-CAPACITY`, `H-GRF-OBSERVATION`.
- Hand-split conclusions: the anthro model has only the closing-hand wrench
  `lambda` (the lead-hand grip point needed for the couple split is unavailable,
  and the axial split needs the shaft direction in grip axes). The six `lambda`
  envelopes are unchanged by either observation (median width ratio 1.0, no sign
  class change): the hand wrench is already determined independently of how the
  feet share the load. What a lower-body observation changes is the leg joint
  torque envelopes and the foot-load split, not the hand split.
- Honest residuals: data derive from the same recorded state, so the normalised
  disagreement with the balance is at most `2.4e-10` sigma (consistency, not
  independent validation); frame 510 is refused; HiGHS returns `NUMERICAL`
  (not `INFEASIBLE`) on the empty set of frame 342 with plate rows, which the
  library raises rather than recodes, so observed variants skip frames already
  empty without them.

Non-claims: rigid unilateral contacts only; friction coefficient is a declared
hypothesis; the stand-in plates are simulation, not measurement; no
scientific qualification (software correctness only).

## NSX-61 and NSX-62 Ensembles (#12586, #12587)

`kinematic_ensemble(source, n, seed)`; seeded determinism, rejection counting,
envelope union contains every member. NSX-62 samples only along unidentified
directions inside the pseudo-inertia constraint; identified combinations stay
fixed (see the NSX-62 section below).

