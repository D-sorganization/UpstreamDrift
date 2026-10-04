# ZTCF-Anchored Kinematic Matching With Bounded Local Torque Bands

Governing epic: [#11421](https://github.com/D-sorganization/UpstreamDrift/issues/11421)
(DIME). Slices: DIME-04 [#11425](https://github.com/D-sorganization/UpstreamDrift/issues/11425)
(drift prediction), DIME-16 [#11437](https://github.com/D-sorganization/UpstreamDrift/issues/11437)
(feasibility and missing data), DIME-09 [#11430](https://github.com/D-sorganization/UpstreamDrift/issues/11430)
(continuous replay, partial).

Status: software implementation on the planar double-pendulum reference model.
**Not** a capture, engine or contact qualification. Synthetic truth is truth of
the simulator. The calculation is registered as an inventory blocker in
`manuals/upstreamdrift/calculation-registry.json`
(`UP-D1-dime-drift-anchored-matching-inventory`) until manual coverage lands in
QMD.

## The Owner's Strategy, Mapped to Code

| Owner note (2026-10-04 strategy page)                                             | Implementation                                                                                         |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| ZTCF is a good predictor of the next increment                                    | `drift_prediction.linearize_drift` (reuses `ztcf_zvcf.ztcf_acceleration`)                              |
| ZTCF + constant torque (same as prior) is a better estimate                       | every window warm-starts from the previous torque held constant (`q_constant_torque`)                  |
| Pick the probable next state, weighted against data                               | robust Gauss-Newton window fit; weighting comes from noise model + posterior, not a hand-tuned blend   |
| Recursive movement through data to compile the force profile                      | `drift_anchored_matcher.match_kinematics` marches overlapping windows, carrying state and torque       |
| Enforce continuity of profiles and assemble                                       | torque-rate limit (`ControlBand.rate_limit`) + tapered inverse-variance overlay of local solutions     |
| Weight of ZTCF vs ZVCF/input should vary with velocity (ZTCF high at high speed)  | `drift_dominance_index`, reported per sample; verified to rise with speed                              |
| ZTCF with ZVCF shows the range of viable options; rule out all else               | per-sample plausible band from admissible torques; observations outside it are gated out and labelled |

## Mathematical Contract

Smooth, contact-free mode, local coordinates (`len(q) == len(v)`):

```math
a = f(q, v) + B(q)\,\tau, \qquad f = M^{-1}(-h(q, v)), \qquad B = M^{-1} S^{T}.
```

`f` is the pointwise ZTCF drift; `B tau` is the control (input) channel. The
admissible torque set is a box intersected with a rate limit around the
previous torque:

```math
\tau_k \in [\underline\tau, \overline\tau] \cap [\tau_{k-1} - \dot\tau_{\max}\Delta t,\ \tau_{k-1} + \dot\tau_{\max}\Delta t].
```

**One-step reachable acceleration** (exact axis-aligned hull of the parallelotope):

```math
a \in f + B\tau_c \pm |B|\,\tau_h, \quad \tau_c = \tfrac12(\overline\tau_k + \underline\tau_k),\ \tau_h = \tfrac12(\overline\tau_k - \underline\tau_k).
```

**Uncertain-control prediction** (declared Gaussian control, first order):

```math
x^+ = A x + G\tau,\quad G = \begin{bmatrix}\tfrac12\Delta t^2 B\\ \Delta t B\end{bmatrix},\quad
\operatorname{Cov}(x^+) = A P A^{T} + G\Sigma_u G^{T}.
```

Zero mean control is an initialisation with declared covariance, never a claim
of inactivity.

**Drift dominance** compares drift with control *authority*, not with the
realised total acceleration (which can cancel to zero):

```math
D = \frac{\lVert f\rVert}{\lVert f\rVert + \lVert |B|\,\tau_h^{\text{global}}\rVert} \in [0, 1].
```

### Local Window Problem

For a window of `S = W + 1` samples starting from a carried estimate
`(q_0, v_0)` the decision vector is `z = [dq_0, dv_0, c]`, with `c` the torque
knots (hat basis, `K` knots). Positions come from RK4 forward dynamics with
zero-order-hold torque. The objective is

```math
\min_{z}\ \sum_{j\in\mathcal O} w_j \left\lVert \frac{q_j(z) - y_j}{\sigma}\right\rVert^2
 + \left\lVert \frac{dq_0}{\sigma_{q_0}}\right\rVert^2 + \left\lVert \frac{dv_0}{\sigma_{v_0}}\right\rVert^2
 \ \text{s.t. } c \in \text{band},
```

solved by Gauss-Newton with bounded linear least squares (`lsq_linear`,
BVLS) on forward-difference sensitivities.

1. **Anchor:** `c` starts at the previous torque (ZTCF + constant torque).
2. **Gate:** linearised plausible band
   `q_j(z) + S_c (c_mid - c) +/- (|S_c| c_half + k |S_x| sigma_x + k sigma)`;
   observations outside it start with weight 0 (ruled out).
3. **Robust reweighting:** Huber weights (`k = 2`) with a hard reject (`d > 6`).
4. **Report:** fitted `q, v, tau`, posterior `tau_std`, zero-torque branch
   `q_ztcf`, ZTCF divergence, explained fraction
   `1 - sum w r^2 / sum w d_ztcf^2`, band, inlier `chi2_per_dof`, unweighted
   `raw_rms_residual`, and `saturated` (global box) vs `rate_limited`
   (continuity) flags.

### Recursion, Gap Bridging and Overlay

* Windows advance by `stride`; each inherits the previous fitted state and
  torque (fixed carried arrival stds, an approximation pending DIME-07).
* A window with fewer than `min_observed` samples is **extended** until it has
  data on both sides of a gap (cap `4 W`; knots scale with length), so an
  occlusion is filled by a dynamics-consistent boundary fit, not by forward
  extrapolation.
* A failed window falls back to drift + constant torque and the next window
  **re-acquires** with a loose configuration prior.
* Torque is assembled as the Hann-tapered, inverse-variance overlay of the
  local solutions. `tau_disagreement` is their weighted spread; `tau_std` is
  the weighted mean posterior std (not combined as independent, because
  windows share data).

### Sample Labels

| Label         | Meaning                                                                           |
| ------------- | --------------------------------------------------------------------------------- |
| `accepted`    | observed and explained by bounded dynamics                                        |
| `outlier`     | rejected by the majority of covering windows, run length `<= max_outlier_run`     |
| `unexplained` | longer rejected run: motion no admissible torque reaches (slip, occluder, model) |
| `gap_filled`  | missing; estimated from dynamics only                                             |

## Measured Software Behaviour (Reference Model)

Synthetic swing from `estimation/synthetic_swing.py`: 0.4 s, 201 samples at
2 ms, torque peaks 200/40 N m. Band +/-300/100 N m, rate 4000/2000 N m/s,
window 12 steps, stride 4, linear knots. Single run each (seeded); these are
software regression numbers, not population statistics.

| Case                                  | Torque RMS (N m)  | Estimate vs raw (rad) | Outlier recall / FP | Gap RMS (rad) | Open-loop replay RMS (rad) |
| ------------------------------------- | ----------------- | --------------------- | ------------------- | ------------- | -------------------------- |
| Clean (sigma 1e-4 rad)                | 2.2 / 0.5         | <1e-4 vs 1e-4         | n/a                 | n/a           | 0.0015                     |
| Noise (sigma 2e-3 rad)                | 16.8 / 3.6        | 6e-4 vs 1.9e-3        | n/a                 | n/a           | 0.038                      |
| Noise + 3 % spikes + 15-sample gap    | 18.8 / 3.9        | 9e-4 vs 2.4e-2        | 1.00 / 0.00         | 0.0024        | 0.069                      |

A 0.4 rad smooth marker slip over 20 samples is labelled `unexplained` rather
than absorbed into torque. Runtime is about 14 s per match on one CPU core
(pure Python finite differences; DIME-14 owns acceleration).

## Findings That Change the Plan

1. **Short-window torque is weakly identifiable.** Over 24 ms a start-velocity
   correction and a constant torque produce nearly collinear position changes.
   With `sigma_v0 = 0.5 rad/s` the posterior torque std is about 19 N m. Pinning
   the start state through the recursion drops it to about 6 N m. Torque
   accuracy therefore comes from the *chain plus overlay*, and windows must
   report posterior spread (they do), never a bare point estimate.
2. **Robust inlier chi-square hides model failure.** When the needed torque is
   out of band, robust weights reject the unexplainable samples and inlier
   chi-square looks benign. The unweighted residual and run-length labels are
   required to tell noise from unreachable motion.
3. **Rate-limit activity is not saturation.** Hitting the continuity limit is
   the regulariser working; only the global actuator box signals unreasonable
   inputs.
4. **Open-loop replay drifts with noisy torque.** Per-window fits are
   locally consistent, but integrating 8 % torque noise for 0.4 s gives
   0.04-0.07 rad replay error. A final whole-trajectory replay refinement
   (single uninterrupted forward simulation, overlay as prior) is required
   before a forward-dynamics match is claimed.
5. **Gaps must be bridged, not extrapolated.** Forward-only fallback through a
   15-sample occlusion walked 0.05 rad off and every later window gated out
   all data. Extending windows across the gap fixed both.

## Limits

* Contact, closed chains and stance are refused (`contact_active=True`); the
  constrained #10286 providers are required (DIME-06).
* Quaternion/floating-base coordinates are refused; a manifold retraction is
  required.
* The plausible band is a first-order envelope around the fit, not a certified
  reachable set; very wide bands over long windows linearise poorly.
* Carried arrival stds are fixed approximations, not marginalised arrival
  information (DIME-07).
* No capture data, no native engine beyond the analytic ODE backend, and no
  runtime claim.

## Reproduction

```bash
MPLBACKEND=Agg python3 -m pytest tests/unit/estimation/test_dime_drift_prediction.py \
  tests/unit/estimation/test_dime_local_window.py \
  tests/unit/estimation/test_dime_drift_anchored_matcher.py \
  tests/unit/estimation/test_synthetic_swing.py -q --timeout=60
```
