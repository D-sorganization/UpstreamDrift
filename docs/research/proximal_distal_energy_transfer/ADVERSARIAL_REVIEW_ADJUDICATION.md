# Adversarial Review Adjudication

## Review Identity and Method

- **Review:** _Comprehensive Technical & Adversarial Review: Proximal-Distal
  Energy Transfer Project_
- **Received:** 2026-08-11
- **SHA-256:**
  `981F89C143158903A221C5E21AAC2B3041457853F5946ED78CF21A07289102C0`
- **Tracking Epic:**
  [#8499](https://github.com/D-sorganization/UpstreamDrift/issues/8499)
- **Adjudication Rule:** A criticism changes the scientific product only when
  its factual premise is reproduced against the current source or when it
  exposes a claim whose evidence does not support its scope. Stale findings
  are documented rather than silently accepted, and open research extensions
  are not represented as completed validation.

The review assessed an earlier and narrower paper surface. The current resource
already contains forward same-state killswitches, moving-base and flexible-club
models, a coupled modal shaft, reduced spatial inverse and forward tiers, a
12-input coupled uncertainty screen, and explicit non-human claim boundaries.
Those later tiers are considered in every disposition below.

## Finding-by-Finding Disposition

| Review Finding                                                                               | Disposition                                                                    | Verification and Result                                                                                                                                                                                                                                      | Implemented Response                                                                                                                                                                                                                      |
| -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Early/late trapezoidal masks omit the boundary interval                                      | **Confirmed defect**                                                           | A synthetic constant signal showed that independently masked trapezoids did not sum to the full integral.                                                                                                                                                    | `_split_integrals` inserts one linearly interpolated split sample into both phase domains. Unit tests require exact full-interval recovery. All affected data were regenerated.                                                           |
| COM acceleration is obtained by differentiating sampled velocity                             | **Confirmed numerical weakness**                                               | The old implementation used `np.gradient`; exact joint accelerations are available from the forward dynamics.                                                                                                                                                | Added analytic rigid-link COM acceleration and exact kinetic/mechanical energy rates. Pointwise interface-power closure is now tested to numerical precision.                                                                             |
| The 2.0 rad arm-angle rule silently discards 29 of 92 trials and may bias the winner         | **Transparency concern confirmed; winner-bias allegation not supported**       | At the registered bound, 63 trials are accepted and 29 first crossings lie outside the registered delivery zone. Across 1.5--2.5 rad bounds, 58--69 are accepted and the selected winner at both shoulder torques is unchanged.                              | Every sweep row now retains the unfiltered first crossing and a reason-coded status. E1c reports attempted and accepted counts for every threshold. The bound is described as an estimand boundary, not an anatomical validity threshold. |
| A 92-program grid is presented as a continuous optimum                                       | **Claim-boundary criticism confirmed**                                         | The experiment is a finite heuristic comparison and does not solve a variational or direct-collocation problem.                                                                                                                                              | Replaced global-optimum language with `grid-selected`, `highest among tested`, and equivalent bounded terms. A constrained smooth optimal-control comparator remains an explicit research priority.                                       |
| Instantaneous torque steps are physiological                                                 | **Claim-boundary criticism confirmed**                                         | Step commands are model inputs, not muscle excitation or activation histories.                                                                                                                                                                               | Added E1e, which filters all post-preload command transitions with 20, 35, and 50 ms first-order time constants. The tested ordering persists, but the result is labeled command-filter sensitivity, not physiology or optimal control.   |
| Negative early wrist work proves active human retention                                      | **Over-interpretation confirmed**                                              | The sign follows directly from the imposed actuator torque opposing the modeled opening velocity.                                                                                                                                                            | The abstract, results, discussion, and conclusions now identify this as modeled actuator work only and explicitly exclude muscle, neural-intent, eccentric-physiology, and human-technique interpretations.                               |
| Pointwise ZTCF is treated as a future zero-torque trajectory                                 | **Partly valid for residual prose; obsolete as a model-gap claim**             | Pointwise drift is an instantaneous tangent-field decomposition. The current resource also executes a 96-case forward double-pendulum killswitch ensemble and forward same-state killswitches in higher tiers.                                               | Results text now separates the instantaneous vector field from forward futures and points to the matched-state ensemble. No forward-trajectory claim is assigned to pointwise ZTCF alone.                                                 |
| Fixed hub, rigid shaft, and planar motion are omitted limitations                            | **Stale for the current resource; still valid for the primary 2-DOF estimand** | Later executed tiers include mobile-base/flexible and modal-club models, a distributed-shaft reference, reduced 20-DOF spatial cross-formulation dynamics, and native MuJoCo/Pinocchio spatial forward contact. None is subject-calibrated human validation. | The paper retains the simple model as analytical ground truth, reports later tiers separately, and preserves anatomical, equipment-calibration, physiological, and human validation as open gates.                                        |
| One-at-a-time sensitivity is the only robustness analysis                                    | **Stale**                                                                      | The current coupled study varies 12 inputs simultaneously by deterministic Latin-hypercube screening, reports PRCC, identifiability, and held-out strategy tradeoffs.                                                                                        | OAT remains clearly labeled as a local screen. The coupled screen remains an engineering envelope, not a population distribution or Sobol variance decomposition.                                                                         |
| The manuscript invents a novel coaching controversy                                          | **Partly valid framing criticism**                                             | Delayed release is established in the cited historical literature; the finite program labels are not evidence about the prevalence of coaching views.                                                                                                        | Introduction and discussion now call the strategy poles experimental shorthand, disclaim novelty over delayed-release literature, and state that no coaching recommendation follows.                                                      |
| Modern hand-path, shaft, spatial, and sequence-variability literature is absent              | **Stale in its broad form**                                                    | The current manuscript cites and executes dedicated hand-path, shaft, spatial, and uncertainty sections and retains human sequence variability as a limitation on translation.                                                                               | No unsupported numeric shaft-energy range from the review was imported. Existing primary-source links and model-tier boundaries are retained.                                                                                             |
| Mass, Coriolis, gravity, pointwise ZTCF/ZVCF equations, and impact interpolation are correct | **Confirmed strength**                                                         | Equation/code comparison and regression tests found no contradictory evidence.                                                                                                                                                                               | Retained without substantive change; terminology was narrowed where trajectory semantics could be misread.                                                                                                                                |

## New Falsifiability Surface

The remediation adds three reviewer-visible checks:

1. **Accounting falsifier:** exact analytic segment power must close pointwise,
   and early plus late work must equal the full-interval trapezoid.
2. **Selection falsifier:** every attempted program must retain its geometric
   first crossing and status; changing the registered angular bound must expose
   count changes and whether the selected winner changes.
3. **Command-rise falsifier:** the declared strategy ordering must be reported
   for every registered time constant. A reversal is a reported failure, not a
   discarded run.

These checks strengthen the bounded mechanism claim. They do not supply human
validation, a population-optimal strategy, physiological activation, or a
globally optimal control law.

## Reproduction Commands

```bash
python -m scripts.research.proximal_distal_energy.run_experiments
python -m scripts.research.proximal_distal_energy.e1c_impact_sensitivity
python -m scripts.research.proximal_distal_energy.e1e_smooth_command_sensitivity
python -m scripts.research.proximal_distal_energy.make_figures
pytest -q tests/unit/research/test_proximal_distal_energy.py
pytest -q tests/research/test_e1e_smooth_command_sensitivity.py \
  tests/research/test_adversarial_review_remediation.py
```
