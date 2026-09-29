# Adversarial Transmission Review and Path Forward

## Review Standard

This review treats every implication as unproven until an intervention and an
observable discriminate it from plausible alternatives. It separates five
questions that are often collapsed in kinetic-chain discussions:

1. Did mechanical energy reach the club?
2. Through which modeled pathway did it cross the chosen boundary?
3. Did a declared intervention change that pathway?
4. Did the resulting controller reject declared perturbations?
5. Can a human realize the controller repeatably, safely, and across contexts?

Evidence for an earlier question does not prove a later one. In particular,
kinematic peak order does not identify an energy pathway; torque sign does not
determine power sign; pointwise drift is not a forward future; and nominal
speed is not stability.

## Severity-Ranked Findings

The machine-readable register is in
[`data/transmission_robustness_study.json`](data/transmission_robustness_study.json).
The most consequential findings are:

- **Critical — human translation remains untested.** No current model proves a
  universal technique, passive human negative torque, injury reduction, or a
  population optimum.
- **Critical — speed and repeatability are different objectives.** The fastest
  nominal program in the new study is not the least variable under held-out
  perturbations.
- **Critical — sign language can mislead.** Negative torque can deliver
  positive power when angular velocity is also negative. Torque, velocity,
  power, and work must be shown together.
- **High — pathway identity is nonunique from kinematics.** Similar segmental
  sequencing can coexist with different actuator, constraint-force, elastic,
  gravity, and dissipative work.
- **High — the impact estimand is incomplete.** A fixed terminal state and a
  planar face/path proxy do not include strike location, dynamic loft, attack
  angle, restitution, or ball flight.
- **High — biological stability is not an impedance scalar.** Coactivation can
  reject displacement while increasing force and effort, and delayed noisy
  feedback can destabilize an otherwise stable perfect-state controller.
- **High — engineering envelopes are not population distributions.** Latin
  hypercube samples test a declared box; they do not estimate golfer
  prevalence or individualized likelihood.
- **Medium — movement variability is task-relative.** Variability along a local
  task-null direction may be useful, while smaller joint variability can still
  produce worse club outcomes.

## Path Forward

| Priority | Question                                                 | Required Implementation                                                                        | Decisive Evidence                                                                       |
| -------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| P0       | Does the transfer survive a true impact event?           | Couple 3-D club state to calibrated impact and ball-flight models.                             | Held-out carry and dispersion retain or reject the model-strategy ordering.             |
| P0       | Are hand-force and wrist-moment pathways identifiable?   | Synchronize bilateral six-axis grip wrenches, club motion, force plates, and inverse dynamics. | Wrench, power, work, and residual closure distinguish competing pathway models.         |
| P0       | Is apparent stability biological?                        | Apply phase-registered perturbations with EMG and time-varying impedance identification.       | Recovery improves without unacceptable force, effort, accuracy, or injury-risk proxies. |
| P1       | Does state triggering remain robust with sensing limits? | Add observer delay, phase-estimation error, noise, and actuator saturation.                    | Closed-loop outcome amplification remains bounded across held-out disturbances.         |
| P1       | Is variability structured rather than merely small?      | Perform participant-level UCM/task-Jacobian covariance analyses.                               | Task-null variance exceeds task-relevant variance near the delivery manifold.           |
| P1       | Which strategy generalizes to whom?                      | Fit hierarchical subject, equipment, intent, and session distributions.                        | Participant-held-out posterior predictions calibrate speed, dispersion, and loads.      |
| P2       | Are local null directions valid at useful amplitudes?    | Add nonlinear manifold curvature and second-order outcome maps.                                | Linear predictions remain within preregistered error or are explicitly retired.         |

The practical implication is a measurement strategy, not a coaching command:
optimize the lower tail of the complete impact outcome while constraining face,
path, strike, load, and effort; permit task-null coordination variability; and
prefer event/state variables only when their sensing and actuation delays are
explicitly modeled and experimentally validated.
