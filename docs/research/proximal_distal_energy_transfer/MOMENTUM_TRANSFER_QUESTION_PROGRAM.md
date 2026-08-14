# Momentum Transfer Question Program

## Purpose

This register translates the photographed momentum-transfer agenda into
questions that can be answered, contradicted, or left unresolved by declared
models and governed measurements. It does not treat a higher clubhead speed,
an appealing animation, or a familiar coaching description as proof of a
mechanism.

The executable companion is
[`data/momentum_transfer_question_registry.json`](data/momentum_transfer_question_registry.json).
Every question has a current evidence boundary, a next model experiment, a
measurement requirement, and a result that would contradict the proposed
mechanism.

The implementation-level registration is
[`data/momentum_transfer_experiment_registry.json`](data/momentum_transfer_experiment_registry.json).
It freezes seven model experiments and one participant-held-out human stage,
including interventions, controls, outcomes, uncertainty axes, falsifiers,
required data, and present execution status. A status describes what is
available; it is not a favorable-result label.

## Current Answer Map

| ID  | Critical Question                                                                                                      | Present Answer                                                                                                                                                                                                                                      | Evidence Already Available                                                                                                                                                                    | Material Boundary                                                                                                                                                | Decisive Path Forward                                                                                                                                                                                                                                  |
| --- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Q1  | How much of transfer is drift?                                                                                         | **Conditionally answered.** Drift share is defined only for a named observable, coordinate/frame, event window, model, and decomposition. Signed shares can be negative or exceed 100% when drift and control oppose.                               | Exact pointwise closure, event-window impulse/work, force-power traces, and forward killswitch persistence across declared synthetic tiers.                                                   | There is no meaningful universal percentage for “the swing,” a golfer, or every joint. Human inverse dynamics will remain convention- and uncertainty-dependent. | Publish a common estimand table across tiers; repeat with articulated spatial contact; propagate measurement uncertainty; test participant-held-out bilateral-wrench data.                                                                             |
| Q2  | What are the geometry dependencies?                                                                                    | **Conditionally answered through the spatial mechanism tier.** Force–velocity and relative-link angles, signed grip separation, transverse differential force, reference point, and frame-explicit wrench transport gate sign, zero, and magnitude. | Executable analytical atlas; exact orthogonal/coincident/axial nulls; force and moment-arm reversals; moving-base controls; independently authored MuJoCo/Pinocchio spatial-contact controls. | No subject-scaled arm/scapular geometry, calibrated grip compliance, distributed 3-D shaft, participant distribution, or bilateral-wrench evaluation.            | Extend the same frozen gates and controls through subject-scaled articulated arms, compliant grips/equipment, held-out digital twins, and governed human trials.                                                                                       |
| Q3  | What timing patterns and common flaws matter, including casting, weak early proximal acceleration, and release timing? | **Conditionally answered.** Early distal release can alter the available geometry and pathway; proximal rate, acceleration, and braking have nonmonotonic effects. No universal optimal release time is supported.                                  | Release-program comparisons, forward killswitch ensembles, state/clock triggers, rotating-base torso-rate grid, work and braking metrics.                                                     | “Casting” lacks one universal measurement definition; the current tiers do not establish a human error taxonomy or causal coaching prescription.                 | Register casting by distal-angle/rate and event criteria; intervene independently on early proximal acceleration, distal release, and braking; match work and delivery state; retain null/adverse results.                                             |
| Q4  | Does passive or drift-mediated transfer require less timing precision?                                                 | **Unresolved for humans.** A delayed/noisy observer screen found no recovery advantage over the clock comparator; this does not estimate viable timing volume or neural timing demand.                                                              | Paired clock/state/impedance/observer policies, deterministic sensor noise, matched perturbations, lower-tail and adverse-cost outputs.                                                       | No phase-grid volume, identified neural delay, impact outcome, or participant variability.                                                                       | Sweep viable timing regions and phase error with matched work, force, and effort; then execute participant-held-out perturbations.                                                                                                                     |
| Q5  | Are any strategies self-correcting or more robust to noise?                                                            | **Not established.** Sustained half-error recovery occurred in only 13–20% of the 15 engineering cases and no policy separated decisively.                                                                                                          | Trajectory error, recovery time, viable-set return, delayed/noisy observation, hand force, and effort under common random numbers.                                                            | Small planar envelope, simple observer, no attraction-region boundary or adverse external load, and no human perturbation data.                                  | Expand the attraction-region and load sweeps, identify observer dynamics, and require recovery benefit across held-out disturbances without hidden force/load cost.                                                                                    |
| Q6  | Must proximal velocity be maximized to maximize transfer?                                                              | **No general rule is supported.** Higher torso rate can accompany higher delivery speed, but does not consistently reduce braking work, and driver killswitch effects are nonmonotonic.                                                             | Rotating-base torso-velocity study, matched-rate grids, torso/arm/wrist killswitches, work and grip-load outcomes.                                                                            | Planar torso surrogate, no impact ball outcome, no anatomical capacity or subject calibration.                                                                   | Estimate a dose–response surface for proximal rate and acceleration while matching work, state, contact load, and impact definition; locate interior optima and reversals across subjects/models.                                                      |
| Q7  | Is slack beneficial, harmful, or required?                                                                             | **Unanswered until slack is typed.** Contact disengagement, backlash, compliant preload, muscle–tendon series compliance, and control deadband/co-contraction are different states and cannot share one conclusion.                                 | Preload/dead-zone sensitivity, compliant shaft and contact models, reduced muscle–tendon history bridge, explicit warning against unspecified “slack.”                                        | No unified slack state vector, calibrated transmission/contact properties, reflex model, or human measurement.                                                   | Introduce one slack class at a time with energy/passivity accounting; sweep onset, magnitude, and hysteresis; test contact loss, backlash, tendon compliance, and activation deadband separately; estimate each from governed data where identifiable. |

## Required Definitions

### Drift Attribution

Every reported fraction must identify its numerator and denominator. Report at
least total, drift, and control values for force, impulse, power, work, and the
relevant generalized acceleration. Pair signed shares with magnitude-normalized
shares and an explicit cancellation indicator. Pointwise attribution is not a
forward future; forward persistence requires a same-state intervention.

### Casting

“Casting” is experimental shorthand, not a diagnosis. A study must declare its
criterion before outcomes, for example an early threshold crossing of relative
club angle, distal angular rate, or wrist-to-club release while the proximal
segment remains inside a declared event window. Multiple criteria must be
reported separately and their disagreement retained.

### Timing Demand

Timing demand is measured by outcome sensitivity to event-time error, the
width or volume of the viable timing region, lower-tail performance, and
robustness to observation delay and phase-estimation error. It is not inferred
from whether a controller is labeled passive or state triggered.

### Self-Correction

A strategy is self-correcting only with a declared feedback/physical mechanism
and a perturbation-recovery result. Required observables include error decay,
recovery time, probability of returning to the viable set, attraction-region
size, and force/load cost. Open-loop repeatability alone is not
self-correction.

### Slack Classes

1. **Contact Slack:** temporary loss or partial loss of a hand–grip constraint.
2. **Transmission Slack:** backlash or a dead zone before force transmission.
3. **Structural Slack:** compliant preload and elastic series deformation.
4. **Biological Series Compliance:** muscle–tendon state and stored energy.
5. **Control Slack:** activation deadband, delay, or co-contraction reserve.

Each class requires its own state, constitutive law, energy accounting, and
measurement strategy. “Slack is good” and “slack is bad” are prohibited global
claims.

## Experiment Sequence

### Stage 1: Analytical and Planar Identifiability

- Recompute Q1 estimands for every common observable and event window.
- Sweep relative angles, moment arms, grip separation, and velocity direction
  with null and sign-reversing geometry controls.
- Intervene independently on proximal acceleration, distal release, and
  proximal braking while matching initial state, input work, and impact event.
- Introduce each slack class separately; reject energy creation and closure
  residuals before interpreting performance.

### Stage 2: Robust and Nonlinear Control

- Compare clock, state, impedance, and delayed observer policies using common
  random numbers.
- Perturb phase, state estimate, torque capacity, contact stiffness, shaft
  properties, and external load on distinct training and held-out ensembles.
- Report viable-region volume, recovery, lower-tail delivery, orientation,
  hand force, tissue-load proxies, and effort separately.
- Search for interior optima and reversals rather than assuming monotonic
  benefit from proximal speed or drift share.

### Stage 3: Articulated Spatial and Equipment Models

- Repeat the geometry and killswitch registry in independently authored
  engines with articulated arms, calibrated two-hand contact, and a
  distributed bending/torsion club.
- Require common state, wrench, power, event, and model-identity records.
- Treat cross-engine disagreement as a result and execute a discrepancy plan.

### Stage 4: Subject-Scaled Neuromusculoskeletal Models

- Estimate feasible muscle/activation families rather than one inferred
  recruitment solution.
- Separate series compliance, passive tissue, reflex delay, and contact slack.
- Carry parameter and structural uncertainty into every proposed strategy.

### Stage 5: Governed Human Falsification

- Acquire synchronized full-body and club kinematics, force plates, bilateral
  six-axis grip wrenches, grip pressure/contact state, EMG where governed, and
  impact/ball outcomes.
- Freeze participant-held-out splits and all event/estimand definitions before
  outcome inspection.
- Execute null geometry, timing-shuffle, adverse-load, and alternative-model
  sensitivity tests.
- Preserve contradicted and inconclusive outcomes. Synthetic evidence cannot
  close this stage.

## Repository Ownership

- [#8595](https://github.com/D-sorganization/UpstreamDrift/issues/8595) owns
  this question program and registry.
- [#8448](https://github.com/D-sorganization/UpstreamDrift/issues/8448) owns
  the mechanism and model-fidelity ladder.
- [#8449](https://github.com/D-sorganization/UpstreamDrift/issues/8449) and
  [#8551](https://github.com/D-sorganization/UpstreamDrift/issues/8551) own
  geometry, timing, proximal-rate, and robust-control interventions.
- [#8450](https://github.com/D-sorganization/UpstreamDrift/issues/8450) and
  [#8556](https://github.com/D-sorganization/UpstreamDrift/issues/8556) own
  governed human validation.
- [#8592](https://github.com/D-sorganization/UpstreamDrift/issues/8592) and
  [#8593](https://github.com/D-sorganization/UpstreamDrift/issues/8593) own
  subject-scaled models, participant digital twins, and population ensembles.
- [#8557](https://github.com/D-sorganization/UpstreamDrift/issues/8557) remains
  the scheduling and scientific-governance authority.
