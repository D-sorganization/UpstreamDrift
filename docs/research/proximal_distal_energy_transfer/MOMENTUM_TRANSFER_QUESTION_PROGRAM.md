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

| ID  | Critical Question                                                                                                      | Present Answer                                                                                                                                                                                                                                                                                                                                                   | Evidence Already Available                                                                                                                                                                    | Material Boundary                                                                                                                                                                             | Decisive Path Forward                                                                                                                                                                                                                                    |
| --- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Q1  | How much of transfer is drift?                                                                                         | **Conditionally answered.** Drift share is defined only for a named observable, coordinate/frame, event window, model, and decomposition. Signed shares can be negative or exceed 100% when drift and control oppose.                                                                                                                                            | Exact pointwise closure, event-window impulse/work, force-power traces, and forward killswitch persistence across declared synthetic tiers.                                                   | There is no meaningful universal percentage for “the swing,” a golfer, or every joint. Human inverse dynamics will remain convention- and uncertainty-dependent.                              | Publish a common estimand table across tiers; repeat with articulated spatial contact; propagate measurement uncertainty; test participant-held-out bilateral-wrench data.                                                                               |
| Q2  | What are the geometry dependencies?                                                                                    | **Conditionally answered through the spatial mechanism tier.** Force–velocity and relative-link angles, signed grip separation, transverse differential force, reference point, and frame-explicit wrench transport gate sign, zero, and magnitude.                                                                                                              | Executable analytical atlas; exact orthogonal/coincident/axial nulls; force and moment-arm reversals; moving-base controls; independently authored MuJoCo/Pinocchio spatial-contact controls. | No subject-scaled arm/scapular geometry, calibrated grip compliance, distributed 3-D shaft, participant distribution, or bilateral-wrench evaluation.                                         | Extend the same frozen gates and controls through subject-scaled articulated arms, compliant grips/equipment, held-out digital twins, and governed human trials.                                                                                         |
| Q3  | What timing patterns and common flaws matter, including casting, weak early proximal acceleration, and release timing? | **Conditionally answered.** Early distal release can alter the available geometry and pathway; proximal rate, acceleration, and braking have nonmonotonic effects. No universal optimal release time is supported.                                                                                                                                               | Release-program comparisons, forward killswitch ensembles, state/clock triggers, rotating-base torso-rate grid, work and braking metrics.                                                     | “Casting” lacks one universal measurement definition; the current tiers do not establish a human error taxonomy or causal coaching prescription.                                              | Register casting by distal-angle/rate and event criteria; intervene independently on early proximal acceleration, distal release, and braking; match work and delivery state; retain null/adverse results.                                               |
| Q4  | Does passive or drift-mediated transfer require less timing precision?                                                 | **Not supported in the registered planar comparison; unresolved for humans.** Across six named load cohorts, the clock policy retained 4/5 robust task-viable phase points and 45 ms of contiguous sampled width, versus 1/5 and no multi-point span for the delayed/noisy state trigger.                                                                        | Common nominal phase map, paired references and perturbations, deterministic observer noise, and shared speed, face/path, force, effort, and numerical guards.                                | Five sampled phase points, local nominal map, delivery rather than impact, engineering loads, and no identified human observer or participant variability.                                    | Repeat with continuous phase sampling, independently identified observers, spatial impact, subject scaling, and participant-held-out perturbations.                                                                                                      |
| Q5  | Are any strategies self-correcting or more robust to noise?                                                            | **Not established.** The earlier screen produced 13–20% sustained recovery; the expanded 60-case phase/load comparison produced none for either policy.                                                                                                                                                                                                          | Trajectory error, common-phase timing, sustained viable-set return, delayed/noisy observation, adverse equipment/actuator loads, hand force, effort, and half-step checks.                    | The perturbations and observer remain engineering surrogates; the study does not identify a continuous attraction boundary, external-contact disturbance response, or human correction.       | Estimate continuous attraction regions with stronger and independently identified feedback families, external-contact disturbances, spatial impact, and participant holdout.                                                                             |
| Q6  | Must proximal velocity be maximized to maximize transfer?                                                              | **No general rule is supported.** Exact equal-energy rate sweeps are nonmonotonic; a 216-program forward screen yields 46 independent work- and load-matched pairs with both favorable and adverse higher-rate outcomes; the identical-state acceleration response also reverses before impact.                                                                  | Three rate-matching rules, 45 acceleration interventions, full actuator-work ledgers, 46 disjoint work/load matches, rotating-base study, killswitches, braking and load outcomes.            | The forward pairs differ in actuator commands and use greedy finite-grid matching; acceleration is pointwise; no ball outcome, anatomical capacity, subject calibration, or human validation. | Add full-delivery-state matching and independent rate/acceleration interventions; then locate interior optima and reversals across spatial, subject-scaled, and held-out human tiers.                                                                    |
| Q7  | Is slack beneficial, harmful, or required?                                                                             | **Partly answered at a synthetic constitutive tier.** Five declared classes have been exercised separately under two excitations with passivity, closure, delayed-control, and local-sensitivity audits. No global benefit, necessity, intentionality, or delivery advantage is supported; one scalar output nearly confounds contact and biological surrogates. | One-class-at-a-time dynamic constitutive suite, mechanical passivity and closure, delayed control boundary, local sensitivity, and pairwise output-separation audit.                          | Synthetic scalar channels, memoryless backlash, reduced biological compliance, no forward delivery outcome, no calibrated properties, and no human measurement.                               | Embed one class at a time in moving-base, two-hand, and spatial delivery models under matched state/work/load; add stateful play and subject-specific tissue; acquire independent contact, shaft, tendon, activation, and bilateral-wrench measurements. |

## Source-Agenda Readiness

The reviewed handwritten agenda resolves to nine points because the timing
bullet contains three distinct questions. The generated
[`data/momentum_transfer_readiness_audit.json`](data/momentum_transfer_readiness_audit.json)
is the completion authority for this transcription and planning layer.

- Eight points have a bounded model answer, partial answer, or a supported
  rejection of a general maximization rule.
- One point remains unresolved as a single construct: casting. Human timing
  demand, self-correction, delivery effects, and mechanism identification
  remain open at higher evidence tiers even where reduced models provide
  bounded negative or partial answers.
- Every point maps to at least one model experiment, a decisive falsifier, and
  the participant-held-out human stage.
- Human execution remains blocked because no qualifying governed dataset with
  synchronized bilateral six-axis grip wrenches is available.

Run the readiness gate with:

```text
python -m scripts.research.proximal_distal_energy.momentum_question_readiness build
python -m scripts.research.proximal_distal_energy.momentum_question_readiness validate
```

The validator fails if any of the nine source points is omitted, if a point is
linked to an experiment that does not cover its parent question, or if an
answer, decisive test, falsifier, data gate, or human stage is absent.

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
