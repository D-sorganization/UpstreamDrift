# Comprehensive Open Golf Modeling Research Program

## Program Objective

The proximal-to-distal framework becomes the common language for an open,
engine-neutral program that tests golf-swing and human-motion mechanisms from
analytical mechanics through governed human evidence. The program is managed
by [epic #8557](https://github.com/D-sorganization/UpstreamDrift/issues/8557),
a child of the existing model-completion epic. It is not a promise that the
current theory will survive. A useful outcome may be support, contradiction,
inconclusive evidence, or discovery that a proposed quantity is not
identifiable.

## Program Questions

The next model or experiment must answer a discriminating question:

1. Which forces and couples transfer power, redirect momentum, store elastic
   energy, or merely satisfy a constraint?
2. Which effects arise from achieved state and geometry, which require current
   control, and which depend on passive, contact, shaft, base, or impact terms?
3. Which bilateral hand, joint, muscle, or ground contributions are identifiable
   from the proposed measurements?
4. Which timing or state-triggered programs remain effective under uncertainty,
   delay, phase error, loading limits, and competing performance objectives?
5. Which model predictions survive independent formulations, contact models,
   equipment calibration, subject scaling, and held-out human data?

## Model Ladder and Promotion Gates

| Tier                             | Question Added                                                                                      | Promotion Gate                                                                                                               |
| -------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Analytical Double Pendulum       | Exact drift/control, energy, geometry, and intervention identities                                  | Closed equations, manufactured cases, convergence, and complete work-energy accounting                                       |
| Forward Planar Two-Arm           | Bilateral constraint forces, direct wrist moment, force-generated couple, and internal modes        | Rank/conditioning, no silent regularization, action-reaction, contact power, and killswitch controls                         |
| Moving Base and Distributed Club | Endogenous base motion, calibrated bending/torsion, grip compliance, and recoil                     | Base plus strain-energy closure, modal calibration uncertainty, and discrepancy against a beam reference                     |
| Articulated Spatial Body         | Scapula, shoulder, forearm, wrists, hands, ground contacts, plane evolution, and long-axis rotation | Common named wrenches/twists, two independent engines, contact-model discrepancy, and proper-frame invariance                |
| Neuromusculoskeletal             | Activation delay, muscle-tendon dynamics, redundancy, co-contraction, fatigue, and strength limits  | Subject-scaled paths, identifiability, posterior predictive checks, and no inference of muscle action from net moments alone |
| Club-Ball-Flight                 | Strike location, impact impulse, face/path, equipment, and aerodynamic uncertainty                  | Momentum/energy loss accounting, calibrated impact parameters, and launch/flight validation                                  |
| Human and Population             | Participant-specific and hierarchical variability, prospective falsification, and transportability  | Governed synchronized data, participant holdout, preregistered outcomes, and preserved null/contradictory findings           |

Added fidelity is justified only when it reduces a registered discrepancy or
makes a new hypothesis testable. Degrees of freedom alone do not constitute
progress.

## Biomechanics Program

The source and model map must cover pelvis-thorax coupling, scapulothoracic and
glenohumeral motion, forearm rotation, multi-axis wrist motion, bilateral grip
compliance, lower-limb and ground pathways, and the club as a distributed
structure. Muscle-tendon work must separate activation, contractile, series
elastic, passive, and joint-level quantities. Anthropometric, skill, sex, age,
injury, handedness, impairment, equipment, and task variation enter as declared
domains or stratifiers, not hidden residual variance.

Every biological statement must name the measurement that could distinguish it
from alternatives. Candidate modalities include synchronized motion capture,
force plates, instrumented bilateral grip wrenches, shaft sensing, EMG with
electromechanical-delay treatment, ultrasound where tissue behavior is central,
club/ball impact measurement, and launch-monitor outcomes. Sensor bandwidth,
coordinate calibration, soft-tissue artifact, filtering, synchronization,
missingness, and inverse-dynamics sensitivity propagate to the claim level.

## Nonlinear Dynamics and Control Program

The swing is treated as a hybrid constrained nonlinear system with uncertain
events, unilateral contacts, impacts, actuator states, and state-dependent
geometry. The program evaluates observability, controllability,
identifiability, constraint singularities, internal-force null spaces,
finite-time stability, phase sensitivity, basins, and bifurcations where the
mathematics is applicable.

Open-loop timing, state-triggered policies, impedance control, robust model
predictive control, stochastic or optimal feedback control, differential
dynamic programming, and risk-sensitive formulations are compared under the
same actuator and state constraints. Speed, face/path, strike, balance,
loading, effort proxy, consistency, and robustness remain separate objectives.
The output is a Pareto set and failure map, not a universal optimum.

Parameter estimation uses structural and practical identifiability checks
before fitting. Bayesian or simulation-based inference requires prior and
posterior predictive checks, held-out participants or trajectories, explicit
model discrepancy, and sensitivity to event and measurement models.

## Dependency on the Side Task

[Tools #4142](https://github.com/D-sorganization/Tools/issues/4142) owns the
reusable ensemble-variation, typed outcome, quiet-zone, and global-sensitivity
authority. This program does not copy that logic. Completion requires protected
merge of its implementation, deterministic replay across worker counts,
retention of misses and failures as scientific outcomes, adequacy and method
assumption reporting, matched desktop/web behavior, an immutable UpstreamDrift
pin, and cross-repository parity fixtures.

The torso-velocity side study is complete: UpstreamDrift
[#8555](https://github.com/D-sorganization/UpstreamDrift/issues/8555) merged
through PR [#8577](https://github.com/D-sorganization/UpstreamDrift/pull/8577)
at commit `967c40f54cc03f8cae89cde09268d62771d220fe`, which remains an
ancestor of remote `main`. It found no universal relation among torso rate,
delivery speed, and braking work; matching-rule choice and nonmonotonic
torso/arm/wrist killswitch effects are retained as adverse evidence.

Human validation [#8556](https://github.com/D-sorganization/UpstreamDrift/issues/8556)
is intentionally open at an external acquisition boundary. Its
participant-held-out registration, null and adverse-load tests, sensitivities,
and fail-closed intake controls are implemented, but no qualifying governed
participant dataset with synchronized bilateral six-axis grip wrenches was
found in the workspace or public-data search. Published instrumented-grip
studies such as [Choi and Park](https://mdpi-res.com/d_attachment/sensors/sensors-20-03672/article_deploy/sensors-20-03672-v2.pdf)
and [Koike](https://ojs.ub.uni-konstanz.de/cpa/article/download/6828/6125)
inform acquisition design but are not participant-level deposits satisfying
the frozen contract. Synthetic traces and paper-level summary curves are
prohibited substitutes. The companion GUI is tracked by Tools
[#4430](https://github.com/D-sorganization/Tools/issues/4430), stacked after
consolidated PR [#4450](https://github.com/D-sorganization/Tools/pull/4450).
PR #4411 was intentionally closed as superseded;
unchanged CI must not be redundantly rerun.

## Research Collection Review

The Biomechanics and Nonlinear Control NotebookLM collections are reviewed for
contrary evidence, disputed definitions, missing mechanisms, measurement
limitations, model classes, and decisive experiments. Each collection receives
a source manifest and coverage note. Notebook output remains a lead; every
change to the paper or bibliography is supported by an independently checked
original source. The initial 2026-08-12 live review is explicitly pending
because the local profile failed network token validation. The 2026-08-14
retry again redirected to manual Google authentication; no credentials or
authentication dialogs were automated. Repository evidence and independently
checked original sources therefore remain the current authority.

## Delivery Milestones

The photographed momentum-transfer agenda is governed by
[`MOMENTUM_TRANSFER_QUESTION_PROGRAM.md`](MOMENTUM_TRANSFER_QUESTION_PROGRAM.md)
and its machine-readable registry. It requires separate answers for drift
attribution, geometry, timing and casting, timing demand, closed-loop
robustness, proximal-velocity dose response, and typed slack. Those questions
remain subject to the model and human-evidence boundaries below.

1. **Audit Infrastructure:** deterministic candidate inventory, strict claim
   schema, release reconciliation, link/source checks, and figure-data checks.
2. **Paper Adjudication:** every material claim classified and reviewed; every
   number and figure regenerated; weak claims narrowed, relabeled, or removed.
3. **Research Map:** source manifests, coverage gaps, competing hypotheses,
   model-to-measurement map, and preregistered experiment matrix.
4. **Reusable Uncertainty Authority:** Tools #4142 merged, pinned, and parity
   verified.
5. **Articulated Spatial Model:** calibrated grip and club, whole-body contact,
   closed-contact inverse kinematics, joint-limit and collision checks,
   independent-engine comparison, and discrepancy report. The first
   subject-scaled atlas is complete and rejects the current prescribed common
   states as anatomical contact configurations: hand-to-grip error is
   0.171--0.616 m despite full local constraint rank.
6. **Neuromuscular and Control Models:** activation, redundancy,
   identifiability, robust control, and held-out simulation evidence.
7. **Human Falsification:** acquire governed synchronized bilateral six-axis
   grip-wrench data for #8556, freeze the participant holdout before outcomes,
   run preregistered null/adverse tests, and retain null or contradictory
   results. Literature-only evidence and synthetic dry runs do not satisfy this
   milestone.
8. **Open Release:** tutorials, reviewer surfaces, qualified manifests,
   protected merges, visual QA, free artifacts, and archival persistent ID.

Each milestone updates the repository handoff, falsification matrix, evidence
schema, release qualification, and AffineDrift's pinned review surface. The
GitHub epic is the scheduling authority; this document defines the durable
scientific contract.

## Completed Paper-Wide Claim Audit

The current audit adjudicates all 982 narrative candidates against 261
atomic claim contracts; no candidate remains unreviewed. Repeated methods,
summary, limitation, and release passages point back to their primary claim
instead of acquiring stronger authority through repetition. The final pass
added missing primary records for the rotating-base torso experiment, the
isolated synthetic beam experiment, and the exploratory-interface/open-release
boundary.

The bilateral-wrench extension adds a sensor-level falsifiability result. Two
separated three-axis point forces map to net club wrench with rank five and one
invisible equal-and-opposite axial mode. Full bilateral six-axis hand wrenches
map to the same six-component net wrench with nullity six. One independently
measured axial scalar closes only the point-force rank gap; direct full
allocation still requires bilateral sensing. MT-E07 now adds a deterministic
trajectory-level synthetic point-force qualification under normalized noise,
cross-talk calibration error, and contact-center migration. It demonstrates
that net-wrench closure can hide large allocation error and that calibration
and contact tracking are part of the measurement contract. Traceable bilateral
six-axis device calibration, distributed contact, subject scaling, and held-out
human qualification remain open.

Completion of the paper audit is not completion of the research program. In
particular, the audit rejected quantitative small-deflection shaft inference
for the current coupled baseline, retained only bounded synthetic structural
results, found scale-dependent practical rank in the allocation/transmission
map, and preserves #8556 as an external human-data acquisition gate. Future
milestones must create new claim records and falsifiers before new conclusions
enter summaries or conclusions.
