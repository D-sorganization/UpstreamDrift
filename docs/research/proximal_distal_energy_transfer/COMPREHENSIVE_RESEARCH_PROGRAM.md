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

## Independently Checked Biomechanics Leads

The following original sources sharpen the next tests without being treated as
confirmation of the proposed transfer mechanism:

| Source                                                                        | What It Contributes                                                                                                                            | Registered Use and Boundary                                                                                                                                                                                                       |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Seth et al. (2016)](https://doi.org/10.1371/journal.pone.0141028)            | An OpenSim shoulder complex with a four-degree-of-freedom scapula constrained on an ellipsoidal thorax and comparison with bone-pin kinematics | Use as the first articulated scapulothoracic implementation candidate; repeat contact closure, moment-arm reversal, and killswitch studies. Its kinematic agreement does not identify golf muscle force or a preferred technique. |
| [Verikas et al. (2016)](https://doi.org/10.3390/s16040592)                    | Bilateral forearm, rhomboid, and trapezius EMG onset and peak measurements in elite golfers                                                    | Use to define candidate channels and synchronization bandwidth for MT-H01. The small observational sample and surface EMG do not identify bilateral grip wrench or causal muscle contribution.                                    |
| [Silva et al. (2013)](https://doi.org/10.1016/j.jelekin.2013.05.007)          | Demonstrates that golf-swing EMG onset depends on the selected baseline, threshold, and muscle                                                 | Preregister onset-definition sensitivity and prohibit a single threshold from deciding preactivation, passive resistance, or role reversal.                                                                                       |
| [Mizoguchi and Yoneyama (2005)](https://doi.org/10.1299/jsmesports.2005.0_35) | Distributed radial, tangential, and axial grip-force measurements across fingers and palm                                                      | Add pressure/contact-distribution sensing to the bilateral six-axis wrench plan and test whether point-contact reduction hides internal modes. A proceedings paper is not a governed participant dataset.                         |

The corresponding acquisition minimum is synchronized motion, ground reaction,
club/shaft state, bilateral six-axis grip wrenches, distributed grip pressure,
and bilateral forearm/scapular EMG. The analysis must report at least two EMG
onset definitions, electromechanical-delay sensitivity, contact-center
migration, and participant-held-out prediction. These additions can distinguish
several proposed mechanisms; they still cannot make activation, intent, or
tissue slack uniquely identifiable without an explicit model and adverse
controls.

## Critical-Question Completion Map

| Handwritten Question                                     | Current Answer                                                                                                                                        | Decisive Completion Path                                                                                                                                      |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| How much is drift?                                       | Exactly defined and computed at declared planar states and windows; no model-independent or human fraction exists.                                    | Repeat the work/impulse attribution in articulated spatial forward dynamics and held-out participants with uncertainty intervals and cancellation reporting.  |
| What are the geometry dependencies?                      | Moment arm, force--velocity projection, reference point, grip span, and constraint conditioning can change magnitude or sign in current tiers.        | Subject-specific scapula/arms, calibrated grip/contact, distributed shaft, null and sign-reversal controls in two engines.                                    |
| What is the timing of momentum flow?                     | Phase-resolved model windows exist; clock versus state-trigger results are conditional and no sustained recovery was observed in 60 registered cases. | Independent proximal acceleration, braking, and distal-release interventions with common-phase event definitions, observer delay, and held-out perturbations. |
| What constitutes casting or early body deceleration?     | Definition-dependent model events are registered; no unique physiological event has been established.                                                 | Preregister competing event definitions and require agreement or report disagreement against motion, wrench, shaft, and impact measurements.                  |
| Does passive drift reduce timing demand or self-correct? | Not established; the registered screen did not show sustained half-error recovery.                                                                    | Estimate attraction/recovery regions under delay, saturation, contact loads, and subject scaling, then test the frozen human endpoint.                        |
| Does maximizing proximal velocity maximize transfer?     | No universal rule is supported; rate effects are nonmonotonic and matching-rule dependent.                                                            | Full-delivery-state matching with speed, braking work, load, face/path, strike, and robustness Pareto outcomes.                                               |
| Is slack useful or necessary?                            | There is no global answer; five distinct slack classes are separated and intentional slack remains unidentifiable.                                    | Embed each class separately in higher-order models, measure the corresponding state, and test benefit, harm, and null regions without cross-class inference.  |

This map distinguishes a bounded model answer from project completion. A row is
complete only when its registered model, measurement, uncertainty, negative
control, and participant-holdout gates pass; narrative coverage alone is not a
completion criterion.

## Delivery Milestones

The photographed momentum-transfer agenda is governed by
[`MOMENTUM_TRANSFER_QUESTION_PROGRAM.md`](MOMENTUM_TRANSFER_QUESTION_PROGRAM.md)
and its machine-readable registry. It requires separate answers for drift
attribution, geometry, timing and casting, timing demand, closed-loop
robustness, proximal-velocity dose response, and typed slack. Those questions
remain subject to the model and human-evidence boundaries below.

1. **Audit Infrastructure:** deterministic candidate inventory, strict claim
   schema, resolvable `path:line` checks, hash-pinned local support, external-URL
   inventory, release reconciliation, link/source checks, and figure-data
   checks.
2. **Paper Adjudication:** every material claim classified and reviewed; every
   number and figure regenerated; weak claims narrowed, relabeled, or removed.
3. **Research Map:** source manifests, coverage gaps, competing hypotheses,
   model-to-measurement map, and preregistered experiment matrix.
4. **Reusable Uncertainty Authority:** Tools #4142 merged, pinned, and parity
   verified.
5. **Articulated Spatial Model:** calibrated grip and club, whole-body contact,
   executed closed-contact inverse kinematics, subject-specific joint-limit and
   mesh-level collision replacement,
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

## Completed Candidate Census and Ongoing Release Review

The narrative census adjudicates all 994 paper candidates against 266 atomic
claim contracts; no candidate remains unreviewed. That completion status
applies to candidate coverage, not to scientific closure of every release
claim. Ten of the 31 public release claims remain pending or in progress, and
the validator now reports that open count explicitly. Repeated methods,
summary, limitation, and release passages point back to their primary claim
instead of acquiring stronger authority through repetition.

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

The first paired scapulothoracic intervention is now executed as MT-E09. With
trunk and club pose fixed, fixed shoulder centers close none of 54 states; the
bounded scapula-on-ellipsoid surrogate reaches residual tolerance in 31 and
passes the separate optimizer-termination gate in 16. Twenty-eight states
activate a screening bound, the maximum shoulder-center excursion is 0.101 m,
and the retained 2.0 m adverse span fails. Both contact Jacobians remain full
row rank while local coordinate nullity increases from two to ten. This is a
structural result and an identifiability warning, not an anatomical or coaching
result. The next milestone replaces the surrogate with a validated articulated
shoulder and calibrated forward grip contact.

Completion of the candidate census is not completion of release review or the
research program. In
particular, the audit rejected quantitative small-deflection shaft inference
for the current coupled baseline, retained only bounded synthetic structural
results, found scale-dependent practical rank in the allocation/transmission
map, and preserves #8556 as an external human-data acquisition gate. Future
milestones must create new claim records and falsifiers before new conclusions
enter summaries or conclusions.

The separate claim-evidence integrity manifest covers all 1,639 evidence
references. It hash-pins 200 distinct repository artifacts and inventories 85
external URLs. Hash agreement establishes content identity, not independence
or correctness; URL inventory establishes traceability, not availability or
scientific validity.
