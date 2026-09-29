# Advanced Expansion Review

## Review Outcome

The project now spans analytical, planar forward, two-hand constrained,
moving-base, modal-shaft, spatial common-state, reduced spatial forward-contact,
uncertainty/control, arm--wrist allocation, ground-reaction, reference-frame,
and reduced biological tiers. The principal opportunity is no longer adding
unconnected model complexity. It is establishing registered equivalence and
failure boundaries while moving from generalized moments to measured human
mechanisms.

## Implemented in Epic 8505

1. **Visual communication:** a consistent phase-resolved motion plate and four
   reviewer-facing diagrams for frames, muscle redundancy, activation history,
   and engine roles, each released as vector PDF and SVG.
2. **Mathematical depth:** explicit wrench/twist ordering, reference-point
   transport, invariant power, Jacobian virtual work, muscle moment-arm mapping,
   null-space redundancy, activation dynamics, Hill-type force, and series-force
   dynamics.
3. **Executable evidence:** deterministic frame and virtual-work closure,
   forty-one matched-moment activation allocations, two continuous preparation
   histories, and five canonical-pose adapter round trips.
4. **Terminology:** a normative contract separates sequencing from transfer,
   drift from passivity, torque from power, generalized controls from anatomy,
   and specific transmission states from an unspecified use of “slack.”
5. **Cross-publication integration:** the monograph and AffineDrift article now
   link the mechanism to motion language, force/torque, affine structure,
   constraints, parallel mechanisms, triple pendulums, distributed control,
   inverse dynamics, induced acceleration, screw theory, rotations, and
   reference-point transport.

## Highest-Value Next Experiments

| Priority | Executable Program                                        | Decisive Observable                                             | Failure Exposure                               |
| -------- | --------------------------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------- |
| 1        | Canonical dynamic-state replay in every available backend | Event-aligned state, wrench, moment, power, and residual traces | Convention and runtime discrepancies           |
| 2        | Articulated two-arm MuJoCo/Pinocchio contact model        | Same-state driver killswitch and grip wrench                    | Carriage-model dependence                      |
| 3        | Subject-scaled OpenSim feasibility families               | Moment arms, feasible muscle forces, residual actuators         | Anatomical and recruitment non-identifiability |
| 4        | Matched MyoSuite preparation histories                    | Excitation, activation, force, contact, and joint moment        | Reduced activation-model dependence            |
| 5        | Drake multi-objective trajectory families                 | Speed, orientation, effort, tissue-load, robustness Pareto set  | Objective dependence and false optimality      |
| 6        | Calibrated distributed shaft plus bilateral grip          | Shaft strain, grip wrench, pathway power                        | Rigid-club and contact simplification          |
| 7        | Participant-held-out synchronized experiment              | Bilateral wrench, pressure, motion, shaft, EMG, uncertainty     | Human generalization and causal overreach      |

## Visualization Roadmap

- Animate the same canonical pose and event clock in each engine with identical
  camera, axes, segment colors, force scale, and reference-point marker.
- Add force-vector small multiples at preparation, transition, maximum proximal
  speed, maximum distal acceleration, and delivery.
- Pair every animation with quantitative traces and a downloadable evidence
  record; appearance alone is not verification.
- Add uncertainty ribbons, residual panels, and negative-control branches to
  advanced figures before adding decorative realism.
- Use subject geometry only after its provenance, scaling method, and privacy
  boundary are explicit.

## Completion Boundary

Epic 8505 completes the review, convention, reduced-biology, graphical, and
pose-interchange foundation. It does not complete subject-scaled anatomy,
five-engine dynamics parity, governed human acquisition, or external archival
publication. Those are deliberately falsifiable next tiers rather than claims
hidden behind a more realistic rendering.
